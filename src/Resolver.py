import params
import time
import pickle
import random
import sys
import os
import fnmatch
import numpy as np
from KBPParser import KBPParser
from ECBParser import ECBParser
from HDDCRPParser import HDDCRPParser
from ECBHelper import ECBHelper
from StanParser import StanParser
from FeatureHandler import FeatureHandler
from DataHandler import DataHandler
from FFNN import FFNN
from CCNN import CCNN
class Resolver:
	def __init__(self, args, presets, scope, ids=None, preds=None):
		self.args = args
		self.scope = scope
		self.presets = presets
		self.ids = ids
		self.preds = preds

		if self.scope != "doc" and self.scope != "dir":
			print("* ERROR: invalid scope!  must be doc or dir, for WD or CD, respectively")
			exit(1)

	def resolve(self, mention_type, supp_features_type, event_pronouns, entity_pronouns, num_runs):
		# supp_features_type  = {none, shortest, one, type}

		# classifier params
		useCCNN = True
		devMode = False
		runStanford = False
		useRelationalFeatures = False

		stopping_points = [0.51] #, 0.401, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.501, 0.51, 0.52, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.601]

		if self.args.useECBTest:
			f_suffix = "ecb"
		else:
			f_suffix = "hddcrp"
		wordFeaturesFile = "../data/features/" + str(f_suffix) + "/word.f"
		lemmaFeaturesFile = "../data/features/" + str(f_suffix) + "/lemma.f"
		charFeaturesFile = "../data/features/" + str(f_suffix) + "/char.f"
		posFeaturesFile = "../data/features/" + str(f_suffix) + "/pos.f"
		dependencyFeaturesFile = "../data/features/" + str(f_suffix) + "/dependency.f"
		wordnetFeaturesFile = "../data/features/" + str(f_suffix) + "/wordnet.f"
		bowFeaturesFile = "../data/features/" + str(f_suffix) + "/bow.f"

		# handles passed-in args
		'''
		testing this w/ auto login
		args = params.setCorefEngineParams()
		kbp_parser = KBPParser(args, "../data/KBP/")
		kbp_parser.parseCorpus()
		exit(1) testing this
		'''
		start_time = time.time()
		helper = ECBHelper(self.args, event_pronouns, entity_pronouns)

		# parses the real, actual corpus (ECB's XML files)
		ecb_parser = ECBParser(self.args, helper)
		corpus = ecb_parser.parseCorpus(helper.docToVerifiedSentences)

		helper.addECBCorpus(corpus)

		#if self.ids != None:
		# adds the predictions from a past model run
		helper.addPredictions(self.ids, self.preds)

		#helper.printCorpus("corpusMentions.txt")

		# parses the HDDCRP Mentions
		if not self.args.useECBTest:
			hddcrp_parser = HDDCRPParser(self.args)
			# returns a list of tokens for *every* HDDCRP mention, even if it's not in validSentences
			allHDDCRPMentionTokens = hddcrp_parser.parseCorpus(self.args.hddcrpFullFile)
			helper.createHDDCRPMentions(allHDDCRPMentionTokens) # constructs correct Mentions
		
		# loads Stanford's parse
		if runStanford:
			stan = StanParser(self.args, corpus)
			helper.addStanfordAnnotations(stan)
			helper.saveStanTokens()
		else:
			helper.loadStanTokens()
		helper.createStanMentions()
		helper.printCorpusStats()

		#helper.printHDDCRPMentionCoverage()
		#corpus.checkMentions()

		'''
		# only used for saving features
		fh = FeatureHandler(args, helper) #, trainXUIDs, devXUIDs, testXUIDs)
		fh.saveLemmaFeatures(lemmaFeaturesFile)
		fh.saveCharFeatures(charFeaturesFile)
		exit(1)
		fh.savePOSFeatures(posFeaturesFile)
		fh.saveDependencyFeatures(dependencyFeaturesFile)
		fh.saveWordNetFeatures(wordnetFeaturesFile)
		fh.saveBoWFeatures(bowFeaturesFile)
		'''

		# DEFINES WHICH MENTIONS TO USE
		trainXUIDs, devXUIDs, testXUIDs = helper.getCorpusMentions(mention_type)
		dh = DataHandler(helper, trainXUIDs, devXUIDs, testXUIDs)
		helper.addDependenciesToMentions(dh)
		
		#helper.checkDependencyRelations()
		#corpus.calculateEntEnvAgreement()
		
		# within-doc first, then cross-doc
		if useCCNN:
			# DEV-WD
			'''
			print("***** DEV SET ******")
			wd_model = CCNN(helper, dh, useRelationalFeatures, "doc", presets, None, True) # True means use DEVSET
			(wd_docPreds, wd_pred, wd_gold, sp_wd) = wd_model.train_and_test_wd(10) # 1 means only
			
			cd_model = CCNN(helper, dh, useRelationalFeatures, cd_scope, presets, wd_docPreds, True)  # True means use DEV SET
			(cd_docPreds, cd_pred, cd_gold, sp_cd) = cd_model.train_and_test_cd(3)

			print("\t** BEST DEV-WD stopping points:", sp_wd,"and",sp_cd)
			'''
			ensemble_predictions = []
			while ensemble_predictions == [] or len(ensemble_predictions[0]) < num_runs:

				# self.scope == doc or dir (WD or CD)
				model = CCNN(helper, dh, supp_features_type, self.scope, self.presets, None, devMode, stopping_points)
				dirs, ids, preds, golds, best_f1 = model.train_and_test()
				if best_f1 > 0.4:
					if self.scope == "doc": # WD
						helper.addEnsemblePredictions(True, dirs, ids, preds, ensemble_predictions) # True means WD
						print("DOING WITHIN-DOC ENSEMBLE!")
					elif self.scope == "dir": # CD
						helper.addEnsemblePredictions(False, dirs, ids, preds, ensemble_predictions) # False means CD
						print("DOING CROSS-DOC ENSEMBLE!")
					else:
						print("** ERROR: invalid scope.  should be doc or dir")
						exit(1)
					
					print("len(ensemble_predictions[0]):", str(len(ensemble_predictions[0])))

			preds = helper.getEnsemblePreds(ensemble_predictions) # normalizes them
			(f1, prec, rec, bestThreshold) = helper.evaluatePairwisePreds(ids, preds, golds)
			print("[***",mention_type,"ENSEMBLE CCNN BEST PAIRWISE TEST RESULTS] f1:", round(f1,4), " prec: ", round(prec,4), " recall: ", round(rec,4), " threshold: ", round(bestThreshold,3))
			print("* done.  took ", str((time.time() - start_time)), "seconds")
			return ids, preds, golds