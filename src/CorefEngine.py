# AUTHOR: Chris Tanner (christanner@cs.brown.edu) 
# PURPOSE: CRETE: Coreference Resolution for EnTities and Events (uses ECB+ corpus)
import params
import time
import pickle
import random
import sys
import os
import fnmatch
import numpy as np

from collections import defaultdict
from KBPParser import KBPParser
from ECBParser import ECBParser
from HDDCRPParser import HDDCRPParser
from ECBHelper import ECBHelper
from StanParser import StanParser
from FeatureHandler import FeatureHandler
from DataHandler import DataHandler
from FFNN import FFNN
from CCNN import CCNN
#from LibSVM import LibSVM
#from HDF5Reader import HDF5Reader
from sklearn import svm

class CorefEngine:

	# TODO:
	# X - Q2: which ones do HDDCRP include?
	# Q3B: which ones do Stanford include? (NER)
	# Q3A: which ones do Stanford include? (coref only)
	# Q3C: which ones do Stanford include? (coref only+NER)
	# - how many of these does Stanford contain? (for Ent+Events) -- TRAIN/DEV
	#	- vary the cutoff point (N=inf, 10,9,8,7,6,5,4,3,2,1) -- TRAIN/DEV
	# - make 2 new Gold files (CoNLL format) which includes
	#     entity information: (1) all Ent+Events; (2) Ent+Events and remove sindgletons
	#      (3) Ents (minus pronouns)+Events
	# measure performance on entities, events, entities+events:
	# - (1) how well does our system (CCNN+AGG) do on:
	#	 (A) test on all and hope our system doesn't mix ents and events
	#    (B) test on non-events
	#    (C) test on non-events non-pronouns
 	# - (2) how well does StanCoreNLP do on:
	#	 (A) test on all and hope our system doesn't mix ents and events
	#    (B) test on non-events

	if __name__ == "__main__":

		runStanford = False
		supp_features_type = "one" # {none, shortest, one}
		# classifier params
		numRuns = 10
		useCCNN = True
		devMode = False
		cd_scope = "dir" # {dir, dirHalf}
		useRelationalFeatures = False
		#wdPresets = [256, 3, 2, 16, 0.0]
		wdPresets = [64, 20, 2, 32, 0.0] # batchsize, num epochs, num layers, num filters, dropout

		wd_stopping_points = [0.51] #, 0.401, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.501, 0.51, 0.52, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.601]
		cd_stopping_points = [0.5]

		# handles passed-in args
		args = params.setCorefEngineParams()

		if args.useECBTest:
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
		# parses elmo embeddings
		#elmo = HDF5Reader('/Users/christanner/research/CRETE/data/features/alloutput3.hdf5')
		#exit(1)
		# most functionality lives here
		helper = ECBHelper(args)

		# parses the real, actual corpus (ECB's XML files)
		ecb_parser = ECBParser(args, helper)
		corpus = ecb_parser.parseCorpus(helper.docToVerifiedSentences)
		
		helper.addECBCorpus(corpus)

		# parses the HDDCRP Mentions
		if not args.useECBTest:
			hddcrp_parser = HDDCRPParser(args)
			# returns a list of tokens for *every* HDDCRP mention, even if it's not in validSentences
			allHDDCRPMentionTokens = hddcrp_parser.parseCorpus(args.hddcrpFullFile)
			helper.createHDDCRPMentions(allHDDCRPMentionTokens) # constructs correct Mentions
		
		# loads Stanford's parse
		if runStanford:
			stan = StanParser(args, corpus)
			helper.addStanfordAnnotations(stan)
			helper.saveStanTokens()
		else:
			helper.loadStanTokens()
		helper.createStanMentions()
		helper.printCorpusStats()
		#helper.printHDDCRPMentionCoverage()
		#corpus.checkMentions()

		# DEFINES WHICH MENTIONS TO USE
		trainXUIDs = set()
		devXUIDs = set()
		testXUIDs = set()
		for m in corpus.ecb_mentions:
			if not m.isPred:
				continue
			if m.dir_num in helper.trainingDirs:
				trainXUIDs.add(m.XUID)
			elif m.dir_num in helper.devDirs:
				devXUIDs.add(m.XUID)
			elif args.useECBTest and m.dir_num in helper.testingDirs:
				testXUIDs.add(m.XUID)

		# conditionally add HDDCRP Mentions (as the test set)
		if not args.useECBTest:
			for xuid in corpus.HMUIDToMention:
				testXUIDs.add(xuid)
		
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
		
		dh = DataHandler(helper, trainXUIDs, devXUIDs, testXUIDs)
		helper.addDependenciesToMentions(dh)
		#helper.checkDependencyRelations()
		#corpus.calculateEntEnvAgreement()
		
		# within-doc first, then cross-doc
		if useCCNN:
			# DEV-WD
			'''
			print("***** DEV SET ******")
			wd_model = CCNN(helper, dh, useRelationalFeatures, "doc", wdPresets, None, True) # True means use DEVSET
			(wd_docPreds, wd_pred, wd_gold, sp_wd) = wd_model.train_and_test_wd(10) # 1 means only
			
			cd_model = CCNN(helper, dh, useRelationalFeatures, cd_scope, wdPresets, wd_docPreds, True)  # True means use DEV SET
			(cd_docPreds, cd_pred, cd_gold, sp_cd) = cd_model.train_and_test_cd(3)

			print("\t** BEST DEV-WD stopping points:", sp_wd,"and",sp_cd)
			'''

			# WITHIN DOC
			ensemble_predictions = []
			while ensemble_predictions == [] or len(ensemble_predictions[0]) < numRuns:
				wd_model = CCNN(helper, dh, supp_features_type, "dir", wdPresets, None, devMode, wd_stopping_points)
				#wd_model = CCNN(helper, dh, supp_features_type, "doc", wdPresets, None, devMode, wd_stopping_points)
				dirs, ids, preds, golds, best_f1 = wd_model.train_and_test()
				if best_f1 > 0.5:
					helper.addEnsemblePredictions(False, dirs, ids, preds, ensemble_predictions) # True means WD
					#helper.addEnsemblePredictions(True, dirs, ids, preds, ensemble_predictions) # True means WD
					print("len(ensemble_predictions[0]):", str(len(ensemble_predictions[0])))
			preds = helper.getEnsemblePreds(ensemble_predictions) # normalizes them
			(f1, prec, rec, bestThreshold) = helper.evaluatePairwisePreds(preds, golds)
			print("[*** ENSEMBLE CCNN BEST PAIRWISE TEST RESULTS] f1:", round(f1,4), " prec: ", round(prec,4), " recall: ", round(rec,4), " threshold: ", round(bestThreshold,3))
			
			# saves WITHIN-DOC PREDS
			# CROSS DOC
			#wd_docPreds = pickle.load(open("hddcrp_clusters_ONLY_EVENTS_wd_0.51_9.p", 'rb'))
			#cd_model = CCNN(helper, dh, supp_features_type, cd_scope, wdPresets, None, devMode, cd_stopping_points)
			#cd_model = CCNN(helper, dh, supp_features_type, cd_scope, wdPresets, wd_docPreds, devMode, sp_cd)
			#cd_model.train_and_test()
		else:
			wd_model = FFNN(helper, dh)
			wd_model.train_and_test_wd(numRuns)
		
		print("* done.  took ", str((time.time() - start_time)), "seconds")