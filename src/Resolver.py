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
		devMode = True
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

		# only used for saving features
		#fh = FeatureHandler(self.args, helper) #, trainXUIDs, devXUIDs, testXUIDs)
		#fh.saveLemmaFeatures(lemmaFeaturesFile)
		#fh.saveWordFeatures(wordFeaturesFile)
		#fh.saveCharFeatures(charFeaturesFile)
		#fh.savePOSFeatures(posFeaturesFile)
		#fh.saveDependencyFeatures(dependencyFeaturesFile)
		#fh.saveWordNetFeatures(wordnetFeaturesFile)
		#fh.saveBoWFeatures(bowFeaturesFile)
		#exit(1)

		# DEFINES WHICH MENTIONS TO USE
		trainXUIDs, devXUIDs, testXUIDs = helper.getCorpusMentions(mention_type)
		dh = DataHandler(helper, trainXUIDs, devXUIDs, testXUIDs)
		helper.addDependenciesToMentions(dh)
		#helper.printCorpus("corpusMentions.txt")
		
		#print("tmp_xuidpair_event_entity:", dh.tmp_xuidpair_event_entity)
	
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
			ensemble_dev_predictions = []
			ensemble_test_predictions = []

			dev_best_f1s = []
			while ensemble_test_predictions == [] or len(ensemble_test_predictions[0]) < num_runs:

				# self.scope == doc or dir (WD or CD)
				model = CCNN(helper, dh, supp_features_type, self.scope, self.presets, None, devMode, stopping_points)
				
				# this is only for printing the mention-tree-links we care about
				'''
				print("#xuid_pairs_that_meet_criterion:", len(dh.xuid_pairs_that_meet_criterion))
				for _ in range(len(dh.xuid_pairs_that_meet_criterion)):
					(xuid1, xuid2) = dh.xuid_pairs_that_meet_criterion[_]
					print("\nboth events have 1-hops to two dobj and nsubj:", str(_))
					sentenceTokenToMention = defaultdict(lambda: defaultdict(set))
					m1 = corpus.EUIDToMention[xuid1]
					m2 = corpus.EUIDToMention[xuid2]

					if m1.doc_id != m2.doc_id:
						print("* ERROR: diff docs")
						exit(1)
					
					for euid in corpus.doc_idToDocs[m1.doc_id].EUIDs:
						m = corpus.EUIDToMention[euid]
						sentNum = m.globalSentenceNum
						for t in m.tokens:
							sentenceTokenToMention[sentNum][t].add(m)

					stanTokenToECBTokens = defaultdict(set)
					for t in corpus.doc_idToDocs[m1.doc_id].tokens:
						for stan in t.stanTokens:
							stanTokenToECBTokens[stan].add(t)

					first_token = None
					sent1_text = ""
					sentNum1 = m1.globalSentenceNum
					for t in corpus.globalSentenceNumToTokens[sentNum1]:
						if t.tokenID == "-1":
							continue
						bestStan = dh.getBestStanToken(t.stanTokens)
						for pl in bestStan.parentLinks[helper.dependency_parse_type]:
							parentToken = pl.parent
							if parentToken.isRoot:
								first_token = parentToken
						sent1_text += t.text + " "
					#print("\t** m1:", m1, "from sentence:", sent1_text)
					#helper.dfs_tree(first_token, [], [], "", sentenceTokenToMention[sentNum1], stanTokenToECBTokens)

					sentNum2 = m2.globalSentenceNum
					sent2_text = ""
					for t in corpus.globalSentenceNumToTokens[sentNum2]:
						if t.tokenID == "-1":
							continue
						bestStan = dh.getBestStanToken(t.stanTokens)
						for pl in bestStan.parentLinks[helper.dependency_parse_type]:
							parentToken = pl.parent
							if parentToken.isRoot:
								first_token = parentToken
						sent2_text += t.text + " "
					#print("\t*** m2:", m2, "from sentence:", sent2_text)
					#helper.dfs_tree(first_token, [], [], "", sentenceTokenToMention[sentNum2], stanTokenToECBTokens)
				'''


				results = model.train_and_test()

				dev_dirs, dev_ids, dev_preds, dev_golds, dev_best_f1 = results[0]
				test_dirs, test_ids, test_preds, test_golds, test_best_f1 = results[1]

				
				if dev_best_f1 > 0.4:
					is_wd = True
					'''
					if self.scope == "doc": # WD
						print("DOING WITHIN-DOC ENSEMBLE!")
					elif self.scope == "dir": # CD
						is_wd = False
						print("DOING CROSS-DOC ENSEMBLE!")
					else:
						print("** ERROR: invalid scope.  should be doc or dir")
						exit(1)
					'''
					helper.addEnsemblePredictions(is_wd, dev_dirs, dev_ids, dev_preds, ensemble_dev_predictions)	
					helper.addEnsemblePredictions(is_wd, test_dirs, test_ids, test_preds, ensemble_test_predictions) # True means WD
					dev_best_f1s.append(dev_best_f1)

					#print("lenensemble_test_predictions:", len(ensemble_test_predictions))
					#print("len(ensemble_test_predictions[0]):", str(len(ensemble_test_predictions[0])))
				print("# dev runs:", len(dev_best_f1s), dev_best_f1s)
			#print("# tmp_minipreds:", len(dh.tmp_minipreds))

			dev_preds = helper.getEnsemblePreds(ensemble_dev_predictions) # normalizes them
			(dev_f1, dev_prec, dev_rec, dev_bestThreshold) = helper.evaluatePairwisePreds(dev_ids, dev_preds, dev_golds, dh)
			
			(any_F1, all_F1, cs_f1, l2_f1) = model.baseline_tests()
			print("samelemma_any:", round(any_F1, 4))
			print("samelemma_all:", round(all_F1, 4))
			print("cosine sim:", round(cs_f1, 4))
			print("l2:", round(l2_f1, 4))
			print("CCNN AVERAGE:", round(sum(dev_best_f1s) / float(len(dev_best_f1s)), 4), "(", model.standard_deviation(dev_best_f1s), ")")
			print("CCNN ENSEMBLE:", round(dev_f1, 4))
			#print("[***",mention_type,"ENSEMBLE DEV RESULTS] f1:", round(dev_f1,4), " prec: ", round(dev_prec,4), " recall: ", round(dev_rec,4), " threshold: ", round(dev_bestThreshold,3))

			test_preds = helper.getEnsemblePreds(ensemble_test_predictions) # normalizes them
			(test_f1, test_prec, test_rec, test_bestThreshold) = helper.evaluatePairwisePreds(test_ids, test_preds, test_golds, dh)
			#print("[***",mention_type,"ENSEMBLE TEST RESULTS] f1:", round(test_f1,4), " prec: ", round(test_prec,4), " recall: ", round(test_rec,4), " threshold: ", round(test_bestThreshold,3))
			print("* done.  took ", str((time.time() - start_time)), "seconds")

			return test_ids, test_preds, test_golds
			
	def aggCluster(self, relevant_dirs, event_ids, event_preds, event_golds):
		print("event ids:", event_ids)
		print("event_preds:", event_preds)
		print("event_golds:", event_golds)
		#for sp in self.stopping_points:
		sp = 0.5
		print("* [agg] sp:", sp)

	# agglomerative cluster the within-doc predicted pairs
	#def aggClusterWD(self, relevant_dirs, ids, preds, stoppingPoint):
		#print("** in aggClusterWD(), stoppingPoint:",stoppingPoint)
		start_time = time.time()
		docToXUIDPredictions = defaultdict(lambda: defaultdict(float))
		docToXUIDsFromPredictions = defaultdict(list) # this list is constructed just to ensure it's the same as the corpus'
		for ((xuid1, xuid2), pred) in zip(ids, preds):
			m1 = self.corpus.XUIDToMention[xuid1]
			m2 = self.corpus.XUIDToMention[xuid2]

			if not m1.isPred or not m2.isPred:
				print("* ERROR: we're trying to do AGG on some non-event mentions")
				exit(1)

			pred = pred[0] # NOTE: the lower the score, the more likely they are the same.  it's a dissimilarity score
			doc_id = m1.doc_id
			if m1.dir_num not in relevant_dirs:
				print("* ERROR: passed in predictions which belong to a dir other than what we specify")
				exit(1)
			if m2.doc_id != doc_id:
				print("* ERROR: xuids are from diff docs!")
				exit(1)
			if xuid1 not in docToXUIDsFromPredictions[doc_id]:
				docToXUIDsFromPredictions[doc_id].append(xuid1)
			if xuid2 not in docToXUIDsFromPredictions[doc_id]:
				docToXUIDsFromPredictions[doc_id].append(xuid2)
			docToXUIDPredictions[doc_id][(xuid1, xuid2)] = pred

		ourClusterID = 0
		ourClusterSuperSet = {}
		goldenClusterID = 0
		goldenSuperSet = {}
		
		docToPredClusters = defaultdict(list) # only used for pickling' and later loading WD preds again

		XUIDToDocs = defaultdict(set)

		for doc_id in self.dh.docToXUIDsWeWantToUse.keys():
			dir_num = int(doc_id.split("_")[0])
			if dir_num not in relevant_dirs:
				continue
			#print("-----------\ncurrent doc:",str(doc_id),"\n-----------")
			
			curDoc = self.corpus.doc_idToDocs[doc_id]
			ourDocClusters = {}

			# construct the golden truth for the current doc (events only)
			# we don't need to check if xuid is in our corpus or predictions because Doc.assignECBMention() adds
			# xuids to REFToEUIDs and .XUIDS() -- the latter we checked, so it's all good
			for curREF in curDoc.REFToEUIDs:
				if "entity" not in self.corpus.refToMentionTypes[curREF]:
					goldenSuperSet[goldenClusterID] = set(curDoc.REFToEUIDs[curREF])
					goldenClusterID += 1

			# if our doc only has 1 XUID, then we merely need to:
			# (1) ensure it's an event mention; (2) add a golden cluster; and (3) add 1 single returned cluster
			if len(self.dh.docToXUIDsWeWantToUse[doc_id]) == 1:

				# check if the solo mention is an event or not
				xuid = next(iter(self.dh.docToXUIDsWeWantToUse[doc_id]))
				solo_mention = self.corpus.XUIDToMention[xuid]
				if solo_mention.isPred:

					# count how many REFs are only-events
					num_refs_only_contains_events = 0
					for ref in curDoc.REFToEUIDs:
						if "entity" not in self.corpus.refToMentionTypes[ref]:
							num_refs_only_contains_events += 1

					if num_refs_only_contains_events != 1:
						print("* WARNING: doc:",doc_id,"has only 1 XUID (per DataHandler), but per Corpus has more")
						if self.args.useECBTest:
							print("(this shouldn't happen w/ ECBTest, so we're exiting")
							exit(1)

					ourDocClusters[0] = set([next(iter(self.dh.docToXUIDsWeWantToUse[doc_id]))])

			else: # we have more than 1 XUID for the given doc

				# we check our mentions to the corpus, and we correctly
				# use HDDCRP Mentions if that's what we're working with
				docXUIDsFromCorpus = curDoc.EUIDs
				if not self.args.useECBTest:
					docXUIDsFromCorpus = curDoc.HMUIDs
				
				# ensures our predictions include each of the Doc's mentions
				for xuid in docXUIDsFromCorpus:
					if xuid not in docToXUIDsFromPredictions[doc_id] and self.corpus.XUIDToMention[xuid].isPred:
						print("* ERROR: missing xuid from our predictions")
						exit(1)

				# ensures each of our predicted mentions are valid per our corpus
				for xuid in docToXUIDsFromPredictions[doc_id]:
					if xuid not in docXUIDsFromCorpus:
						print("* ERROR: missing xuid from our corpus")
						exit(1)

				# constructs our base clusters (singletons)
				for i in range(len(docToXUIDsFromPredictions[doc_id])):
					xuid = docToXUIDsFromPredictions[doc_id][i]
					XUIDToDocs[xuid].add(doc_id)
					if len(XUIDToDocs[xuid]) > 1: # check in real-time, as we build it up
						print("* ERROR, we have multiple XUIDs that share the same ID, despite being in diff docs")
						exit(1)
					a = set()
					a.add(xuid)
					ourDocClusters[i] = a

				# the following keeps merging until our shortest distance > stopping threshold,
				# or we have 1 cluster, whichever happens first
				while len(ourDocClusters.keys()) > 1:
					# find best merge, having looked at every pair of clusters
					closestAvgAvgDist = 999999
					closestAvgAvgClusterKeys = (-1,-1)
					i = 0
					for c1 in ourDocClusters.keys():
						j = 0
						for c2 in ourDocClusters.keys():
							if j > i:
								avgavgdists = []
								for xuid1 in ourDocClusters[c1]:
									for xuid2 in ourDocClusters[c2]:
										dist = 99999
										if (xuid1, xuid2) in docToXUIDPredictions[doc_id]:
											dist = docToXUIDPredictions[doc_id][(xuid1, xuid2)]
											avgavgdists.append(dist)
										elif (xuid2, xuid1) in docToXUIDPredictions[doc_id]:
											dist = docToXUIDPredictions[doc_id][(xuid2, xuid1)]
											avgavgdists.append(dist)
										else:
											print("* error, why don't we have either xuid1 or xuid2 in doc_id")
											exit(1)
								avgavgDist = float(sum(avgavgdists)) / float(len(avgavgdists))
								if avgavgDist < closestAvgAvgDist:
									closestAvgAvgDist = avgavgDist
									closestAvgAvgClusterKeys = (c1,c2)
							j += 1
						i += 1
					# print("closestAvgAvgDist is:", str(closestAvgAvgDist),"which is b/w:", str(closestAvgAvgClusterKeys))
					
					# only merge clusters if it's less than our threshold
					if closestAvgAvgDist > stoppingPoint:
						break

					newCluster = set()
					(c1,c2) = closestAvgAvgClusterKeys
					for _ in ourDocClusters[c1]:
						newCluster.add(_)
					for _ in ourDocClusters[c2]:
						newCluster.add(_)
					ourDocClusters.pop(c1, None)
					ourDocClusters.pop(c2, None)
					ourDocClusters[c1] = newCluster

			# end of clustering current doc
			docToPredClusters[doc_id] = ourDocClusters
			#print("ourDocClusters has # keys:", len(ourDocClusters))
			for i in ourDocClusters.keys():
				#print("doc:",doc_id,"adding a cluster")
				ourClusterSuperSet[ourClusterID] = ourDocClusters[i]
				ourClusterID += 1
		print("\tagg took ", str((time.time() - start_time)), "seconds")
		#print("# golden clusters:",str(len(goldenSuperSet.keys())), "; # our clusters:",str(len(ourClusterSuperSet)))
		return (docToPredClusters, ourClusterSuperSet, goldenSuperSet)



		(wd_docPredClusters, wd_predictedClusters, wd_goldenClusters) = self.aggClusterWD(self.helper.devDirs, self.devID, preds, sp)