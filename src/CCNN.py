import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import time
import keras
import pickle
import random
import numpy as np
import tensorflow as tf
import keras.backend as K
from math import sqrt, floor
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, merge, Merge, Flatten, Input, Lambda, Conv2D, AveragePooling2D, MaxPooling2D
from keras.optimizers import Adam
from collections import defaultdict
from get_coref_metrics import get_conll_scores
class CCNN:
	def __init__(self, helper, dh, useRelationalFeatures, scope, presets, wd_docPreds, devMode, stopping_points):
		self.devMode = devMode
		self.helper = helper
		self.dh = dh
		self.corpus = helper.corpus
		self.args = helper.args
		self.wd_pred_clusters = wd_docPreds
		self.scope = scope # used by aggClusterCD()
		self.stopping_points = stopping_points
		self.ensembleDocPairPredictions = defaultdict(lambda: defaultdict(list)) # for ensemble approach
		if presets == []:
			self.bs = self.args.batchSize
			self.ne = self.args.numEpochs
			self.nl = self.args.numLayers
			self.nf = self.args.numFilters
			self.do = self.args.dropout
		else:
			(self.bs, self.ne, self.nl, self.nf, self.do) = presets
		
		print("[ccnn] scope:",self.scope,"bs:",self.bs,"ne:",self.ne,"nl:",self.nl,"nf:",self.nf,"do:",self.do, "dm:",self.devMode, "sp:",self.stopping_points)
		sys.stdout.flush()

		if self.scope != "doc":
			#self.wd_pred_clusters = pickle.load(open("wd_clusters_FULL_dirHalf_812.p", 'rb'))
			self.sanityCheck1()

		self.dh.loadNNData(useRelationalFeatures, True, self.scope) # True means use CCNN
		(self.trainID, self.trainX, self.trainY) = (dh.trainID, dh.trainX, dh.trainY)
		(self.devID, self.devX, self.devY) = (dh.devID, dh.devX, dh.devY)
		(self.testID, self.testX, self.testY) = (dh.testID, dh.testX, dh.testY)
		print("self.testY:", self.testY)

		if self.args.native:
			tf.Session(config=tf.ConfigProto(log_device_placement=True))
			os.environ['CUDA_VISIBLE_DEVICES'] = ''

	# takes the CCNN pairwise predictions and adds to them our list, which will be averaged
	# over all of the runs
	def addEnsemblePredictions(self, relevant_dirs, ids, preds):
		new_preds = []
		for ((xuid1, xuid2), pred) in zip(ids, preds):
			m1 = self.corpus.XUIDToMention[xuid1]
			m2 = self.corpus.XUIDToMention[xuid2]
			# NOTE: the lower the score, the more likely they are the same.  it's a dissimilarity score
			pred = pred[0]
			doc_id = m1.doc_id
			if m1.dir_num not in relevant_dirs:
				print("* ERROR: passed in predictions which belong to a dir other than what we specify")
				exit(1)
			if m2.doc_id != doc_id:
				print("* ERROR: xuids are from diff docs!")
				exit(1)
			self.ensembleDocPairPredictions[doc_id][(xuid1, xuid2)].append(pred)
			thesum = sum(self.ensembleDocPairPredictions[doc_id][(xuid1, xuid2)])
			thelength = len(self.ensembleDocPairPredictions[doc_id][(xuid1, xuid2)])
			new_preds.append([thesum / float(thelength)])
		return new_preds
	# WITHIN-DOC MODEL
	def train_and_test_wd(self, numRuns):
		f1s = []
		recalls = []
		precs = []
		spToCoNLL = defaultdict(list)
		spToPredictedCluster = {}
		spToDocPredictedCluster = {}

		for _ in range(numRuns):
			
			bestRunCoNLL = -1
			bestRunSP = -1

			# define model
			input_shape = self.trainX.shape[2:]
			base_network = self.create_base_network(input_shape)
			input_a = Input(shape=input_shape)
			input_b = Input(shape=input_shape)
			processed_a = base_network(input_a)
			processed_b = base_network(input_b)
			distance = Lambda(self.euclidean_distance, output_shape=self.eucl_dist_output_shape)([processed_a, processed_b])
			model = Model(inputs=[input_a, input_b], outputs=distance)
			model.compile(loss=self.contrastive_loss, optimizer=Adam())
			#print(model.summary())
			model.fit([self.trainX[:, 0], self.trainX[:, 1]], self.trainY, \
				batch_size=self.bs, \
				epochs=self.ne, \
				validation_data=([self.devX[:, 0], self.devX[:, 1]], self.devY))

			if self.devMode:
				preds = model.predict([self.devX[:, 0], self.devX[:, 1]])
			else:
				preds = model.predict([self.testX[:, 0], self.testX[:, 1]])
			print("* done training")

			# append to the ensemble of pairwise predictions
			if self.devMode:
				ensemblePreds = self.addEnsemblePredictions(self.helper.devDirs, self.devID, preds)
			else:
				ensemblePreds = self.addEnsemblePredictions(self.helper.testingDirs, self.testID, preds)

			# if it's our last run, let's use the ensemble'd preds
			if _ == numRuns - 1:
				print("*** SETTING PREDS = ensemble!!")
				preds = ensemblePreds

			# performs WD agglomerative clustering
			for sp in self.stopping_points:
				print("* [agg] sp:", sp)
				if self.devMode:
					(wd_docPredClusters, wd_predictedClusters, wd_goldenClusters) = self.aggClusterWD(self.helper.devDirs, self.devID, preds, sp)
				else:
					(wd_docPredClusters, wd_predictedClusters, wd_goldenClusters) = self.aggClusterWD(self.helper.testingDirs, self.testID, preds, sp)
				#(bcub_p, bcub_r, bcub_f1, muc_p, muc_r, muc_f1, ceafe_p, ceafe_r, ceafe_f1, conll_f1)
				start_time = time.time()

				if self.args.useECBTest: # uses ECB Test Mentions
					scores = get_conll_scores(wd_goldenClusters, wd_predictedClusters)
					print("* getting conll score took", str((time.time() - start_time)), "seconds")
					spToCoNLL[sp].append(scores[-1])
					spToPredictedCluster[sp] = wd_predictedClusters
					spToDocPredictedCluster[sp] = wd_docPredClusters

					if scores[-1] > bestRunCoNLL:
						bestRunCoNLL = scores[-1]
						bestRunSP = sp

				else: # uses HDDCRP Test Mentions
					self.helper.writeCoNLLFile(wd_predictedClusters, "wd", sp)

					# pickles the predictions
					with open("wd_hddcrp_clusters_FULL_sp" + str(sp) + ".p", 'wb') as pickle_out:
						pickle.dump(wd_docPredClusters, pickle_out)
				#print("[DEV] AGGWD SP:", str(round(sp,4)), "CoNLL F1:", str(round(conll_f1,4)), "MUC:", str(round(muc_f1,4)), "BCUB:", str(round(bcub_f1,4)), "CEAF:", str(round(ceafe_f1,4)))
	
			if self.args.useECBTest:
				numGoldPos = 0
				scoreToGoldTruth = defaultdict(list)
				for _ in range(len(preds)):
					if self.devMode:
						if self.devY[_]:
							numGoldPos += 1
							scoreToGoldTruth[preds[_][0]].append(1)
						else:
							scoreToGoldTruth[preds[_][0]].append(0)

					else:
						if self.testY[_]:
							numGoldPos += 1
							scoreToGoldTruth[preds[_][0]].append(1)
						else:
							scoreToGoldTruth[preds[_][0]].append(0)
				s = sorted(scoreToGoldTruth.keys())
				TP = 0.0
				FP = 0.0
				bestF1 = 0
				bestVal = -1
				bestR = 0
				bestP = 0
				numReturnedSoFar = 0
				for eachVal in s:
					for _ in scoreToGoldTruth[eachVal]:
						if _ == 1:
							TP += 1
						else:
							FP += 1

					numReturnedSoFar += len(scoreToGoldTruth[eachVal])
					recall = float(TP / numGoldPos)
					prec = float(TP / numReturnedSoFar)
					f1 = 0
					if (recall + prec) > 0:
						f1 = 2*(recall*prec) / (recall + prec)
					if f1 > bestF1:
						bestF1 = f1
						bestVal = eachVal
						bestR = recall
						bestP = prec

				if bestF1 > 0:
					f1s.append(bestF1)
					recalls.append(bestR)
					precs.append(bestP)
					print("ccnn_best_f1 (run ", len(f1s), "): best_pairwise_f1: ", round(bestF1,4), " prec: ",round(bestP,4), " recall: ", round(bestR,4), " threshold: ", round(bestVal,3), sep="")
					print("** AGG - run", str(_), "bestRunCoNLL:",bestRunCoNLL, "sp: ",bestRunSP)
			sys.stdout.flush()

		# clears ram
		self.trainX = None
		self.trainY = None
		if self.args.useECBTest:
			stddev = -1
			if len(f1s) > 1:
				stddev = self.standard_deviation(f1s)
			print("pairwise f1 (over",len(f1s),"runs) -- avg:", round(sum(f1s)/len(f1s),4), "max:", round(max(f1s),4), "min:", round(min(f1s),4), "avgP:",sum(precs)/len(precs),"avgR:",round(sum(recalls)/len(recalls),4),"stddev:", round(100*stddev,4))
			(best_sp, best_conll, min_conll, max_conll, std_conll) = self.calculateBestKey(spToCoNLL)

			sys.stdout.flush()
			
			print("* [AGGWD] conll f1 -- best sp:",best_sp, "yielded: min:",round(100*min_conll,4),"avg:",round(100*best_conll,4),"max:",round(max_conll,4),"stddev:",round(std_conll,4))
			
			# ENSEMBLE
			#print("* NOW RUNNING AGG ON OUR ENSEMBLE PAIRWISE PREDICTIONS")


			fout = open("ccnn_agg_dirHalf.csv", "a+")
			fout.write(str(self.args.devDir) + ",wd," + str(self.devMode) + "," + str(best_conll) + "\n")
			fout.close()

			return (spToDocPredictedCluster[best_sp], spToPredictedCluster[best_sp], wd_goldenClusters, best_sp)
		
		else: # HDDCRP
			# non-sensical return; only done so that the return handler doesn't complain
			return wd_docPredClusters, wd_predictedClusters, wd_goldenClusters, 0
##########################
##########################

	# CROSS-DOC MODEL
	def train_and_test_cd(self, numRuns): #, wd_pred, wd_gold, numRuns):
		f1s = []
		recalls = []
		precs = []
		spToCoNLL = defaultdict(list)
		while len(f1s) < numRuns:
			# define model
			input_shape = self.trainX.shape[2:]
			base_network = self.create_base_network(input_shape)
			input_a = Input(shape=input_shape)
			input_b = Input(shape=input_shape)
			processed_a = base_network(input_a)
			processed_b = base_network(input_b)
			distance = Lambda(self.euclidean_distance, output_shape=self.eucl_dist_output_shape)(
				[processed_a, processed_b])
			model = Model(inputs=[input_a, input_b], outputs=distance)
			model.compile(loss=self.contrastive_loss, optimizer=Adam())
			#print(model.summary())
			model.fit([self.trainX[:, 0], self.trainX[:, 1]], self.trainY,
							batch_size=self.bs,
							epochs=self.ne,
							validation_data=([self.devX[:, 0], self.devX[:, 1]], self.devY))

			if self.devMode:
				preds = model.predict([self.devX[:, 0], self.devX[:, 1]])
			else:
				preds = model.predict([self.testX[:, 0], self.testX[:, 1]])
			print("* done training")

			numGoldPos = 0
			scoreToGoldTruth = defaultdict(list)
			for _ in range(len(preds)):
				if self.devMode:
					if self.devY[_]:
						numGoldPos += 1
						scoreToGoldTruth[preds[_][0]].append(1)
					else:
						scoreToGoldTruth[preds[_][0]].append(0)
				else:
					if self.testY[_]:
						numGoldPos += 1
						scoreToGoldTruth[preds[_][0]].append(1)
					else:
						scoreToGoldTruth[preds[_][0]].append(0)
			s = sorted(scoreToGoldTruth.keys())
			TP = 0.0
			FP = 0.0
			bestF1 = 0
			bestVal = -1
			bestR = 0
			bestP = 0
			numReturnedSoFar = 0
			for eachVal in s:
				for _ in scoreToGoldTruth[eachVal]:
					if _ == 1:
						TP += 1
					else:
						FP += 1

				numReturnedSoFar += len(scoreToGoldTruth[eachVal])
				recall = float(TP / numGoldPos)
				prec = float(TP / numReturnedSoFar)
				f1 = 0
				if (recall + prec) > 0:
					f1 = 2*(recall*prec) / (recall + prec)
				if f1 > bestF1:
					bestF1 = f1
					bestVal = eachVal
					bestR = recall
					bestP = prec

				if f1 == 1:
					print("*** somehow, F1 is 1, so numReturnedSoFar:", numReturnedSoFar)
				print("\t\tCD eachVal:",eachVal,"=>",f1)
			print("CD bestF1:",bestF1)
			if bestF1 > 0:
				f1s.append(bestF1)
				recalls.append(bestR)
				precs.append(bestP)

				# performs CD agglomerative clustering				
				for sp in self.stopping_points:
					print("* [agg] sp:", sp)
					if self.devMode:
						(cd_docPredClusters, cd_predictedClusters, cd_goldenClusters) = self.aggClusterCD(self.devID, preds, sp)
					else:
						(cd_docPredClusters, cd_predictedClusters, cd_goldenClusters) = self.aggClusterCD(self.testID, preds, sp)  # self.aggClusterCD(self.devID, preds, sp)
					#(bcub_p, bcub_r, bcub_f1, muc_p, muc_r, muc_f1, ceafe_p, ceafe_r, ceafe_f1, conll_f1)
					
					start_time = time.time()
					if self.args.useECBTest:  # uses ECB Test Mentions
						scores = get_conll_scores(cd_goldenClusters, cd_predictedClusters)
						print("* getting conll score took", str((time.time() - start_time)), "seconds")
						spToCoNLL[sp].append(scores[-1])
					else:  # uses HDDCRP Test Mentions
						self.helper.writeCoNLLFile(cd_predictedClusters, "cd", sp)

					#print("[DEV] AGGCD SP:", str(round(sp, 4)), "CoNLL F1:", str(round(scores[-1], 4)))
					#, "MUC:", str(round(muc_f1, 4)), "BCUB:", str(round(bcub_f1, 4)), "CEAF:", str(round(ceafe_f1, 4)))

			print("ccnn_best_f1 (run ", len(f1s), "): best_pairwise_f1: ", round(bestF1, 4), " prec: ", round(bestP, 4), " recall: ", round(bestR, 4), " threshold: ", round(bestVal, 3), sep="")
			sys.stdout.flush()



		# clears ram
		self.trainX = None
		self.trainY = None
		
		if self.args.useECBTest:
			stddev = -1
			if len(f1s) > 1:
				stddev = self.standard_deviation(f1s)
			print("pairwise f1 (over",len(f1s),"runs) -- avg:", round(sum(f1s)/len(f1s),4), "max:", round(max(f1s),4), "min:",
			      round(min(f1s),4), "avgP:",sum(precs)/len(precs),"avgR:",round(sum(recalls)/len(recalls),4),"stddev:", round(100*stddev,4))
			(best_sp, best_conll, min_conll, max_conll,
			 std_conll) = self.calculateBestKey(spToCoNLL)

			sys.stdout.flush()

			print("* [AGGCD] conll f1 -- best sp:",best_sp, "yielded: min:",round(100*min_conll,4), "avg:",round(100*best_conll,4),"max:",round(max_conll,4),"stddev:",round(std_conll,4))
			fout = open("ccnn_agg_dirHalf.csv", "a+")
			fout.write(str(self.args.devDir) + ",cd," + str(self.devMode) + "," + str(best_conll) + "\n")
			fout.close()
			return (None, None, None, None) # we don't ever care about using the return values

		else: # HDDCRP
			# non-sensical return; only done so that the return handler doesn't complain
			return cd_docPredClusters, cd_predictedClusters, cd_goldenClusters, 0
		

		#### 
		stddev = -1
		if len(f1s) > 1:
			stddev = self.standard_deviation(f1s)
		print("pairwise f1 (over", len(f1s), "runs) -- avg:", sum(f1s)/len(f1s), "max:", max(f1s), "min:",
			  min(f1s), "avgP:", sum(precs)/len(precs), "avgR:", sum(recalls)/len(recalls), "stddev:", stddev)

		(best_sp, best_conll, min_conll, max_conll, std_conll) = self.calculateBestKey(spToCoNLL)
		sys.stdout.flush()

		print("* [AGGCD] conll f1 -- best sp:", best_sp, "yielded: min:", round(100*min_conll, 4), "avg:", round(100*best_conll, 4), "max:", round(max_conll, 4), "stddev:", round(std_conll, 4))
		
		fout = open("ccnn_agg_dirHalf.csv", "a+")
		fout.write(str(self.args.devDir) + ",cd," + str(self.devMode) + "," + str(best_conll) + "\n")
		fout.close()
		return (None, cd_predictedClusters, cd_goldenClusters, best_sp)

	def calculateBestKey(self, dict):
		best_conll = 0
		best_sp = 0
		for sp in dict.keys():
			avg = float(sum(dict[sp])/len(dict[sp]))
			if avg > best_conll:
				best_conll = avg
				best_sp = sp
		std_dev = self.standard_deviation(dict[best_sp])
		return (best_sp, best_conll, min(dict[best_sp]), max(dict[best_sp]), std_dev)

	# agglomerative cluster the within-doc predicted pairs
	def aggClusterWD(self, relevant_dirs, ids, preds, stoppingPoint):
		#print("** in aggClusterWD(), stoppingPoint:",stoppingPoint)
		start_time = time.time()
		docToXUIDPredictions = defaultdict(lambda: defaultdict(float))
		docToXUIDsFromPredictions = defaultdict(list) # this list is constructed just to ensure it's the same as the corpus'
		for ((xuid1, xuid2), pred) in zip(ids, preds):
			m1 = self.corpus.XUIDToMention[xuid1]
			m2 = self.corpus.XUIDToMention[xuid2]
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

			# construct the golden truth for the current doc
			# we don't need to check if xuid is in our corpus or predictions because Doc.assignECBMention() adds
			# xuids to REFToEUIDs and .XUIDS() -- the latter we checked, so it's all good
			for curREF in curDoc.REFToEUIDs:
				goldenSuperSet[goldenClusterID] = set(curDoc.REFToEUIDs[curREF])
				goldenClusterID += 1

			# if our doc only has 1 XUID, then we merely need to:
			# (1) add a golden cluster and (2) add 1 single returned cluster
			if len(self.dh.docToXUIDsWeWantToUse[doc_id]) == 1:
				
				#print("\tdoc:", doc_id, "has only 1 XUID (per DataHandler):", self.dh.docToXUIDsWeWantToUse[doc_id])
				if len(curDoc.REFToEUIDs) != 1:
					print("* WARNING: doc:",doc_id,"has only 1 XUID (per DataHandler), but per Corpus has more")
					if self.args.useECBTest:
						print("(this shouldn't happen w/ ECBTest, so exiting")
						exit(1)

				ourDocClusters[0] = set([next(iter(self.dh.docToXUIDsWeWantToUse[doc_id]))])
				#ourClusterSuperSet[ourClusterID] = set([next(iter(self.dh.docToXUIDsWeWantToUse[doc_id]))])
				#ourClusterID += 1

			else: # we have more than 1 XUID for the given doc

				# we check our mentions to the corpus, and we correctly
				# use HDDCRP Mentions if that's what we're working with
				docXUIDsFromCorpus = curDoc.EUIDs
				if not self.args.useECBTest:
					docXUIDsFromCorpus = curDoc.HMUIDs
				
				# ensures our predictions include each of the Doc's mentions
				for xuid in docXUIDsFromCorpus:
					if xuid not in docToXUIDsFromPredictions[doc_id]:
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

	# agglomerative cluster the cross-doc predicted pairs
	# NOTE: 'dir_num' in this function is used to refer to EITHER
	# dirHalf or the actual dir; 
	def aggClusterCD(self, ids, preds, stoppingPoint):

		# NOTE: this is called dir, but we may be operating on dirHalf, which is fine;
		# the ids and predictions passed-in will only be of legit, valid pairs we care about
		# so even if it is dirHalf, it's not like we'll wrongly consider pairs that belong
		# to *other* dirHalves within the same dir, despite the name being 'dirTo...' 
		dirToXUIDPredictions = defaultdict(lambda: defaultdict(float))
		# this list is constructed just to ensure it's the same as the corpus'
		dirToXUIDsFromPredictions = defaultdict(list)
		xuidsFromPredictions = set()
		for ((xuid1, xuid2), pred) in zip(ids, preds):
			m1 = self.corpus.XUIDToMention[xuid1]
			m2 = self.corpus.XUIDToMention[xuid2]
			# NOTE: the lower the score, the more likely they are the same.  it's a dissimilarity score
			pred = pred[0]

			if xuid1 in self.corpus.SUIDToMention:
				print("* ERROR: why is xuid:",xuid1,"in our local predictions? it's a SUID")
				exit(1)
			if xuid2 in self.corpus.SUIDToMention:
				print("* ERROR: why is xuid:", xuid2,"in our local predictions? it's a SUID")
				exit(1)

			dir_num = m1.dir_num
			if self.scope == "dirHalf":
				dir_num = m1.dirHalf

			if self.scope == "dir" and m2.dir_num != dir_num:
				print("* ERROR: xuids are from diff dirs!")
				exit(1)
			if self.scope == "dirHalf" and m2.dirHalf != dir_num:
				print("m2.dirHalf:", m2.dirHalf)
				print("dir_num:", dir_num)
				print("* ERROR: xuids are from diff dirHalves!")
				exit(1)
			if xuid1 not in dirToXUIDsFromPredictions[dir_num]:
				dirToXUIDsFromPredictions[dir_num].append(xuid1)
			if xuid2 not in dirToXUIDsFromPredictions[dir_num]:
				dirToXUIDsFromPredictions[dir_num].append(xuid2)
			xuidsFromPredictions.add(xuid1)
			xuidsFromPredictions.add(xuid2)
			dirToXUIDPredictions[dir_num][(xuid1, xuid2)] = pred
		
		print("* SANITY CHECK: xuidsFromPredictions:",len(xuidsFromPredictions))
		self.sanityCheck2(xuidsFromPredictions)

		ourClusterID = 0
		ourClusterSuperSet = {}
		goldenClusterID = 0
		goldenSuperSet = {}
		
		# only used for output'ing CD clusters to file
		docToPredClusters = defaultdict(list)

		for dir_num in dirToXUIDPredictions.keys():
			
			# adds to our golden clusters
			REFToUIDs = None
			if self.scope == "dirHalf":
				REFToUIDs = self.corpus.dirHalves[dir_num].REFToEUIDs
			elif self.scope == "dir":
				REFToUIDs = self.corpus.ECBDirs[dir_num].REFToEUIDs
			else:
				print("* incorrect scope")
				exit(1)

			tmp_goldXUIDs = set() # TODO REMOVE
			for curREF in REFToUIDs:
				goldenSuperSet[goldenClusterID] = set(REFToUIDs[curREF])
				goldenClusterID += 1

				for _ in REFToUIDs[curREF]:
					tmp_goldXUIDs.add(_)

			# used for our local base clusters
			ourDirNumClusters = {}
			clusterNumToDocs = defaultdict(set)
			curClusterNum = 0
			


			tmpGoldClusters = {}
			tmpGoldNum = 0
			for doc_id in self.wd_pred_clusters:
				dn = int(doc_id.split("_")[0])
				extension = doc_id[doc_id.find("ecb"):]
				dirHalf = str(dn) + extension
				if self.scope == "dirHalf" and dirHalf != dir_num:
					continue
				elif self.scope == "dir" and dn != dir_num:
					continue
				elif self.scope != "dirHalf" and self.scope != "dir":
					print("* incorrect scope")
					exit(1)
				#print("we believe doc_id:", doc_id, "is valid")
				
				# wd predictions for current dir
				for cluster in self.wd_pred_clusters[doc_id]:
					a = set()
					for xuid in self.wd_pred_clusters[doc_id][cluster]:
						a.add(xuid)
					ourDirNumClusters[curClusterNum] = a
					clusterNumToDocs[curClusterNum].add(doc_id)
					curClusterNum += 1

			# agg cluster.  check every combination O(n^2) but n is small (e.g., 10-30)
			while len(ourDirNumClusters.keys()) > 1:

				closestDist = 999999
				closestClusterKeys = (-1, -1)

				closestAvgDist = 999999
				closestAvgClusterKeys = (-1,-1)
				
				closestAvgAvgDist = 999999
				closestAvgAvgClusterKeys = (-1, -1)

				added = set()
				for c1 in ourDirNumClusters:
					docsInC1 = clusterNumToDocs[c1]
					for c2 in ourDirNumClusters:
						if (c2, c1) in added or (c1, c2) in added or c1 == c2:
							continue
						docsInC2 = clusterNumToDocs[c2]

						# only consider merging clusters that are disjoint in their docs
						containsOverlap = False
						for d1 in docsInC1:
							if d1 in docsInC2:
								containsOverlap = True
								break
						if containsOverlap:
							continue

						avgavgdists = []
						dmToDists = defaultdict(list)
						for dm1 in ourDirNumClusters[c1]:
							for dm2 in ourDirNumClusters[c2]:
								if dm1 == dm2:
									print("* ERROR: somehow dm1 == dm2")
									exit(1)

								dist = 9999
								if (dm1, dm2) in dirToXUIDPredictions[dir_num]:
									dist = dirToXUIDPredictions[dir_num][(dm1, dm2)]
								elif (dm2, dm1) in dirToXUIDPredictions[dir_num]:
									dist = dirToXUIDPredictions[dir_num][(dm2, dm1)]
								else:
									print("* ERROR: missing dist for dm1,dm2")
									print("dms:", str(dm1), str(dm2))
									exit(1)

								avgavgdists.append(dist)
								dmToDists[dm1].append(dist)
								dmToDists[dm2].append(dist)
								if dist < closestDist:
									closestDist = dist
									closestClusterKeys = (c1, c2)
						for dm in dmToDists:
							avg = float(sum(dmToDists[dm])/float(len(dmToDists[dm])))
							if avg < closestAvgDist:
								closestAvgDist = avg
								closestAvgClusterKeys = (c1, c2)
						avgavg = float(sum(avgavgdists)) / float(len(avgavgdists))
						if avgavg < closestAvgAvgDist:
							closestAvgAvgDist = avgavg
							closestAvgAvgClusterKeys = (c1, c2)
						added.add((c1, c2))
						added.add((c2, c1))

				# min pair (could also be avg or avgavg)
				#dist = closestDist
				#(c1,c2) = closestClusterKeys

				#dist = closestAvgDist
				#(c1, c2) = closestAvgClusterKeys

				dist = closestAvgAvgDist
				(c1,c2) = closestAvgAvgClusterKeys

				#print("* dist:",dist,"we think we should merge:",c1,c2,"which are:",ourDirNumClusters[c1],"and",ourDirNumClusters[c2])

				if dist > stoppingPoint:  # also handles the case when no candidate clusters were used
					break

				newCluster = set()
				for _ in ourDirNumClusters[c1]:
					newCluster.add(_)
				for _ in ourDirNumClusters[c2]:
					newCluster.add(_)
				ourDirNumClusters.pop(c1, None)
				ourDirNumClusters.pop(c2, None)
				ourDirNumClusters[curClusterNum] = newCluster
				newDocSet = set()
				for _ in clusterNumToDocs[c1]:
					newDocSet.add(_)
				for _ in clusterNumToDocs[c2]:
					newDocSet.add(_)
				clusterNumToDocs.pop(c1, None)
				clusterNumToDocs.pop(c2, None)
				clusterNumToDocs[curClusterNum] = newDocSet
				curClusterNum += 1
			
			# done merging clusters for current 'dir_num' (aka dir or dirHalf)
			docToPredClusters[dir_num] = ourDirNumClusters
			for i in ourDirNumClusters.keys():
				ourClusterSuperSet[ourClusterID] = ourDirNumClusters[i]
				ourClusterID += 1

		print("\tagg took ", str((time.time() - start_time)), "seconds")
		#print("# golden clusters:",str(len(goldenSuperSet.keys())), "; # our clusters:",str(len(ourClusterSuperSet)))
		return (docToPredClusters, ourClusterSuperSet, goldenSuperSet)


		# SANITY CHECK -- ensures our returned gold and predicted clusters all contain the same XUIDs
		if self.args.useECBTest:
			golds = [x for c in goldenSuperSet for x in goldenSuperSet[c]]
			preds = [x for c in ourClusterSuperSet for x in ourClusterSuperSet[c]]
			for xuid in golds:
				if xuid not in preds:
					print("* ERROR: missing",xuid,"from preds")
					exit(1)
			for xuid in preds:
				if xuid not in golds:
					print("* ERROR: missing",xuid,"from golds")
					exit(1)

		# our base clusters are dependent on our scope (dir vs dirHalf)
		#print("# golden clusters:",str(len(goldenSuperSet.keys())), "; # our clusters:",str(len(ourClusterSuperSet)))
		return (   ourClusterSuperSet, goldenSuperSet)

	# Base network to be shared (eq. to feature extraction).
	def create_base_network(self, input_shape):
		seq = Sequential()
		kernel_rows = 1

		for i in range(self.nl):
			nf = self.nf
			if i == 1: # meaning 2nd layer, since i in {0,1,2, ...}
				nf = 96
			seq.add(Conv2D(nf, kernel_size=(kernel_rows, 3), activation='relu', padding="same", input_shape=input_shape, data_format="channels_first"))
			seq.add(Dropout(float(self.do)))
			seq.add(MaxPooling2D(pool_size=(kernel_rows, 2), padding="same", data_format="channels_first"))

		seq.add(Flatten())
		seq.add(Dense(self.nf, activation='relu'))
		return seq

	def euclidean_distance(self, vects):
		x, y = vects
		return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

	def eucl_dist_output_shape(self, shapes):
		shape1, _ = shapes
		return (shape1[0], 1)

	# Contrastive loss from Hadsell-et-al.'06
	# http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
	def contrastive_loss(self, y_true, y_pred):
		margin = 1
		return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

	def standard_deviation(self, lst):
		num_items = len(lst)
		mean = sum(lst) / num_items
		differences = [x - mean for x in lst]
		sq_differences = [d ** 2 for d in differences]
		ssd = sum(sq_differences)
		variance = ssd / (num_items - 1)
		return sqrt(variance)

	# SANITY CHECKS: ensure the loaded predictions use XUID mentions like intended (w/ our corpus)
	def sanityCheck1(self):
		for doc_id in self.wd_pred_clusters:
			# wd predictions for current dir
			for cluster in self.wd_pred_clusters[doc_id]:
				for xuid in self.wd_pred_clusters[doc_id][cluster]:
					if self.corpus.XUIDToMention[xuid].doc_id != doc_id:
						print("DOCS DIFFER!!")
						exit(1)

		tmp_corpusDirHalfToEUIDs = defaultdict(set)
		tmp_corpusXUIDToDH = {}
		for euid in self.corpus.XUIDToMention:
			m = self.corpus.XUIDToMention[euid]
			tmp_corpusDirHalfToEUIDs[m.dirHalf].add(euid)
			tmp_corpusXUIDToDH[euid] = m.dirHalf

		tmp_wdDirHalfToEUIDs = defaultdict(set)
		tmp_wdXUIDToDH = {}
		for doc in self.wd_pred_clusters:
			for c in self.wd_pred_clusters[doc]:
				for euid in self.wd_pred_clusters[doc][c]:
					m = self.corpus.XUIDToMention[euid]
					tmp_wdDirHalfToEUIDs[m.dirHalf].add(euid)
					tmp_wdXUIDToDH[euid] = m.dirHalf

		for xuid in tmp_wdXUIDToDH:
			if tmp_wdXUIDToDH[xuid] != tmp_corpusXUIDToDH[xuid]:
				print("* ERROR!", xuid, tmp_wdXUIDToDH[xuid], tmp_corpusXUIDToDH[xuid])
				exit(1)
		print("* [PASSED] WD_PREDICTIONS align w/ the corpus in terms of dirHalves")

	def sanityCheck2(self, xuidsFromPredictions):
		# sanity check: ensures our DataHandler's XUID's matches the WD ones we import
		for xuid in xuidsFromPredictions:
			if self.devMode:
				if xuid not in self.dh.devXUIDs:
					print("* ERROR: xuid (from predictions) isn't in dh.devXUIDs")
					exit(1)
					m = self.corpus.XUIDToMention[xuid]
					if m.dir_num not in self.helper.devDirs:
						print("* ERROR: xuid's mention is from a dir other than helper.devDirs")
						exit(1)
			else:
				if xuid not in self.dh.testXUIDs:
					print("* ERROR: xuid (from predictions) isn't in dh.devXUIDs")
					exit(1)
					m = self.corpus.XUIDToMention[xuid]
					if m.dir_num not in self.helper.devDirs:
						print("* ERROR: xuid's mention is from a dir other than helper.devDirs")
						exit(1)
