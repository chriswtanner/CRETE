import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
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
	def __init__(self, helper, dh, useRelationalFeatures, scope, presets):
		self.helper = helper
		self.corpus = helper.corpus
		self.args = helper.args
		self.scope = scope # used by aggClusterCD()
		if presets == []:
			self.bs = self.args.batchSize
			self.ne = self.args.numEpochs
			self.nl = self.args.numLayers
			self.nf = self.args.numFilters
			self.do = self.args.dropout
		else:
			(self.bs, self.ne, self.nl, self.nf, self.do) = presets
		print("[ccnn] scope:",self.scope,"bs:",self.bs,"ne:",self.ne,"nl:",self.nl,"nf:",self.nf,"do:",self.do)

		print("loading wd predicted clusters")
		self.wd_pred_clusters = pickle.load(open("wd_clusters", 'rb'))

		dh.loadNNData(useRelationalFeatures, True, self.scope) # this 'True' means use CCNN
		(self.trainID, self.trainX, self.trainY) = (dh.trainID, dh.trainX, dh.trainY)
		(self.devID, self.devX, self.devY) = (dh.devID, dh.devX, dh.devY)
		#(self.testID, self.testX, self.testY) = (coref.testID, coref.testX, coref.testY)

		if self.args.native:
			tf.Session(config=tf.ConfigProto(log_device_placement=True))
			os.environ['CUDA_VISIBLE_DEVICES'] = ''

	# WITHIN-DOC MODEL
	def train_and_test_wd(self, numRuns):
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
			distance = Lambda(self.euclidean_distance, output_shape=self.eucl_dist_output_shape)([processed_a, processed_b])
			model = Model(inputs=[input_a, input_b], outputs=distance)
			model.compile(loss=self.contrastive_loss, optimizer=Adam())
			#print(model.summary())
			model.fit([self.trainX[:, 0], self.trainX[:, 1]], self.trainY, \
				batch_size=self.bs, \
				epochs=self.ne, \
				validation_data=([self.devX[:, 0], self.devX[:, 1]], self.devY))

			preds = model.predict([self.devX[:, 0], self.devX[:, 1]])
			#preds = model.predict([self.testX[:, 0], self.testX[:, 1]])
			
			numGoldPos = 0
			scoreToGoldTruth = defaultdict(list)
			for _ in range(len(preds)):
				if self.devY[_]:
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

				# performs agglomerative clustering
				stoppingPoints = [s for s in np.arange(0.25, 0.5, 0.02)]
				for sp in stoppingPoints:
					(wd_docPredClusters, wd_predictedClusters, wd_goldenClusters) = self.aggClusterWD(self.devID, preds, sp)
					#(bcub_p, bcub_r, bcub_f1, muc_p, muc_r, muc_f1, ceafe_p, ceafe_r, ceafe_f1, conll_f1)
					scores = get_conll_scores(wd_goldenClusters, wd_predictedClusters)
					spToCoNLL[sp].append(scores[-1])

					#print("[DEV] AGGWD SP:", str(round(sp,4)), "CoNLL F1:", str(round(conll_f1,4)), "MUC:", str(round(muc_f1,4)), "BCUB:", str(round(bcub_f1,4)), "CEAF:", str(round(ceafe_f1,4)))

			print("ccnn_best_f1 (run ", len(f1s), "): best_pairwise_f1: ", round(bestF1,4), " prec: ",round(bestP,4), " recall: ", round(bestR,4), " threshold: ", round(bestVal,3), sep="")
			sys.stdout.flush()

		# clears ram
		self.trainX = None
		self.trainY = None
		stddev = -1
		if len(f1s) > 1:
			stddev = self.standard_deviation(f1s)
		print("pairwise f1 (over",len(f1s),"runs) -- avg:", sum(f1s)/len(f1s), "max:", max(f1s), "min:", min(f1s), "avgP:",sum(precs)/len(precs),"avgR:",sum(recalls)/len(recalls),"stddev:", stddev)

		(best_sp, best_conll) = self.calculateBestKey(spToCoNLL)
		sys.stdout.flush()
		print("* conll f1 -- best sp:",best_sp, "yielded an avg:",best_conll)
		return (wd_docPredClusters, wd_predictedClusters, wd_goldenClusters)

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

			preds = model.predict([self.devX[:, 0], self.devX[:, 1]])
			#preds = model.predict([self.testX[:, 0], self.testX[:, 1]])

			numGoldPos = 0
			scoreToGoldTruth = defaultdict(list)
			for _ in range(len(preds)):
				if self.devY[_]:
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

				# performs agglomerative clustering
				stoppingPoints = [s for s in np.arange(0.5, 1.0, 0.02)] # should be 0.1 to 0.8 with 0.05
				for sp in stoppingPoints:
					(wd_predictedClusters, wd_goldenClusters) = self.aggClusterCD(self.devID, preds, sp)
					exit(1)
					#(bcub_p, bcub_r, bcub_f1, muc_p, muc_r, muc_f1, ceafe_p, ceafe_r, ceafe_f1, conll_f1)
					scores = get_conll_scores(wd_goldenClusters, wd_predictedClusters)
					spToCoNLL[sp].append(scores[-1])

					#print("[DEV] AGGWD SP:", str(round(sp,4)), "CoNLL F1:", str(round(conll_f1,4)), "MUC:", str(round(muc_f1,4)), "BCUB:", str(round(bcub_f1,4)), "CEAF:", str(round(ceafe_f1,4)))

			print("ccnn_best_f1 (run ", len(f1s), "): best_pairwise_f1: ", round(bestF1, 4), " prec: ",
			      round(bestP, 4), " recall: ", round(bestR, 4), " threshold: ", round(bestVal, 3), sep="")
			sys.stdout.flush()

		# clears ram
		self.trainX = None
		self.trainY = None
		stddev = -1
		if len(f1s) > 1:
			stddev = self.standard_deviation(f1s)
		print("pairwise f1 (over", len(f1s), "runs) -- avg:", sum(f1s)/len(f1s), "max:", max(f1s), "min:",
		      min(f1s), "avgP:", sum(precs)/len(precs), "avgR:", sum(recalls)/len(recalls), "stddev:", stddev)

		(best_sp, best_conll) = self.calculateBestKey(spToCoNLL)
		sys.stdout.flush()
		print("* conll f1 -- best sp:", best_sp, "yielded an avg:", best_conll)
		return (wd_predictedClusters, wd_goldenClusters)

	def calculateBestKey(self, dict):
		best_conll = 0
		best_sp = 0
		for sp in dict.keys():
			avg = float(sum(dict[sp])/len(dict[sp]))
			if avg > best_conll:
				best_conll = avg
				best_sp = sp
		return (best_sp, best_conll)

	# agglomerative cluster the within-doc predicted pairs
	def aggClusterWD(self, ids, preds, stoppingPoint):
		docToXUIDPredictions = defaultdict(lambda: defaultdict(float))
		docToXUIDs = defaultdict(list) # this list is constructed just to ensure it's the same as the corpus'
		for ((xuid1, xuid2), pred) in zip(ids, preds):
			m1 = self.corpus.XUIDToMention[xuid1]
			m2 = self.corpus.XUIDToMention[xuid2]
			pred = pred[0] # NOTE: the lower the score, the more likely they are the same.  it's a dissimilarity score
			doc_id = m1.doc_id
			
			if m2.doc_id != doc_id:
				print("* ERROR: xuids are from diff docs!")
				exit(1)
			if xuid1 not in docToXUIDs[doc_id]:
				docToXUIDs[doc_id].append(xuid1)
			if xuid2 not in docToXUIDs[doc_id]:
				docToXUIDs[doc_id].append(xuid2)
			docToXUIDPredictions[doc_id][(xuid1, xuid2)] = pred

		ourClusterID = 0
		ourClusterSuperSet = {}
		goldenClusterID = 0
		goldenSuperSet = {}
		
		docToPredClusters = defaultdict(list) # only used for pickling' and later loading WD preds again

		XUIDToDocs = defaultdict(set)

		for doc_id in docToXUIDPredictions.keys():
			print("-----------\ncurrent doc:",str(doc_id),"\n-----------")
			# construct the golden truth for the current doc
			curDoc = self.corpus.doc_idToDocs[doc_id]
			
			# we check our mentions to the corpus, and we correctly
			# use HDDCRP Mentions if that's what we're working with
			docMentionSet = curDoc.EUIDs
			if not self.args.useECBTest:
				docMentionSet = curDoc.HMUIDs
			
			# ensures our predictions include each of the Doc's mentions
			for xuid in docMentionSet:
				if xuid not in docToXUIDs[doc_id]:
					print("* ERROR: missing xuid from our predictions")
					exit(1)

			# ensures each of our predicted mentions are valid per our corpus
			for xuid in docToXUIDs[doc_id]:
				if xuid not in docMentionSet:
					print("* ERROR: missing xuid from our corpus")
					exit(1)

			# we don't need to check if xuid is in our corpus or predictions because Doc.assignECBMention() adds
			# xuids to REFToEUIDs and .XUIDS() -- the latter we checked, so it's all good
			for curREF in curDoc.REFToEUIDs:
				goldenSuperSet[goldenClusterID] = set(curDoc.REFToEUIDs[curREF])
				goldenClusterID += 1

			# constructs our base clusters (singletons)
			ourDocClusters = {}
			for i in range(len(docToXUIDs[doc_id])):
				xuid = docToXUIDs[doc_id][i]
				XUIDToDocs[xuid].add(doc_id)
				if len(XUIDToDocs[i]) > 1:
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
			print("ourDocClusters has # keys:", len(ourDocClusters))
			for i in ourDocClusters.keys():
				print("doc:",doc_id,"adding a cluster")
				ourClusterSuperSet[ourClusterID] = ourDocClusters[i]
				ourClusterID += 1
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
		dirToXUIDs = defaultdict(list)
		for ((xuid1, xuid2), pred) in zip(ids, preds):
			m1 = self.corpus.XUIDToMention[xuid1]
			m2 = self.corpus.XUIDToMention[xuid2]
			# NOTE: the lower the score, the more likely they are the same.  it's a dissimilarity score
			pred = pred[0]

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
			if xuid1 not in dirToXUIDs[dir_num]:
				dirToXUIDs[dir_num].append(xuid1)
			if xuid2 not in dirToXUIDs[dir_num]:
				dirToXUIDs[dir_num].append(xuid2)
			dirToXUIDPredictions[dir_num][(xuid1, xuid2)] = pred

		ourClusterID = 0
		ourClusterSuperSet = {}
		goldenClusterID = 0
		goldenSuperSet = {}

		XUIDToDocs = defaultdict(set)
		for dir_num in dirToXUIDPredictions.keys():
			print("-----------\ncurrent dir_num:", str(dir_num), "\n-----------")
			
			# construct the golden truth for the current doc
			cur_dir = None
			if self.scope == "dirHalf":
				cur_dir = self.corpus.dirHalves[dir_num]
			elif self.scope == "dir":
				cur_dir = self.corpus.ECBDirs[dir_num]
			else:
				print("* incorrect scope")
				exit(1)

			# we check our mentions to the corpus, and we correctly
			# use HDDCRP Mentions if that's what we're working with
			curMentionSet = cur_dir.EUIDs
			if not self.args.useECBTest:
				curMentionSet = cur_dir.HMUIDs

			# ensures our predictions include each of the Doc's mentions
			for xuid in curMentionSet:
				if xuid not in dirToXUIDs[dir_num]:
					print("* ERROR: missing xuid from our predictions for dir:", dir_num)
					exit(1)

			# ensures each of our predicted mentions is valid per our corpus
			print("# dirToXUIDs[dirnum]:", len(dirToXUIDs[dir_num]))
			print("curMentionSet:", len(curMentionSet))
			for xuid in dirToXUIDs[dir_num]:
				if xuid not in curMentionSet:
					print("* ERROR: missing xuid from our corpus, but we think it belongs to dir:", dir_num)
					exit(1)

			# we don't need to check if xuid is in our corpus or predictions because Doc.assignECBMention() adds
			# xuids to REFToEUIDs and .XUIDS() -- the latter we checked, so it's all good
			for curREF in cur_dir.REFToEUIDs:
				print("ref:",curREF)
				goldenSuperSet[goldenClusterID] = set(cur_dir.REFToEUIDs[curREF])
				print("golden cluster has size:", len(goldenSuperSet[goldenClusterID]))
				goldenClusterID += 1

			# constructs our base clusters (singletons)
			ourDirHalfClusters = {}
			clusterNumToDocs = defaultdict(set)
			print("* constructing base clusters (wd predictions)")
			# check if the doc belongs to our current dirHalf or dir (whichever is of our focus)
			ij = 0
			wd_relevant_xuid = set()
			for doc_id in self.wd_pred_clusters:
				tmp_dir_num = int(doc_id.split("_")[0])
				tmp_extension = doc_id[doc_id.find("ecb"):]
				tmp_dirHalf = str(tmp_dir_num) + tmp_extension
				if self.scope == "dirHalf" and tmp_dirHalf != dir_num: # yes, this 'dir_num' is correct
					continue
				if self.scope == "dir" and tmp_dir_num != dir_num:
					continue
				for c in self.wd_pred_clusters[doc_id]:
					print("adding:", doc_id, "for dirhalf:",dir_num)
					ourDirHalfClusters[ij] = self.wd_pred_clusters[doc_id][c]
					clusterNumToDocs[ij].add(doc_id)
					for xuid in self.wd_pred_clusters[doc_id][c]:
						wd_relevant_xuid.add(xuid)
					ij += 1
			print("wd_relevant_xuid:", len(wd_relevant_xuid), "; len(dirToXUIDs[dir_num]:", len(dirToXUIDs[dir_num]))
			print("wd_relevant_xuid:", wd_relevant_xuid)
			print("dirToXUIDs:",dirToXUIDs)
			for xuid in wd_relevant_xuid:
				print("xuid:",xuid)
				if xuid not in dirToXUIDs[dir_num]:
					print("* ERROR: wd had ", xuid, "but our passed-in predictions didn't... it's mention:",self.corpus.XUIDToMention[xuid])
					print(dirToXUIDs[dir_num])
			for xuid in dirToXUIDs[dir_num]:
				if xuid not in wd_relevant_xuid:
					print("* ERROR: passed-in had ", xuid, "but our wd didn't... it's mention:", self.corpus.XUIDToMention[xuid])
			if len(wd_relevant_xuid) != len(dirToXUIDs[dir_num]):
				print("* ERROR: we have a different number of mentions via passed-in WD clusters than what we have predictions for")
				exit(1)
			else:
				print("* same #")
			print("we have", len(ourDirHalfClusters.keys()), "clusters for current dirhalf:")
			print("ourDirHalfClusters:",ourDirHalfClusters)
			print("clusterNumToDocs:", clusterNumToDocs)

			'''
			ourDocClusters = {}
			for i in range(len(docToXUIDs[doc_id])):
				xuid = docToXUIDs[doc_id][i]
				XUIDToDocs[xuid].add(doc_id)
				if len(XUIDToDocs[i]) > 1:
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
				closestAvgAvgClusterKeys = (-1, -1)
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
								closestAvgAvgClusterKeys = (c1, c2)
						j += 1
					i += 1
				# print("closestAvgAvgDist is:", str(closestAvgAvgDist),"which is b/w:", str(closestAvgAvgClusterKeys))

				# only merge clusters if it's less than our threshold
				if closestAvgAvgDist > stoppingPoint:
					break

				newCluster = set()
				(c1, c2) = closestAvgAvgClusterKeys
				for _ in ourDocClusters[c1]:
					newCluster.add(_)
				for _ in ourDocClusters[c2]:
					newCluster.add(_)
				ourDocClusters.pop(c1, None)
				ourDocClusters.pop(c2, None)
				ourDocClusters[c1] = newCluster

			# end of clustering current doc
			for i in ourDocClusters.keys():
				ourClusterSuperSet[ourClusterID] = ourDocClusters[i]
				ourClusterID += 1
			'''
		#print("# golden clusters:",str(len(goldenSuperSet.keys())), "; # our clusters:",str(len(ourClusterSuperSet)))
		return (ourClusterSuperSet, goldenSuperSet)

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
