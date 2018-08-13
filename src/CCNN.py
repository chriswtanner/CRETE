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
		self.dh = dh
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

		if self.scope != "doc":

			print("corpus' 3104:", self.corpus.XUIDToMention[3104])
			#print("loading wd predicted clusters")
			self.wd_pred_clusters = pickle.load(open("wd_clusters_test_815", 'rb'))
			
			# creates our base clusters
			ourDocClusters = {}
			curClusterNum = 0

			for doc_id in self.wd_pred_clusters:
				print("doc_id:",doc_id)
				dn = int(doc_id.split("_")[0])
				extension = doc_id[doc_id.find("ecb"):]
				dh = str(dn) + extension

				# wd predictions for current dir
				for cluster in self.wd_pred_clusters[doc_id]:
					for xuid in self.wd_pred_clusters[doc_id][cluster]:
						print("c +", self.corpus.XUIDToMention[xuid])
						if self.corpus.XUIDToMention[xuid].doc_id != doc_id:
							print("DOCS DIFFER!!")
							exit(1)

			#print("self.wd_pred_clusters:", self.wd_pred_clusters)
			#exit(1)
			
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

		'''
		for dh in tmp_wdDirHalfToEUIDs:
			print("dh:", dh, len(tmp_wdDirHalfToEUIDs[dh]), len(tmp_corpusDirHalfToEUIDs[dh]))
			print("\t", tmp_wdDirHalfToEUIDs[dh])
			
		exit(1)
		
		self.wd_xuids = set()
		self.wd_dirHalfToXUIDs = defaultdict(set)
		self.wd_xuidToDirHalf = {}
		for doc_id in self.wd_pred_clusters:
			print("wd loaded, doc:",doc_id,"which has # clusters:",len(self.wd_pred_clusters[doc_id]))
			for c in self.wd_pred_clusters[doc_id]:
				for xuid in self.wd_pred_clusters[doc_id][c]:
					self.wd_xuids.add(xuid)
					m = self.corpus.XUIDToMention[xuid]
					self.wd_dirHalfToXUIDs[m.dirHalf].add(xuid)
					self.wd_xuidToDirHalf[xuid] = m.dirHalf
		print("WD IMPORTED:")
		print("self.wd_xuids:", len(self.wd_xuids))
		for dirHalf in self.wd_dirHalfToXUIDs:
			print("dirHalf:", dirHalf, "(", len(self.wd_dirHalfToXUIDs[dirHalf]), "):", self.wd_dirHalfToXUIDs[dirHalf])
		'''
		dh.loadNNData(useRelationalFeatures, True, self.scope) # this 'True' means use CCNN
		(self.trainID, self.trainX, self.trainY) = (dh.trainID, dh.trainX, dh.trainY)
		(self.devID, self.devX, self.devY) = (dh.devID, dh.devX, dh.devY)
		#(self.testID, self.testX, self.testY) = (coref.testID, coref.testX, coref.testY)

		# sanity check:
		'''
		corpus_xuids = set()
		corpus_dirHalfToXUIDs = defaultdict(set)
		corpus_xuidToDirHalf = {}
		for (xuid1, xuid2) in self.devID:
			corpus_xuids.add(xuid1)
			corpus_xuids.add(xuid2)
			m1 = self.corpus.XUIDToMention[xuid1]
			m2 = self.corpus.XUIDToMention[xuid2]
			if m1.dirHalf != m2.dirHalf:
				print("** ERROR; mismatch dirhalves")
				exit(1)
			corpus_dirHalfToXUIDs[m1.dirHalf].add(xuid1)
			corpus_dirHalfToXUIDs[m2.dirHalf].add(xuid2)
			corpus_xuidToDirHalf[xuid1] = m1.dirHalf
			corpus_xuidToDirHalf[xuid2] = m2.dirHalf

		for dh in self.wd_dirHalfToXUIDs:
			print("dh:",dh)
			print("\twd:",self.wd_dirHalfToXUIDs[dh])
			print("\tco:", corpus_dirHalfToXUIDs[dh])

		exit(1)

		for xuid in self.wd_xuidToDirHalf:
			if self.wd_xuidToDirHalf[xuid] != corpus_xuidToDirHalf[xuid]:
				print("** ERROR, xuid:", xuid, "is:", self.wd_xuidToDirHalf[xuid],corpus_xuidToDirHalf[xuid])
		exit(1)
		for xuid in self.wd_xuids:
			if xuid not in corpus_xuids:
				print("** ERROR:", xuid, "not in corpus xuids")
		for xuid in corpus_xuids:
			if xuid not in self.wd_xuids:
				print("** ERROR:", xuid, "not in wd xuids")
		print("# corpus_xuids:", len(corpus_xuids))
		for dirHalf in corpus_dirHalfToXUIDs:
			print("dirHalf:", dirHalf, "(", len(corpus_dirHalfToXUIDs[dirHalf]), ")", corpus_dirHalfToXUIDs[dirHalf])
			for xuid in corpus_dirHalfToXUIDs[dirHalf]:
				if xuid not in self.wd_dirHalfToXUIDs[dirHalf]:
					print("** ERROR, ", xuid, "is not in wd_dirhalftoxuid")
		for dirHalf in self.wd_dirHalfToXUIDs:
			print("dirHalf:", dirHalf)
			for xuid in self.wd_dirHalfToXUIDs[dirHalf]:
				if xuid not in corpus_dirHalfToXUIDs[dirHalf]:
					print("*** ERROR, ",xuid,"not in corpusdirhalf")
		exit(1)
		'''
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
				stoppingPoints = [s for s in np.arange(0.1, 0.50, 0.02)]
				bestScore = 0
				for sp in stoppingPoints:
					(wd_docPredClusters, wd_predictedClusters, wd_goldenClusters) = self.aggClusterWD(self.helper.devDirs, self.devID, preds, sp)
					#(bcub_p, bcub_r, bcub_f1, muc_p, muc_r, muc_f1, ceafe_p, ceafe_r, ceafe_f1, conll_f1)
					scores = get_conll_scores(wd_goldenClusters, wd_predictedClusters)

					spToCoNLL[sp].append(scores[-1])
					if scores[-1] > bestScore:
						print("* new best score:", scores[-1])
						pickle_out = open("wd_clusters", 'wb')
						pickle.dump(wd_docPredClusters, pickle_out)
						bestScore = scores[-1]

					#print("[DEV] AGGWD SP:", str(round(sp,4)), "CoNLL F1:", str(round(conll_f1,4)), "MUC:", str(round(muc_f1,4)), "BCUB:", str(round(bcub_f1,4)), "CEAF:", str(round(ceafe_f1,4)))
			print("conll scores:", spToCoNLL)
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
		print("* [AGGWD] conll f1 -- best sp:",best_sp, "yielded an avg:",best_conll)
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

			print("ccnn_best_f1 (run ", len(f1s), "): best_pairwise_f1: ", round(bestF1, 4), " prec: ", round(bestP, 4), " recall: ", round(bestR, 4), " threshold: ", round(bestVal, 3), sep="")
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
	def aggClusterWD(self, relevant_dirs, ids, preds, stoppingPoint):
		print("** in aggClusterWD(), stoppingPoint:",stoppingPoint)
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
					print("* ERROR: doc:",doc_id,"has only 1 XUID (per DataHandler), but per Corpus has more")
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
		print("# golden clusters:",str(len(goldenSuperSet.keys())), "; # our clusters:",str(len(ourClusterSuperSet)))
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
		print("* xuidsFromPredictions:",len(xuidsFromPredictions))

		# sanity check: ensures our DataHandler's XUID's matches the WD ones we import
		for xuid in xuidsFromPredictions:
			if xuid not in self.dh.devXUIDs:
				print("* ERROR: xuid (from predictions) isn't in dh.devXUIDs")
				exit(1)
				m = self.corpus.XUIDToMention[xuid]
				if m.dir_num not in self.helper.devDirs:
					print("* ERROR: xuid's mention is from a dir other than helper.devDirs")
					exit(1)

		ourClusterID = 0
		ourClusterSuperSet = {}
		goldenClusterID = 0
		goldenSuperSet = {}

		for dir_num in dirToXUIDPredictions.keys():
			print("dir_num:", dir_num)

			# adds to our golden clusters
			REFToUIDs = None
			if self.scope == "dirHalf":
				REFToUIDs = self.corpus.dirHalves[dir_num].REFToEUIDs
			elif self.scope == "dir":
				REFToUIDs = self.corpus.ECBDirs[dir_num].REFToEUIDs
			else:
				print("* incorrect scope")
				exit(1)
			for curREF in REFToUIDs:
				goldenSuperSet[goldenClusterID] = set(REFToUIDs[curREF])
				goldenClusterID += 1

			# creates our base clusters
			ourDocClusters = {}
			curClusterNum = 0

			tmpGoldClusters = {}
			tmpGoldNum = 0
			for doc_id in self.wd_pred_clusters:
				dn = int(doc_id.split("_")[0])
				extension = doc_id[doc_id.find("ecb"):]
				dh = str(dn) + extension
				if self.scope == "dirHalf" and dh != dir_num:
					continue
				elif self.scope == "dir" and dn != dir_num:
					continue
				elif self.scope != "dirHalf" and self.scope != "dir":
					print("* incorrect scope")
					exit(1)
				print("we believe doc_id:", doc_id, "is valid")
				
				# wd predictions for current dir
				for cluster in self.wd_pred_clusters[doc_id]:
					a = set()
					for xuid in self.wd_pred_clusters[doc_id][cluster]:
						a.add(xuid)
						print("c +", self.corpus.XUIDToMention[xuid],
						      "|", self.corpus.EUIDToMention[xuid])
					ourDocClusters[curClusterNum] = a
					curClusterNum += 1

				# tmp gold for current dir
				for ref in self.corpus.doc_idToDocs[doc_id].REFToEUIDs:
					a = set()
					for xuid in self.corpus.doc_idToDocs[doc_id].REFToEUIDs[ref]:
						a.add(xuid)
						print("g +",self.corpus.XUIDToMention[xuid])
					tmpGoldClusters[tmpGoldNum] = a
					tmpGoldNum += 1
			print("\twill cluster w/ the base clusters:")
			for c in ourDocClusters:
				print("\t\tc:", c, ourDocClusters[c])
			for g in tmpGoldClusters:
				print("\t\tg:", g, tmpGoldClusters[g])
		#for g in goldenSuperSet:
		#	print("g:",g,goldenSuperSet[g])

		# our base clusters are dependent on our scope (dir vs dirHalf)

		'''
			clusterIDToXUIDs = defaultdict(set)
			clusterNum = 0
			for doc_id in self.wd_pred_clusters:
				dir_num = int(doc_id.split("_")[0])
				extension = doc_id[doc_id.find("ecb"):]
				dirHalf = str(dir_num) + extension
				for cluster in self.wd_pred_clusters[doc_id]:
					curCluster = self.wd_pred_clusters[doc_id][cluster]
					print("curCluster:",curCluster)
					for xuid in curCluster:
						clusterIDToXUIDs[clusterNum].add(xuid)
					clusterNum += 1

			#dirHalfToClusters[dirHalf] = s
		print("base clustersS:")
		for dh in dirHalfToClusters:
			print("dh:",dh,dirHalfToClusters[dh])
		exit(1)
		for dirHalf in dirToXUIDPredictions.keys():
			print("-----------\ncurrent dirHalf:", str(dirHalf), "\n-----------")

			# iterate through all wd_clusters to find the ones we care to use
			
			exit(1)

		
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
				if xuid not in dirToXUIDsFromPredictions[dir_num]:
					print("* ERROR: missing xuid from our predictions for dir:", dir_num)
					exit(1)

			# ensures each of our predicted mentions is valid per our corpus
			print("# dirToXUIDsFromPredictions[dirnum]:", len(dirToXUIDsFromPredictions[dir_num]))
			print("curMentionSet:", len(curMentionSet))
			for xuid in dirToXUIDsFromPredictions[dir_num]:
				if xuid not in curMentionSet:
					print("* ERROR: missing xuid from our corpus, but we think it belongs to dir:", dir_num)
					exit(1)

			# we don't need to check if xuid is in our corpus or predictions because Doc.assignECBMention() adds
			# xuids to REFToEUIDs and .XUIDS() -- the latter we checked, so it's all good
			for curREF in cur_dir.REFToEUIDs:
				#print("ref:",curREF)
				goldenSuperSet[goldenClusterID] = set(cur_dir.REFToEUIDs[curREF])
				#print("golden cluster has size:", len(goldenSuperSet[goldenClusterID]))
				goldenClusterID += 1

			# constructs our base clusters (singletons)
			ourDirHalfClusters = {}
			clusterNumToDocs = defaultdict(set)
			print("* constructing base clusters (wd predictions)")
			# check if the doc belongs to our current dirHalf or dir (whichever is of our focus)
			ij = 0
			wd_relevant_xuid = set()

			wd_dirHalf = defaultdict(set)
			for doc_id in self.wd_pred_clusters:
				tmp_dir_num = int(doc_id.split("_")[0])
				tmp_extension = doc_id[doc_id.find("ecb"):]
				tmp_dirHalf = str(tmp_dir_num) + tmp_extension
				print("considering doc_id:",doc_id)
				if self.scope == "dirHalf" and tmp_dirHalf != dir_num: # yes, this 'dir_num' is correct
					continue
				if self.scope == "dir" and tmp_dir_num != dir_num:
					continue
				print("tmp_dirHalf:",tmp_dirHalf, "; self.wd_pred_clusters[doc_id]", self.wd_pred_clusters[doc_id])
				for c in self.wd_pred_clusters[doc_id]:
					print("adding:", doc_id, "for dirhalf:",dir_num)
					ourDirHalfClusters[ij] = self.wd_pred_clusters[doc_id][c]
					clusterNumToDocs[ij].add(doc_id)
					for xuid in self.wd_pred_clusters[doc_id][c]:
						wd_relevant_xuid.add(xuid)
						wd_dirHalf[tmp_dirHalf].add(xuid)
					ij += 1
			print("these are the dirHalves assigned from having loaded in the wd_pred_clusters:")
			print("\twd_dirHalf:", wd_dirHalf)
			print("wd_relevant_xuid:", len(wd_relevant_xuid), "; len(dirToXUIDsFromPredictions[dir_num]:", len(dirToXUIDsFromPredictions[dir_num]))
			print("wd_relevant_xuid:", wd_relevant_xuid)
			print("dirToXUIDsFromPredictions:",dirToXUIDsFromPredictions)
			for xuid in wd_relevant_xuid:
				print("xuid:",xuid)
				print("dir_num:",dir_num)
				if xuid not in dirToXUIDsFromPredictions[dir_num]:
					print("* ERROR: WD (passed-in) had ", xuid, "but our current local predictions didn't... it's mention:",self.corpus.XUIDToMention[xuid])
					print("dir_num:", dir_num)
					print(dirToXUIDsFromPredictions[dir_num])
			for xuid in dirToXUIDsFromPredictions[dir_num]:
				if xuid not in wd_relevant_xuid:
					print("* ERROR: current local predictions had ", xuid, "but our WD (passed-in) didn't... it's mention:", self.corpus.XUIDToMention[xuid])
			if len(wd_relevant_xuid) != len(dirToXUIDsFromPredictions[dir_num]):
				print("* ERROR: we have a different number of mentions via passed-in WD clusters than what we have predictions for")
				exit(1)
			else:
				print("* same #")
			print("we have", len(ourDirHalfClusters.keys()), "clusters for current dirhalf:")
			print("ourDirHalfClusters:",ourDirHalfClusters)
			print("clusterNumToDocs:", clusterNumToDocs)

			
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
