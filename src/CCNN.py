import os
import sys
import keras
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
from get_coref_metrics import *
class CCNN:
	def __init__(self, helper, coref):
		self.helper = helper
		self.corpus = helper.corpus
		self.args = helper.args
		(self.trainID, self.trainX, self.trainY) = (coref.trainID, coref.trainX, coref.trainY)
		(self.devID, self.devX, self.devY) = (coref.devID, coref.devX, coref.devY)
		#(self.testID, self.testX, self.testY) = (coref.testID, coref.testX, coref.testY)

		if self.args.native:
			sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
			os.environ['CUDA_VISIBLE_DEVICES'] = ''

	def train_and_test(self, numRuns):
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
				batch_size=self.args.batchSize, \
				epochs=self.args.numEpochs, \
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
			TN = 0.0
			TP = 0.0
			FN = 0.0
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
					(wd_predictedClusters, wd_goldenClusters) = self.aggClusterWD(self.devID, preds, sp)
					(bcub_p, bcub_r, bcub_f1, muc_p, muc_r, muc_f1, ceafe_p, \
						ceafe_r, ceafe_f1, conll_f1) = get_conll_scores(wd_goldenClusters, wd_predictedClusters)
					spToCoNLL[sp].append(conll_f1)

					#print("[DEV] AGGWD SP:", str(round(sp,4)), "CoNLL F1:", str(round(conll_f1,4)), "MUC:", str(round(muc_f1,4)), "BCUB:", str(round(bcub_f1,4)), "CEAF:", str(round(ceafe_f1,4)))

					# perform CD now


			print("ccnn_best_f1 (run ", len(f1s), "): best pairwisef1", bestF1, " prec: ",bestP, " recall: ", bestR, " threshold: ", bestVal, sep="")
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
		docToMUIDPredictions = defaultdict(lambda: defaultdict(float))
		docToMUIDs = defaultdict(list) # this list is constructed just to ensure it's the same as the corpus'
		for ((muid1, muid2), pred) in zip(ids, preds):
			m1 = self.corpus.XUIDToMention[muid1]
			m2 = self.corpus.XUIDToMention[muid2]
			pred = pred[0] # NOTE: the lower the score, the more likely they are the same.  it's a dissimilarity score
			doc_id = m1.doc_id
			
			if m2.doc_id != doc_id:
				print("* ERROR: muids are from diff docs!")
				exit(1)
			if muid1 not in docToMUIDs[doc_id]:
				docToMUIDs[doc_id].append(muid1)
			if muid2 not in docToMUIDs[doc_id]:
				docToMUIDs[doc_id].append(muid2)
			docToMUIDPredictions[doc_id][(muid1,muid2)] = pred

		ourClusterID = 0
		ourClusterSuperSet = {}
		goldenClusterID = 0
		goldenSuperSet = {}
		
		MUIDToDocs = defaultdict(set)

		for doc_id in docToMUIDPredictions.keys():
			#print("-----------\ncurrent doc:",str(doc_id),"\n-----------")
			# construct the golden truth for the current doc
			curDoc = self.corpus.doc_idToDocs[doc_id]
			# ensures our predictions span all MUIDs in the corpus' doc
			for muid in curDoc.MUIDs:
				if muid not in docToMUIDs[doc_id]:
					print("* ERROR: missing muid from our predictions")
					exit(1)

			for muid in docToMUIDs[doc_id]:
				if muid not in curDoc.MUIDs:
					print("* ERROR: missing muid from our corpus")
					exit(1)

			# we don't need to check if muid is in our corpus or predictions because Doc.assignECBMention() adds
			# muids to REFToMUIDs and .MUIDS() -- the latter we checked, so it's all good
			for curREF in curDoc.REFToMUIDs:
				goldenSuperSet[goldenClusterID] = set(curDoc.REFToMUIDs[curREF])
				goldenClusterID += 1

			# constructs our base clusters (singletons)
			ourDocClusters = {}
			for i in range(len(docToMUIDs[doc_id])):
				muid = docToMUIDs[doc_id][i]
				MUIDToDocs[muid].add(doc_id)
				if len(MUIDToDocs[i]) > 1:
					print("* ERROR, we have multiple MUIDs that share the same ID, despite being in diff docs")
					exit(1)
				a = set()
				a.add(muid)
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
							for muid1 in ourDocClusters[c1]:
								for muid2 in ourDocClusters[c2]:
									dist = 99999
									if (muid1, muid2) in docToMUIDPredictions[doc_id]:
										dist = docToMUIDPredictions[doc_id][(muid1, muid2)]
										avgavgdists.append(dist)
									elif (muid2, muid1) in docToMUIDPredictions[doc_id]:
										dist = docToMUIDPredictions[doc_id][(muid2,muid1)]
										avgavgdists.append(dist)
									else:
										print("* error, why don't we have either muid1 or muid2 in doc_id")
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
			for i in ourDocClusters.keys():
				ourClusterSuperSet[ourClusterID] = ourDocClusters[i]
				ourClusterID += 1
		#print("# golden clusters:",str(len(goldenSuperSet.keys())), "; # our clusters:",str(len(ourClusterSuperSet)))
		return (ourClusterSuperSet, goldenSuperSet)

	# Base network to be shared (eq. to feature extraction).
	def create_base_network(self, input_shape):
		seq = Sequential()
		curNumFilters = self.args.numFilters
		kernel_rows = 1

		for i in range(self.args.numLayers):
			nf = self.args.numFilters
			if i == 1: # meaning 2nd layer, since i in {0,1,2, ...}
				nf = 96
			seq.add(Conv2D(self.args.numFilters, kernel_size=(kernel_rows, 3), activation='relu', padding="same", input_shape=input_shape, data_format="channels_first"))
			seq.add(Dropout(float(self.args.dropout)))
			seq.add(MaxPooling2D(pool_size=(kernel_rows, 2), padding="same", data_format="channels_first"))

		seq.add(Flatten())
		seq.add(Dense(curNumFilters, activation='relu'))
		return seq

	def euclidean_distance(self, vects):
		x, y = vects
		return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

	def eucl_dist_output_shape(self, shapes):
		shape1, shape2 = shapes
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
