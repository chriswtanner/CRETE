import os
import sys
import time
import keras
import pickle
import math
import random
import numpy as np
import tensorflow as tf
import keras.backend as K

from sklearn.metrics.pairwise import cosine_similarity

from FeatureHandler import FeatureHandler

from sklearn.metrics import accuracy_score # TMP -- dependency parse features
from sklearn.neural_network import MLPClassifier # TMP -- dependency parse features
from sklearn.metrics import classification_report # TMP -- dependency parse features

from math import sqrt, floor
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Embedding, merge, Flatten, Input, Lambda, Conv2D, AveragePooling2D, MaxPooling2D
from keras.optimizers import Adam
from keras.optimizers import SGD
from collections import defaultdict
from get_coref_metrics import get_conll_scores
class CCNN:
	def __init__(self, helper, dh, supp_features, scope, presets, wd_docPreds, devMode, stopping_points):
		self.calculateCoNLLScore = True # should always be True, except for debugging things that don't depend on it
		self.CCNNSupplement = False
		if supp_features != "none":
			self.CCNNSupplement = True
		self.devMode = devMode
		self.helper = helper
		self.dh = dh
		self.corpus = helper.corpus
		self.args = helper.args
		self.wd_pred_clusters = wd_docPreds
		self.scope = scope # used by aggClusterCD()
		self.stopping_points = stopping_points
		if presets == []:
			self.bs = self.args.batchSize
			self.ne = self.args.numEpochs
			self.nl = self.args.numLayers
			self.nf = self.args.numFilters
			self.do = self.args.dropout
		else:
			(self.bs, self.ne, self.nl, self.nf, self.do) = presets

		# TMP: loads the saved TreeLSTM model
		self.saved_treelstm = pickle.load(open("dir_20_e5.p", "rb"))
		#self.saved_treelstm = pickle.load(open("dir_shortest20_e17.p", "rb"))
		#self.saved_treelstm = pickle.load(open("dir_1_e0.p", "rb"))

		print("[ccnn] scope:",self.scope,"bs:",self.bs,"ne:",self.ne,"nl:",self.nl,"nf:",self.nf,"do:",self.do, "dm:",self.devMode, "sp:",self.stopping_points)
		sys.stdout.flush()

		'''
		if self.scope != "doc":
			self.sanityCheck1()
		'''

		self.dh.loadNNData(supp_features, True, self.scope) # True means use CCNN
		(self.trainID, self.trainX, self.trainY) = (dh.trainID, dh.trainX, dh.trainY)
		(self.devID, self.devX, self.devY) = (dh.devID, dh.devX, dh.devY)
		(self.testID, self.testX, self.testY) = (dh.testID, dh.testX, dh.testY)
		print("supplementalTrain:", dh.supplementalTrain[0:10])
		print("self.trainY:", self.trainY[0:10])
		#exit(1)
		self.supplementalTrain = dh.supplementalTrain
		self.supplementalDev = dh.supplementalDev
		self.supplementalTest = dh.supplementalTest

		if self.args.native:
			tf.Session(config=tf.ConfigProto(log_device_placement=True))
			os.environ['CUDA_VISIBLE_DEVICES'] = ''
		
	def get_cosine_sim(self, vec1, vec2):
		dot = np.dot(vec1, vec2)
		norma = np.linalg.norm(vec1)
		normb = np.linalg.norm(vec2)
		cs = dot / (norma * normb)
		return cs

	# returns the average of a list of lists (embeddings)
	def get_avg_vector(self, vectors):
		return [sum(i)/len(vectors) for i in zip(*vectors)]

	def baseline_tests(self, xuid_pairs):

		fh = FeatureHandler(self.args, self.helper)
		fh.loadGloveEmbeddings()
		
		xuid_to_embedding = {}

		num_gold = 0
		any_TP = 0
		any_FP = 0

		all_TP = 0
		all_FP = 0

		golds = []
		preds = []

		cs_preds = []
		l2_preds = []

		hidden_cs_preds = []
		hidden_l2_preds = []
		# goes through all pairs
		num_good = 0
		num_bad = 0

		alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
		threshold_env = defaultdict(list)
		threshold_children = defaultdict(list)
		threshold_avgavg = defaultdict(list)
		threshold_minavg = defaultdict(list)
		threshold_min = defaultdict(list)
		processed = set()
		for proc, (xuid1, xuid2) in enumerate(xuid_pairs):
			if proc % 5000 == 0:
				print("processed:", proc)
			m1 = self.corpus.XUIDToMention[xuid1]
			m2 = self.corpus.XUIDToMention[xuid2]
			is_gold = False

			key = str(xuid1) + "_" + str(xuid2)
			if xuid2 < xuid1:
				key = str(xuid2) + "_" + str(xuid1)
			tree_index = self.saved_treelstm.xuid_pairs_to_index[key]
			tree_event_embs = self.saved_treelstm.xuid_to_event_emb[tree_index]

			is_good = True
			if len(m1.tokens) != len(tree_event_embs[xuid1]):
				is_good = False 
				#exit()
			if len(m2.tokens) != len(tree_event_embs[xuid2]):
				is_good = False
			if is_good:
				num_good += 1
			else:
				num_bad += 1
			m1_avg_emb = self.get_avg_vector(tree_event_embs[xuid1])
			m2_avg_emb = self.get_avg_vector(tree_event_embs[xuid2])

			if m1.REF == m2.REF:
				is_gold = True
				num_gold += 1
				golds.append(0)
			else:
				golds.append(1)

			same_lemma_any = False
			same_lemma_all = True

			m1_lemmas = [fh.getBestStanToken(t.stanTokens).lemma.lower() for t in m1.tokens]
			m2_lemmas = [fh.getBestStanToken(t.stanTokens).lemma.lower() for t in m2.tokens]

			if len(m1_lemmas) != len(m2_lemmas):
				same_lemma_all = False
			else:
				for l1, l2 in zip(m1_lemmas, m2_lemmas):
					if l1 != l2:
						same_lemma_all = False
						break

			for l1 in m1_lemmas:
				if l1 in m2_lemmas:
					same_lemma_any = True

			if same_lemma_all:
				if is_gold:
					all_TP += 1
				else:
					all_FP += 1

			if same_lemma_any:
				if is_gold:
					any_TP += 1
				else:
					any_FP += 1
			
			(uid1, uid2) = sorted([self.corpus.XUIDToMention[xuid1].UID, self.corpus.XUIDToMention[xuid2].UID])
			m1_features = []
			m2_features = []
			# loops through each feature (e.g., BoW, lemma) for the given uid pair
			for feature in self.dh.singleFeatures:
				for i in feature[uid1]:  # loops through each val of the given feature
					m1_features.append(i)
				for i in feature[uid2]:
					m2_features.append(i)

			# LEMMA + CHAR -- [COSINE SIM]
			cs = self.get_cosine_sim(m1_features, m2_features)
			cs_preds.append([cs])

			# LEMMA + CHAR -- [L2]
			l2 = 0
			for i, j in zip(m1_features, m2_features):
				l2 += math.pow(i - j, 2)
			l2 = math.sqrt(l2)
			l2_preds.append([l2])

			# ----------------------------------
			# HIDDEN NODES -- [COSINE SIM]
			hidden_cs = self.get_cosine_sim(m1_avg_emb, m2_avg_emb)
			cnn_cs = self.get_cosine_sim(m1_features, m2_features)
			#hidden_cs_preds.append([cs])

			# looks at the entity children
			entity_embs = self.saved_treelstm.xuid_to_childxuid_to_emb[tree_index]
			entity1_embs = entity_embs[xuid1]
			entity2_embs = entity_embs[xuid2]

			'''
			children1 = []
			for child_xuid in entity1_embs.keys():
				child_emb = []
				uid = self.corpus.XUIDToMention[child_xuid].UID
				for feature in self.dh.singleFeatures:
					for i in feature[uid]:
						child_emb.append(i)
				children1.append(child_emb)

			children2 = []
			for child_xuid in entity2_embs.keys():
				child_emb = []
				uid = self.corpus.XUIDToMention[child_xuid].UID
				for feature in self.dh.singleFeatures:
					for i in feature[uid]:
						child_emb.append(i)
				children2.append(child_emb)
			
			highest_children_cs = -1
			for child1 in children1:
				for child2 in children2:
					cur_cs = self.get_cosine_sim(child1, child2)
					if cur_cs > highest_children_cs:
						highest_children_cs = cur_cs
			
			avg_child1 = self.get_avg_vector(children1)
			avg_child2 = self.get_avg_vector(children2)
			highest_children_cs = self.get_cosine_sim(avg_child1, avg_child2)
			'''
			every_ent1 = [] # stores every individual token (can be used for avg everything and min everything)
			avg_ent1 = [] # stores an avg_vector for each child entity (can be used for min)
			for ent1 in entity1_embs:
				#child = []
				for ent_v in entity1_embs[ent1]:
					every_ent1.append(ent_v)
					#child.append(ent_v)
				#avg_child = self.get_avg_vector(child)
				#avg_ent1.append(avg_child)

			every_ent2 = []
			avg_ent2 = []
			for ent2 in entity2_embs:
				child = []
				for ent_v in entity2_embs[ent2]:
					every_ent2.append(ent_v)
					#child.append(ent_v)
				#avg_child = self.get_avg_vector(child)
				#avg_ent2.append(avg_child)
			
			avg_avg_cs = hidden_cs
			highest_avg_cs = -1
			highest_individual_cs = -1

			if len(every_ent1) == 0 or len(every_ent2) == 0:
				print("WTF current size:", len(processed), " of ", len(xuid_pairs))

				highest_avg_cs = hidden_cs
				highest_individual_cs = hidden_cs
			else:
				processed.add(key)
				avg_avg1 = self.get_avg_vector(every_ent1)
				avg_avg2 = self.get_avg_vector(every_ent2)
				avg_avg_cs = self.get_cosine_sim(avg_avg1, avg_avg2)

				'''
				for each_vec1 in avg_ent1:
					for each_vec2 in avg_ent2:
						cur_cs = self.get_cosine_sim(each_vec1, each_vec2)
						if cur_cs > highest_avg_cs:
							highest_avg_cs = cur_cs

				for each_vec1 in avg_avg1:
					for each_vec2 in avg_avg2:
						cur_cs = self.get_cosine_sim(each_vec1, each_vec2)
						if cur_cs > highest_individual_cs:
							highest_individual_cs = cur_cs
				'''
			for alpha in alphas:
				threshold_env[alpha].append([alpha*hidden_cs + (1-alpha)*cnn_cs])
				threshold_children[alpha].append([alpha*avg_avg_cs + (1-alpha)*cnn_cs])
				#threshold_avgavg[alpha].append([alpha*avg_avg_cs + (1-alpha)*cnn_cs])
				#threshold_minavg[alpha].append([alpha*highest_avg_cs + (1-alpha)*cnn_cs])
				#threshold_min[alpha].append([alpha*highest_individual_cs + (1-alpha)*cnn_cs])
			

			# HIDDEN NODES -- [L2]
			v1 = m1_avg_emb + m1_features
			v2 = m2_avg_emb + m2_features

			l2 = 0
			for i, j in zip(v1, v2):
				l2 += math.pow(i - j, 2)
			l2 = math.sqrt(l2)
			hidden_l2_preds.append([l2])


		all_R = all_TP / float(num_gold)
		all_P = all_TP / float(all_TP + all_FP)
		all_F1 = 2*all_P*all_R / (all_P + all_R)

		any_R = any_TP / float(num_gold)
		any_P = any_TP / float(any_TP + any_FP)
		any_F1 = 2*any_P*any_R / (any_P + any_R)

		#print("all_F1:", all_F1, "any_F1:", any_F1)
		(cs_f1, bestP, bestR, bestVal) = self.helper.evaluatePairwisePreds(None, cs_preds, golds, self.dh, True)
		(l2_f1, bestP, bestR, bestVal) = self.helper.evaluatePairwisePreds(None, l2_preds, golds, self.dh, False)

		#(hidden_cs_f1, bestP, bestR, bestVal) = self.helper.evaluatePairwisePreds(None, hidden_cs_preds, golds, self.dh, True)
		(hidden_l2_f1, bestP, bestR, bestVal) = self.helper.evaluatePairwisePreds(None, hidden_l2_preds, golds, self.dh, False)

		max_hidden_cs = -1
		max_children = -1
		max_avgavg = -1
		max_minavg = -1
		max_min = -1
		for alpha in alphas:
			(hidden_cs_f1, bestP, bestR, bestVal) = self.helper.evaluatePairwisePreds(None, threshold_env[alpha], golds, self.dh, True)
			print("ENV alpha:", alpha, " =", hidden_cs_f1)
			if hidden_cs_f1 > max_hidden_cs:
				max_hidden_cs = hidden_cs_f1

			(children_cs_f1, bestP, bestR, bestVal) = self.helper.evaluatePairwisePreds(None, threshold_children[alpha], golds, self.dh, True)
			print("ENTITY alpha:", alpha, " =", children_cs_f1)
			if children_cs_f1 > max_children:
				max_children = children_cs_f1

			'''
			(hidden_avgavg_f1, bestP, bestR, bestVal) = self.helper.evaluatePairwisePreds(None, threshold_avgavg[alpha], golds, self.dh, True)
			print("hidden_avgavg_f1 alpha:", alpha, " =", hidden_avgavg_f1)
			if hidden_avgavg_f1 > max_avgavg:
				max_avgavg = hidden_avgavg_f1

			(hidden_minavg_f1, bestP, bestR, bestVal) = self.helper.evaluatePairwisePreds(None, threshold_minavg[alpha], golds, self.dh, True)
			print("hidden_minavg_f1 alpha:", alpha, " =", hidden_minavg_f1)
			if hidden_minavg_f1 > max_minavg:
				max_minavg = hidden_minavg_f1

			(hidden_min_f1, bestP, bestR, bestVal) = self.helper.evaluatePairwisePreds(None, threshold_min[alpha], golds, self.dh, True)
			print("hidden_min_f1 alpha:", alpha, " =", hidden_min_f1)
			if hidden_min_f1 > max_min:
				max_min = hidden_min_f1
			'''
		print("num_good:", num_good, "numbad:", num_bad)
		print("max_children:", max_children)
		'''
		print("max_avgavg:", max_avgavg)
		print("max_minavg:", max_minavg)
		print("max_min:", max_min)
		'''
		return (any_F1, all_F1, cs_f1, l2_f1, max_hidden_cs, hidden_l2_f1)

	# WITHIN-DOC MODEL
	def train_and_test(self):
		f1s = []
		recalls = []
		precs = []
		spToCoNLL = defaultdict(list)
		spToPredictedCluster = {}
		spToDocPredictedCluster = {}

		preds = []
		bestRunCoNLL = -1
		bestRunSP = -1

		# define model
		input_shape = self.trainX.shape[2:]
		base_network = self.create_base_network(input_shape)

		if self.CCNNSupplement: # relational, merged layer way
			input_a = Input(shape=input_shape, name='input_a')
			input_b = Input(shape=input_shape, name='input_b')
		else:
			input_a = Input(shape=input_shape)
			input_b = Input(shape=input_shape)
		
		processed_a = base_network(input_a)
		processed_b = base_network(input_b)
		
		distance = Lambda(self.euclidean_distance)([processed_a, processed_b])
		#distance = Lambda(self.euclidean_distance, output_shape=self.eucl_dist_output_shape)([processed_a, processed_b])

		# TODO: do NOT LEAVE THIS IN HERE.  IT FORCES NOT USING SUPPLEMENTAL FEATURES
		self.CCNNSupplement = False

		# WD PART
		if self.CCNNSupplement:
			print("SUPP!")
			#auxiliary_input = Input(shape=(len(self.supplementalTrain),), name='auxiliary_input')
			auxiliary_input = Input(shape=(len(self.supplementalTrain[0]),), name='auxiliary_input')
			#embedded_input = Embedding(input_dim=1, output_dim=1)(auxiliary_input)
			combined_layer = keras.layers.concatenate([distance, auxiliary_input])
			#x = Dense(5, activation='sigmoid')(combined_layer)
			#x2 = Dense(5, activation='sigmoid', use_bias=True)(x)
			main_output = Dense(1, activation='sigmoid', name='main_output')(combined_layer)
			#model = Model(inputs=[auxiliary_input], outputs=main_output)
			model = Model(inputs=[input_a, input_b, auxiliary_input], outputs=main_output)
			model.compile(loss=self.contrastive_loss, optimizer=Adam()) #, metrics=['accuracy'])
			#model.compile(loss=self.contrastive_loss, optimizer=Adam())
			print("summaryz:",model.summary())
			val_loss = [1]
			num_tries = 0
			while val_loss[-1] > 0.1 and num_tries < 1:
				train_history = model.fit({'input_a': self.trainX[:, 0], 'input_b': self.trainX[:, 1], 'auxiliary_input': self.supplementalTrain},
					{'main_output': self.trainY}, shuffle=True, batch_size=self.bs, epochs=self.ne, \
					validation_data=({'input_a': self.devX[:, 0], 'input_b': self.devX[:, 1], 'auxiliary_input': self.supplementalDev}, {'main_output': self.devY}))
				val_loss = train_history.history['val_loss']
				print("val LOSS:", val_loss)
				num_tries += 1
			'''
			inp = model.input
			outputs = [layer.output for layer in model.layers[5:7]]
			functor = K.function(inp + [K.learning_phase()], outputs)
			print("outputs:", outputs)
			
			# Testing
			test = ({'input_a': self.devX[:, 0], 'input_b': self.devX[:, 1], 'auxiliary_input': self.supplementalDev}, {'main_output': self.devY})
			layer_outs = functor([test, 1.])
			print(layer_outs)
			exit(1)
			'''
			for layer in model.layers:
				print("layer:")
				g=layer.get_config()
				h=layer.get_weights()
				print(g)
				print(h)
			
		else:
			#distance = Lambda(self.L1_distance)([processed_a, processed_b])
			#concat = keras.layers.concatenate([processed_a, processed_b])
			#distance = Dense(1, activation='sigmoid', name='main_output', use_bias=False)(concat)
			model = Model(inputs=[input_a, input_b], outputs=distance)
			#model.compile(loss='mean_squared_error', optimizer=Adam())
			model.compile(loss=self.contrastive_loss, optimizer=Adam(), metrics=['accuracy'])
			#model.compile(loss=self.weighted_binary_crossentropy,optimizer=Adam(),metrics=['accuracy'])
			
			print(model.summary())
			model.fit([self.trainX[:, 0], self.trainX[:, 1]], self.trainY, \
				batch_size=self.bs, shuffle=True, epochs=self.ne, verbose=1, \
				validation_data=([self.devX[:, 0], self.devX[:, 1]], self.devY))
		'''
		# TMP dependency features -- train and test a NN over the dependency features
		print("len:", len(self.trainX), len(self.trainY))
		print("len:", len(self.devX), len(self.devY))
		
		alphas = [0.00001, 0.001, 0.01]
		num_epochs = [200, 500]
		
		for a in alphas:
			for ne in num_epochs:
				preds = []
				#for _ in range(3):
				clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15), random_state=1, max_iter=ne)
				clf.fit(self.trainX, self.trainY)
				blah = clf.predict(self.devX)
				scikits_preds = clf.predict_proba(self.devX)
				print(classification_report(self.devY, blah))

				for _ in range(len(scikits_preds)):
					preds.append([scikits_preds[_][0]])
				#print("preds:", preds)
				#preds = clf.predict(self.devX)
				numGoldPos = 0
				scoreToGoldTruth = defaultdict(list)
				#print("lenpreds:",len(preds))
				for _ in range(len(preds)):
					#print("_:", _)
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
					#print("eachVL:", eachVal, "f1:", f1)
				if bestF1 > 0:
					f1s.append(bestF1)
					recalls.append(bestR)
					precs.append(bestP)
					print("ALPHA:", a, "ne:", ne, "ccnn_best_f1 (run ", len(f1s), "): best_pairwise_f1: ", round(bestF1,4), " prec: ",round(bestP,4), " recall: ", round(bestR,4), " threshold: ", round(bestVal,3), sep="")
				#print("ALPHA:", a, "ne:") #, ne, "run:", int(_))
				#print(classification_report(self.devY, preds))
				#print(accuracy_score(self.devY, preds))
				#print("self.devY:", self.devY)
		'''

		if self.CCNNSupplement:
			dev_preds = model.predict({'input_a': self.devX[:, 0], 'input_b': self.devX[:, 1], 'auxiliary_input': self.supplementalDev})
			test_preds = model.predict({'input_a': self.testX[:, 0], 'input_b': self.testX[:, 1], 'auxiliary_input': self.supplementalTest})

		else:
			dev_preds = model.predict([self.devX[:, 0], self.devX[:, 1]])
			test_preds = model.predict([self.testX[:, 0], self.testX[:, 1]])
		
		# evaluates
		(dev_f1, dev_prec, dev_rec, dev_bestThreshold) = self.helper.evaluatePairwisePreds(self.devID, dev_preds, self.devY, self.dh)
		#print("[CCNN DEV RESULTS] f1:", round(dev_f1,4), " prec: ", round(dev_prec,4), " recall: ", round(dev_rec,4), " threshold: ", round(dev_bestThreshold,3))
		dev_results = (self.helper.devDirs, self.devID, dev_preds, self.devY, dev_f1)
		

		(test_f1, test_prec, test_rec, test_bestThreshold) = self.helper.evaluatePairwisePreds(self.testID, test_preds, self.testY, self.dh)
		#print("[CCNN TEST RESULTS] f1:", round(test_f1,4), " prec: ", round(test_prec,4), " recall: ", round(test_rec,4), " threshold: ", round(test_bestThreshold,3))
		test_results = (self.helper.testingDirs, self.testID, test_preds, self.testY, test_f1)

		return [dev_results, test_results]

		'''
		else:
			if self.devMode:
				preds = model.predict([self.devX[:, 0], self.devX[:, 1]])
			else:
				preds = model.predict([self.testX[:, 0], self.testX[:, 1]])
			#print("[DEV] AGGWD SP:", str(round(sp,4)), "CoNLL F1:", str(round(conll_f1,4)), "MUC:", str(round(muc_f1,4)), "BCUB:", str(round(bcub_f1,4)), "CEAF:", str(round(ceafe_f1,4)))
		'''
		
		'''
		if self.devMode:
			if self.args.useECBTest:
				(f1, prec, rec, bestThreshold, wrong_dev_pairs) = self.helper.evaluatePairwisePreds(self.devID, preds, self.devY)
				print("[CCNN BEST PAIRWISE DEV RESULTS] f1:", round(f1,4), " prec: ", round(prec,4), " recall: ", round(rec,4), " threshold: ", round(bestThreshold,3))
			return (self.helper.devDirs, self.devID, preds, self.devY, f1)
		else:
			if self.args.useECBTest:
				(f1, prec, rec, bestThreshold, wrong_test_pairs) = self.helper.evaluatePairwisePreds(self.testID, preds, self.testY)
				print("[CCNN BEST PAIRWISE TEST RESULTS] f1:", round(f1,4), " prec: ", round(prec,4), " recall: ", round(rec,4), " threshold: ", round(bestThreshold,3))
			return (self.helper.testingDirs, self.testID, preds, self.testY, f1)

		
				print("ccnn_best_f1 (# successful runs:",len(f1s),"): best_pairwise_f1: ", round(f1,4), " prec: ",round(prec,4), " recall: ", round(rec,4), " threshold: ", round(bestThreshold,3))
				if f1 > 0.50:
					print("**** ADDING TO ENSEMBLE!")
					# append to the ensemble of pairwise predictions
					if self.devMode:
						ensemblePreds = self.addEnsemblePredictions(True, self.helper.devDirs, self.devID, preds)
					else:
						ensemblePreds = self.addEnsemblePredictions(True, self.helper.testingDirs, self.testID, preds)

					f1s.append(f1)
					recalls.append(rec)
					precs.append(prec)
					
				sys.stdout.flush()
		'''

##########################
##########################

	'''

			# if it's our last run, let's use the ensemble'd preds
				print("*** ENSEMBLE RESULTS!!:")
				preds = ensemblePreds
				print("len PREDS:", len(preds))
				# NOTE: the following used to not be indented, but i recently decided that
				# the ensemble seems to be teh best performing, and since measuring CoNLL takes 1-2 minutes,
				# i might as well ONLY perform clustering+CoNLL eval on the best predictions.
				# however, pairwise CCNN eval is fast, so let's do that for every run
				# performs WD agglomerative clustering
				for sp in self.stopping_points:
					print("* [agg] sp:", sp)
					if self.devMode:
						(wd_docPredClusters, wd_predictedClusters, wd_goldenClusters) = self.aggClusterWD(self.helper.devDirs, self.devID, preds, sp)
					else:
						(wd_docPredClusters, wd_predictedClusters, wd_goldenClusters) = self.aggClusterWD(self.helper.testingDirs, self.testID, preds, sp)
					#(bcub_p, bcub_r, bcub_f1, muc_p, muc_r, muc_f1, ceafe_p, ceafe_r, ceafe_f1, conll_f1)
					start_time = time.time()

					if self.calculateCoNLLScore:
						suffix = "wd_" + str(sp) + "_" + str(_)
						# pickles the predictions
						pickle_out = open("hddcrp_clusters_ONLY_EVENTS_" + str(suffix) + ".p", 'wb')
						pickle.dump(wd_docPredClusters, pickle_out)
						pickle_out.close()

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
							self.helper.writeCoNLLFile(wd_predictedClusters, suffix)

			STUFF WAS HERE?

		# clears ram
		self.trainX = None
		self.trainY = None
		if self.args.useECBTest:
			stddev = -1
			if len(f1s) > 1:
				stddev = self.standard_deviation(f1s)
			print("pairwise f1 (over",len(f1s),"runs) -- avg:", round(sum(f1s)/len(f1s),4), "max:", round(max(f1s),4), "min:", round(min(f1s),4), "avgP:",sum(precs)/len(precs),"avgR:",round(sum(recalls)/len(recalls),4),"stddev:", round(100*stddev,4))
			
			best_sp = self.stopping_points[0] # just return the 1st one, by default.
			best_conll = -1
			if self.calculateCoNLLScore:
				(best_sp, best_conll, min_conll, max_conll, std_conll) = self.calculateBestKey(spToCoNLL)
				sys.stdout.flush()
				print("* [AGGWD] conll f1 -- best sp:",best_sp, "yielded: min:",round(100*min_conll,4),"avg:",round(100*best_conll,4),"max:",round(max_conll,4),"stddev:",round(std_conll,4))
				
				fout = open("ccnn_agg_dirHalf.csv", "a+")
				fout.write(str(self.args.devDir) + ",wd," + str(self.devMode) + "," + str(best_conll) + "\n")
				fout.close()
				return spToDocPredictedCluster[best_sp], spToPredictedCluster[best_sp], wd_goldenClusters, best_sp
		else: # HDDCRP
			# non-sensical return; only done so that the return handler doesn't complain
			return wd_docPredClusters, wd_predictedClusters, wd_goldenClusters, 0
	'''

	# CROSS-DOC MODEL
	def train_and_test_cd(self):
		f1s = []
		recalls = []
		precs = []
		spToCoNLL = defaultdict(list)
		spToPredictedCluster = {}
		spToDocPredictedCluster = {}
		ensemblePreds = [] # DUMMY INITIALIZER
		while len(f1s) < numRuns:

			bestRunCoNLL = -1
			bestRunSP = -1

			# define model
			input_shape = self.trainX.shape[2:]
			base_network = self.create_base_network(input_shape)

			if self.CCNNSupplement: # relational, merged layer way
				input_a = Input(shape=input_shape, name='input_a')
				input_b = Input(shape=input_shape, name='input_b')
			else:
				input_a = Input(shape=input_shape)
				input_b = Input(shape=input_shape)

			processed_a = base_network(input_a)
			processed_b = base_network(input_b)
			distance = Lambda(self.euclidean_distance, output_shape=self.eucl_dist_output_shape)([processed_a, processed_b])
			
			# CD PART
			if self.CCNNSupplement:
				auxiliary_input = Input(shape=(len(self.supplementalTrain[0]),), name='auxiliary_input')
				combined_layer = keras.layers.concatenate([distance, auxiliary_input])
				#x = Dense(100, activation='relu')(combined_layer)
				#x2 = Dense(5, activation='relu')(x)
				''' # I COMMENTED THIS OUT JUST TO BRING ATTENTION TO THIS BEING CD
				main_output = Dense(1, activation='relu', name='main_output')(x)
				model = Model([input_a, input_b, auxiliary_input], outputs=main_output)
				model.compile(loss=self.contrastive_loss, optimizer=Adam())
				print("model summary:")
				print(model.summary())
				'''
				model.fit({'input_a': self.trainX[:, 0], 'input_b': self.trainX[:, 1], 'auxiliary_input': self.supplementalTrain},
						{'main_output': self.trainY},
						batch_size=self.bs, \
						epochs=self.ne, \
						shuffle=True, \
						validation_data=({'input_a': self.devX[:, 0], 'input_b': self.devX[:, 1], 'auxiliary_input': self.supplementalDev}, {'main_output': self.devY}))
			else:
				model = Model(inputs=[input_a, input_b], outputs=distance)
				model.compile(loss='binary_crossentropy', optimizer=Adam())
				#model.compile(loss=self.contrastive_loss, optimizer=Adam())
				#print(model.summary())
				model.fit([self.trainX[:, 0], self.trainX[:, 1]], self.trainY, \
					batch_size=self.bs, \
					epochs=self.ne, \
					shuffle=True, \
					validation_data=([self.devX[:, 0], self.devX[:, 1]], self.devY))

			if self.CCNNSupplement:
				preds = model.predict({'input_a': self.testX[:, 0], 'input_b': self.testX[:, 1], 'auxiliary_input': self.supplementalTest})
			else:
				if self.devMode:
					preds = model.predict([self.devX[:, 0], self.devX[:, 1]])
				else:
					preds = model.predict([self.testX[:, 0], self.testX[:, 1]])
			print("len of preds:", len(preds))

			if len(f1s) == numRuns-1:
				
				if len(ensemblePreds) > 0:
					preds = ensemblePreds
					print("*** USING ENSEMBLE of PREDIcTIONS!!")

				for sp in self.stopping_points:
					print("* [agg] sp:", sp)
					if self.devMode:
						(cd_docPredClusters, cd_predictedClusters, cd_goldenClusters) = self.aggClusterCD(self.helper.devDirs, self.devID, preds, sp)
					else:
						(cd_docPredClusters, cd_predictedClusters, cd_goldenClusters) = self.aggClusterCD(self.helper.testingDirs, self.testID, preds, sp)
					#(bcub_p, bcub_r, bcub_f1, muc_p, muc_r, muc_f1, ceafe_p, ceafe_r, ceafe_f1, conll_f1)
					start_time = time.time()

					if self.calculateCoNLLScore:
						suffix = "cd_" + str(sp) + "_" + str(len(f1s))

						if self.args.useECBTest: # uses ECB Test Mentions
							scores = get_conll_scores(cd_goldenClusters, cd_predictedClusters)
							print("CONLL SCORES:", scores)
							print("* getting conll score took", str((time.time() - start_time)), "seconds")
							spToCoNLL[sp].append(scores[-1])
							spToPredictedCluster[sp] = cd_predictedClusters
							spToDocPredictedCluster[sp] = cd_docPredClusters

							if scores[-1] > bestRunCoNLL:
								bestRunCoNLL = scores[-1]
								bestRunSP = sp

						else: # uses HDDCRP Test Mentions
							self.helper.writeCoNLLFile(cd_predictedClusters, suffix)

			if self.args.useECBTest:
				(f1, prec, rec, bestThreshold) = self.evaluateCCNNPairwisePreds(preds)
				'''
				print("ccnn_best_f1 (# successful runs: ", len(f1s), "): best_pairwise_f1: ", round(f1,4), " prec: ",round(prec,4), " recall: ", round(rec,4), " threshold: ", round(bestThreshold,3))
				if f1 > 0.20:
					print("**** ADDING TO ENSEMBLE!")
					# append to the ensemble of pairwise predictions
					if self.devMode:
						ensemblePreds = self.addEnsemblePredictions(False, self.helper.devDirs, self.devID, preds)
					else:
						ensemblePreds = self.addEnsemblePredictions(False, self.helper.testingDirs, self.testID, preds)

					f1s.append(f1)
					recalls.append(rec)
					precs.append(prec)
					
				sys.stdout.flush()
				'''
		# clears ram
		self.trainX = None
		self.trainY = None
		
		if self.args.useECBTest:
			stddev = -1
			if len(f1s) > 1:
				stddev = self.standard_deviation(f1s)
			print("pairwise f1 (over",len(f1s),"runs) -- avg:", round(sum(f1s)/len(f1s),4), "max:", round(max(f1s),4), "min:", round(min(f1s),4), "avgP:",sum(precs)/len(precs),"avgR:",round(sum(recalls)/len(recalls),4),"stddev:", round(100*stddev,4))
			
			best_sp = self.stopping_points[0] # just return the 1st one, by default.
			best_conll = -1
			if self.calculateCoNLLScore:
				(best_sp, best_conll, min_conll, max_conll, std_conll) = self.calculateBestKey(spToCoNLL)
				sys.stdout.flush()
				print("* [AGGCD] conll f1 -- best sp:",best_sp, "yielded: min:",round(100*min_conll,4), "avg:",round(100*best_conll,4),"max:",round(max_conll,4),"stddev:",round(std_conll,4))
				fout = open("ccnn_agg_dirHalf.csv", "a+")
				fout.write(str(self.args.devDir) + ",cd," + str(self.devMode) + "," + str(best_conll) + "\n")
				fout.close()
			return (None, None, None, None) # we don't ever care about using the return values

		else: # HDDCRP
			# non-sensical return; only done so that the return handler doesn't complain
			return (cd_docPredClusters, cd_predictedClusters, cd_goldenClusters, 0)

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

			if not m1.isPred or not m2.isPred:
				print("* ERROR: we're trying to do AGG on some non-event mentions")
				exit(1)

			pred = pred[0] # NOTE: the lower the score, the more likely they are the same.  it's a dissimilarity score
			doc_id = m1.doc_id
			if m1.dir_num not in relevant_dirs:
				print("* ERROR: passed in predictions which belong to a dir other than what we specify")
				exit(1)
			if m2.doc_id != doc_id:
				print("* ERROR2: xuids are from diff docs!")
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

	# agglomerative cluster the cross-doc predicted pairs
	# NOTE: 'dir_num' in this function is used to refer to EITHER
	# dirHalf or the actual dir; 
	def aggClusterCD(self, relevant_dirs, ids, preds, stoppingPoint):

		start_time = time.time()
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

			if not m1.isPred or not m2.isPred:
				print("* ERROR: we're trying to do AGG on some non-event mentions")
				exit(1)

			# NOTE: the lower the score, the more likely they are the same.  it's a dissimilarity score
			pred = pred[0]
			doc_id = m1.doc_id
			if m1.dir_num not in relevant_dirs:
				print("* ERROR: passed in predictions which belong to a dir other than what we specify")
				exit(1)
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
			
			if dir_num not in relevant_dirs:
				continue

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
		return (ourClusterSuperSet, goldenSuperSet)

	# Base network to be shared (eq. to feature extraction).
	def create_base_network(self, input_shape):
		seq = Sequential()
		kernel_rows = 1
		print("self.nl:", self.nl)
		#for i in range(self.nl):
		#	nf = self.nf
		#	if i == 1: # meaning 2nd layer, since i in {0,1,2, ...}
		#		nf = 96
		seq.add(Conv2D(self.nf, kernel_size=(kernel_rows, 3), activation='relu', padding="same", input_shape=input_shape, data_format="channels_first"))
		seq.add(Dropout(float(self.do)))
		seq.add(MaxPooling2D(pool_size=(kernel_rows, 2), padding="same", data_format="channels_first"))
		
		seq.add(Conv2D(self.nf, kernel_size=(kernel_rows, 3), activation='relu', padding="same", input_shape=input_shape, data_format="channels_first"))
		seq.add(Dropout(float(self.do)))
		seq.add(MaxPooling2D(pool_size=(kernel_rows, 2), padding="same", data_format="channels_first"))
		
		seq.add(Flatten())
		seq.add(Dense(self.nf, activation='sigmoid'))
		return seq

	def L1_distance(self, vects):
	
		return K.sqrt(K.maximum(K.sum(K.abs(vects[0] - vects[1]), axis=1, keepdims=True), K.epsilon()))



	def euclidean_distance(self, vects):
		#return tf.clip_by_value(K.sqrt(K.maximum(K.sum(K.square(vects[0] - vects[1]), axis=1, keepdims=True), K.epsilon())), 0, 1)
		return K.sqrt(K.maximum(K.sum(K.square(vects[0] - vects[1]), axis=1, keepdims=True), K.epsilon()))

	def eucl_dist_output_shape(self, shapes):
		shape1, _ = shapes
		return (shape1[0], 1)

	def weighted_binary_crossentropy(self, y_true, y_pred):
		epsilon = tf.convert_to_tensor(K.common._EPSILON, y_pred.dtype.base_dtype)
		y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
		y_pred = tf.log(y_pred / (1 - y_pred))
		cost = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, 0.25)
		return K.mean(cost * 0.8, axis=-1)

	# Contrastive loss from Hadsell-et-al.'06
	# http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
	def contrastive_loss(self, y_true, y_pred):
		margin = 1
		return K.mean((1 - y_true)*K.square(y_true - y_pred) + (y_true)*K.square(K.maximum(margin - y_pred, 0)))

	def standard_deviation(self, lst):
		num_items = len(lst)
		if num_items == 1:
			return 0
		
		mean = sum(lst) / num_items
		differences = [x - mean for x in lst]
		sq_differences = [d ** 2 for d in differences]
		variance = sum(sq_differences) / (num_items - 1)
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
