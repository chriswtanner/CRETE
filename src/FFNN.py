import os
import sys
import keras
import random
import numpy as np
import tensorflow as tf
import keras.backend as K
from math import sqrt, floor
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam
from collections import defaultdict
class FFNN:
	def __init__(self, helper, dh, scope, devMode):
		self.helper = helper
		self.dh = dh
		self.scope = scope # used by aggClusterCD()
		self.devMode = devMode

		self.corpus = helper.corpus
		self.args = helper.args
		
		self.dh.loadNNData("None", False, self.scope) # False means use FFNN

		(self.trainID, self.trainX, self.trainY) = (dh.trainID, dh.trainX, dh.trainY)
		(self.devID, self.devX, self.devY) = (dh.devID, dh.devX, dh.devY)
		(self.testID, self.testX, self.testY) = (dh.testID, dh.testX, dh.testY)

		self.trainX = np.array(self.trainX)
		self.trainY = np.array(self.trainY)
		self.devX = np.array(self.devX)
		self.devY = np.array(self.devY)
		self.testX = np.array(self.testX)
		self.testY = np.array(self.testY)

		print("# trainx:", len(self.trainX))
		print("self.trainX:", len(self.trainX[0]))
		print("self.trainX:", self.trainX[0])
		if self.args.native:
			tf.Session(config=tf.ConfigProto(log_device_placement=True))
			os.environ['CUDA_VISIBLE_DEVICES'] = ''

		# model params
		self.model = None
		self.hidden_size = 200
		self.dataDim = len(self.trainX[0])
		self.outputDim = 2
		self.batch_size = 1
		self.num_epochs = int(self.args.numEpochs)
		pos_ratio = 0.8
		neg_ratio = 1. - pos_ratio
		self.pos_ratio = tf.constant(pos_ratio, tf.float32)
		self.weights=  tf.constant(neg_ratio / pos_ratio, tf.float32)
		print("datadim:",self.dataDim)

	def weighted_binary_crossentropy(self, y_true, y_pred):
		epsilon = tf.convert_to_tensor(K.common._EPSILON, y_pred.dtype.base_dtype)
		y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
		y_pred = tf.log(y_pred / (1 - y_pred))
		cost = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, self.weights)
		return K.mean(cost * self.pos_ratio, axis=-1)

	def train_and_test(self):
		f1s = []
		recalls = []
		precs = []
		numRuns = 2
		while len(f1s) < numRuns:
			self.model = Sequential()

			# optionally add a 3rd layer
			if self.dataDim > 2000:
				h1 = floor(self.dataDim / 2)
				print("h1:",h1)
				self.model.add(Dense(units=h1, input_shape=(self.dataDim,), activation='sigmoid', use_bias=True, kernel_initializer='normal'))
				#self.model.add(Activation('sigmoid'))
				if h1 > 1600:
					h2 = floor(h1 / 4.0)
					self.model.add(Dense(units=h2, \
					use_bias=True, kernel_initializer='normal'))
					self.model.add(Activation('relu'))
				self.model.add(Dense(units=self.hidden_size, use_bias=True, kernel_initializer='normal'))
			else:
				self.model.add(Dense(units=self.hidden_size, input_shape=(self.dataDim,), use_bias=True, kernel_initializer='normal'))
			#self.model.add(Dropout(0.2))
			#self.model.add(Activation('relu'))
			self.model.add(Dense(units=200, use_bias=True, activation='relu', kernel_initializer='normal'))
			#self.model.add(Activation('relu'))
			#self.model.add(Dropout(0.1))
			self.model.add(Dense(units=50, use_bias=True, activation='relu', kernel_initializer='normal'))
			#self.model.add(Activation('relu'))
			self.model.add(Dense(units=2, use_bias=True, activation='softmax', kernel_initializer='normal'))
			#self.model.add(Activation('softmax'))
			self.model.compile(loss=self.weighted_binary_crossentropy, optimizer=Adam(lr=0.001), metrics=['accuracy'])
			self.model.summary()
			self.model.fit(self.trainX, self.trainY, epochs=self.num_epochs, shuffle=True, batch_size=self.batch_size, verbose=1)
			
			preds = self.model.predict(self.devX)
			print("# preds:",len(preds)) #, preds)
			numGoldPos = 0
			scoreToGoldTruth = defaultdict(list)
			for _ in range(len(preds)):
				pred = preds[_][1]
				if self.devY[_][1] > self.devY[_][0]:
					numGoldPos += 1
					scoreToGoldTruth[pred].append(1)
				else:
					scoreToGoldTruth[pred].append(0)
			s = sorted(scoreToGoldTruth.keys())
			'''
			TN = 0.0
			TP = 0.0
			FN = 0.0
			FP = 0.0
			numReturnedSoFar = 0
			'''
			bestF1 = 0
			bestVal = -1
			bestR = 0
			bestP = 0

			#s = [xy for xy in np.arange(1, 0, -0.1)]
			for eachVal in s:
				TN = 0.0
				TP = 0.0
				FN = 0.0
				FP = 0.0
				numReturnedSoFar = 0
				numCorrect = 0
				numIncorrect = 0
				for s2 in scoreToGoldTruth.keys():
					for _ in scoreToGoldTruth[s2]:
						if s2 >= eachVal:
							if _ == 1:
								TP += 1
								numCorrect += 1
							else:
								FP += 1
								numIncorrect += 1
							numReturnedSoFar += 1
						else:
							if _ == 0:
								numCorrect += 1
							else:
								numIncorrect += 1
				recall = float(TP / numGoldPos)
				prec = 0
				acc = float(numCorrect / (numCorrect + numIncorrect))
				if numReturnedSoFar > 0:
					prec = float(TP / numReturnedSoFar)
				f1 = 0
				if (recall + prec) > 0:
					f1 = 2*(recall*prec) / (recall + prec)
				if f1 > bestF1:
					bestF1 = f1
					bestVal = eachVal
					bestR = recall
					bestP = prec

				print("ffnn:", eachVal, "(", numReturnedSoFar,"returned)","r:",recall,"p:",prec,"f1:",f1,"acc:",acc)
				sys.stdout.flush()
			if bestF1 > 0:
				f1s.append(bestF1)
				recalls.append(bestR)
				precs.append(bestP)
			print("ffnn_best_f1 (run ", len(f1s), "): ", bestF1, " prec: ", bestP, " recall: ", bestR, " threshold:", bestVal, sep="")
			sys.stdout.flush()
		# clears ram
		self.trainX = None
		self.trainY = None
		stddev = -1
		if len(f1s) > 1:
			stddev = self.standard_deviation(f1s)
		print("avgf1 (over", len(f1s), "runs):", sum(f1s)/len(f1s), "max:", max(f1s), "min:", min(f1s), "avgP:", sum(precs)/len(precs), "avgR:", sum(recalls)/len(recalls), "stddev:", stddev)
		sys.stdout.flush()

	def standard_deviation(self, lst):
		num_items = len(lst)
		mean = sum(lst) / num_items
		differences = [x - mean for x in lst]
		sq_differences = [d ** 2 for d in differences]
		ssd = sum(sq_differences)
		variance = ssd / (num_items - 1)
		return sqrt(variance)
