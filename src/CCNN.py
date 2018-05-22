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
class CCNN:
	def __init__(self, helper, coref):
		self.helper = helper
		self.corpus = helper.corpus
		self.args = helper.args
		(self.trainID, self.trainX, self.trainY) = (coref.trainID, coref.trainX, coref.trainY)
		(self.devID, self.devX, self.devY) = (coref.devID, coref.devX, coref.devY)
		(self.testID, self.testX, self.testY) = (coref.testID, coref.testX, coref.testY)

		if self.args.native:
			sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
			os.environ['CUDA_VISIBLE_DEVICES'] = ''

	def train_and_test(self, numRuns):
		f1s = []
		recalls = []
		precs = []
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
			print(model.summary())
			model.fit([self.trainX[:, 0], self.trainX[:, 1]], self.trainY, \
				batch_size=self.args.batchSize, \
				epochs=self.args.numEpochs, \
				validation_data=([self.devX[:, 0], self.devX[:, 1]], self.devY))


			preds = model.predict([self.devX[:, 0], self.devX[:, 1]])
			#preds = model.predict([self.testX[:, 0], self.testX[:, 1]])
			
			numGoldPos = 0
			scoreToGoldTruth = defaultdict(list)
			for _ in range(len(preds)):
				if self.testY[_]:
					numGoldPos += 1
					scoreToGoldTruth[preds[_][0]].append(1)
				else:
					scoreToGoldTruth[preds[_][0]].append(0)
			#print("numGoldPos:", numGoldPos)
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
				#print("fwd:", eachVal, "(", numReturnedSoFar,"returned)",recall,prec,"f1:",f1)
			print("fwd_best_f1:",bestF1,"val:",bestVal)
			sys.stdout.flush()
			if bestF1 > 0:
				f1s.append(bestF1)
				recalls.append(bestR)
				precs.append(bestP)
		# clears ram
		self.trainX = None
		self.trainY = None
		stddev = -1
		if len(f1s) > 1:
			stddev = self.standard_deviation(f1s)
		print("avgf1:", sum(f1s)/len(f1s), "max:", max(f1s), "min:", min(f1s), "avgP:",sum(bestP)/len(bestP),"avgR:",sum(bestR)/len(bestR),"stddev:", stddev)
		sys.stdout.flush()

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
