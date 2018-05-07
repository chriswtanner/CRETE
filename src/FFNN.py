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
	def __init__(self, helper, coref):
		self.helper = helper
		self.corpus = helper.corpus
		self.args = helper.args
		(self.trainID, self.trainX, self.trainY) = (coref.trainID, coref.trainX, coref.trainY)
		(self.devID, self.devX, self.devY) = (coref.devID, coref.devX, coref.devY)

		if self.args.native:
			sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
			os.environ['CUDA_VISIBLE_DEVICES'] = ''

		# model params
		self.model = None
		self.hidden_size = 200
		self.dataDim = len(self.trainX[0])
		self.outputDim = 2
		self.batch_size = 5
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

	def train_and_test(self, numRuns):
		f1s = []
		for i in range(numRuns):
			self.model = Sequential()

			# optionally add a 3rd layer
			
			if self.dataDim > 1000:
				h1 = floor(self.dataDim / 2)
				print("h1:",h1)
				self.model.add(Dense(units=h1, input_shape=(self.dataDim,), use_bias=True, kernel_initializer='normal'))
				self.model.add(Activation('relu'))
				if h1 > 1600:
					h2 = floor(h1 / 4.0)
					self.model.add(Dense(units=h2, \
					use_bias=True, kernel_initializer='normal'))
					self.model.add(Activation('relu'))
				self.model.add(Dense(units=self.hidden_size, use_bias=True, kernel_initializer='normal'))
			else:
				self.model.add(Dense(units=self.hidden_size, input_shape=(self.dataDim,), use_bias=True, kernel_initializer='normal'))
			self.model.add(Activation('relu'))
			self.model.add(Dense(units=50, use_bias=True, kernel_initializer='normal'))
			self.model.add(Activation('relu'))
			self.model.add(Dense(units=2, use_bias=True, kernel_initializer='normal'))
			self.model.add(Activation('softmax'))
			self.model.compile(loss=self.weighted_binary_crossentropy, optimizer=Adam(lr=0.001), metrics=['accuracy'])
			self.model.summary()
			self.model.fit(self.trainX, self.trainY, epochs=self.num_epochs, batch_size=self.batch_size, verbose=1)
			
			preds = self.model.predict(self.devX)

			TN = 0.0
			TP = 0.0
			FN = 0.0
			FP = 0.0
			for _ in range(len(preds)):
				pred = preds[_]
				pred_label = 0
				gold_label = 0
				if self.devY[_][1] >= self.devY[_][0]:
					gold_label = 1
				if pred[1] >= pred[0]:
					pred_label = 1
				if pred_label and gold_label:
					TP += 1
				elif pred_label and not gold_label:
					FP += 1
				elif not pred_label and gold_label:
					FN += 1
				elif not pred_label and not gold_label:
					TN += 1
				else:
					print("what happened")
					exit(1)
			recall = 0
			if (TP + FN) > 0:
				recall = float(TP / (TP + FN))
			prec = 0
			if (TP + FP) > 0:
				prec = float(TP / (TP + FP))
			acc = float((TP + TN) / len(preds))
			f1 = 0
			if (recall + prec) > 0:
				f1 = 2*(recall*prec) / (recall + prec)
			f1s.append(f1)
			print("acc:", acc, "r:", recall, "p:", prec, "f1:", f1)
			sys.stdout.flush()
		# clears ram
		self.trainX = None
		self.trainY = None
		stddev = -1
		if len(f1s) > 1:
			stddev = self.standard_deviation(f1s)
		print("avgf1:", sum(f1s)/len(f1s), "max:",max(f1s), "min:",min(f1s), "stddev:", stddev)
		sys.stdout.flush()
	def standard_deviation(self, lst):
		num_items = len(lst)
		mean = sum(lst) / num_items
		differences = [x - mean for x in lst]
		sq_differences = [d ** 2 for d in differences]
		ssd = sum(sq_differences)
		variance = ssd / (num_items - 1)
		return sqrt(variance)