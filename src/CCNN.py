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

		if self.args.native:
			sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
			os.environ['CUDA_VISIBLE_DEVICES'] = ''

	def train_and_test(self, numRuns):
		f1s = []
		thresholds = np.linspace(0, 1, 50)
		for i in range(numRuns):
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
			
			for threshold in thresholds:
				TN = 0.0
				TP = 0.0
				FN = 0.0
				FP = 0.0
				for _ in range(len(preds)):
					pred = preds[_][0]
					pred_label = 0
					gold_label = self.devY[_]
					if pred < threshold:
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
				print("thresh:",threshold,"acc:", acc, "r:", recall, "p:", prec, "f1:", f1)
				sys.stdout.flush()
		# clears ram
		self.trainX = None
		self.trainY = None
		stddev = -1
		if len(f1s) > 1:
			stddev = self.standard_deviation(f1s)
		print("avgf1:", sum(f1s)/len(f1s), "max:", max(f1s), "min:", min(f1s), "stddev:", stddev)
		sys.stdout.flush()

	# Base network to be shared (eq. to feature extraction).
	def create_base_network(self, input_shape):
		seq = Sequential()
		curNumFilters = self.args.numFilters
		kernel_rows = 1
		seq.add(Conv2D(self.args.numFilters, kernel_size=(kernel_rows, 3), activation='relu', padding="same", input_shape=input_shape, data_format="channels_first"))
		seq.add(Dropout(float(self.args.dropout)))
		
		seq.add(Conv2D(curNumFilters, kernel_size=(kernel_rows, 3), activation='relu', padding="same", data_format="channels_first"))
		seq.add(MaxPooling2D(pool_size=(kernel_rows, 2), padding="same", data_format="channels_first"))
		
		seq.add(Conv2D(self.args.numFilters, kernel_size=(kernel_rows, 3), activation='relu', padding="same", input_shape=input_shape, data_format="channels_first"))
		seq.add(Dropout(float(self.args.dropout)))

		seq.add(Conv2D(curNumFilters, kernel_size=(kernel_rows, 3), activation='relu', padding="same", data_format="channels_first"))
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
