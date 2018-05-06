import os
import sys
import random
import numpy as np
#from LibSVM import LibSVM
from sklearn import svm
from sklearn.svm import LinearSVC
#from sklearn.datasets import make_classification
from math import sqrt, floor
from collections import defaultdict
class LibSVM:
	def __init__(self, helper, coref):
		self.helper = helper
		self.corpus = helper.corpus
		self.args = helper.args
		(self.trainID, self.trainX, self.trainY) = (coref.trainID, coref.trainX, coref.trainY)
		(self.devID, self.devX, self.devY) = (coref.devID, coref.devX, coref.devY)

		# transforms data to proper format for libsvm
		self.trainY = [i.index(max(i)) for i in self.trainY]
		#self.devY = [i.index(max(i)) for i in self.devY]
		
		self.num_epochs = int(self.args.numEpochs)
		pos_ratio = 0.8
		neg_ratio = 1. - pos_ratio

	def train_and_test(self, numRuns):
		f1s = []
		weights = [1,5,10]
		t = [0.01, 0.0001, 0.000001]
		for w1 in weights:
			for to in t:
				print("w1:", w1, "to",to)
				#print(self.trainY)
				#print(len(self.trainX))
				clf = LinearSVC(C=1, class_weight={0:w1,1:1}, dual=True, fit_intercept=True,
								intercept_scaling=1, loss='squared_hinge', max_iter=5000,
								multi_class='ovr', penalty='l2', random_state=0, tol=to,
								verbose=1)
				clf.fit(self.trainX, self.trainY)
				'''
				clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
								decision_function_shape='ovr', degree=3, gamma='auto',
								kernel='linear', max_iter=-1, probability=False, random_state=None,
								shrinking=True, tol=0.001, verbose=False)
				'''
				#print(clf.coef_)
				#print(clf.intercept_)
				preds = clf.predict(self.devX)
				TN = 0.0
				TP = 0.0
				FN = 0.0
				FP = 0.0
				for _ in range(len(preds)):
					pred_label = preds[_]
					gold_label = 0
					if self.devY[_][1] >= self.devY[_][0]:
						gold_label = 1
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
