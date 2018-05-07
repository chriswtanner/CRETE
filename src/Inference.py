import pickle
import numpy as np
from collections import defaultdict
class Inference:
	def __init__(self, featureHandler, helper, useRelationalFeatures, useWD):
		self.featureHandler = featureHandler
		self.helper = helper
		self.corpus = helper.corpus
		self.useWD = useWD
		self.args = featureHandler.args

		self.singleFeatures = []
		self.relFeatures = []
		if self.args.wordFeature:
			lf = self.loadFeature("../data/features/word.f")
			self.singleFeatures.append(lf.singles)
			self.relFeatures.append(lf.relational)
		if self.args.lemmaFeature:
			lf = self.loadFeature("../data/features/lemma.f")
			self.singleFeatures.append(lf.singles)
			self.relFeatures.append(lf.relational)
		if self.args.charFeature:
			lf = self.loadFeature("../data/features/char.f")
			self.singleFeatures.append(lf.singles)
			self.relFeatures.append(lf.relational)
		if self.args.posFeature:
			lf = self.loadFeature("../data/features/pos.f")
			self.singleFeatures.append(lf.singles)
			self.relFeatures.append(lf.relational)
		if self.args.dependencyFeature:
			lf = self.loadFeature("../data/features/dependency.f")
			self.singleFeatures.append(lf.singles)
			self.relFeatures.append(lf.relational)
		if self.args.bowFeature:
			lf = self.loadFeature("../data/features/bow.f")
			self.singleFeatures.append(lf.singles)
			self.relFeatures.append(lf.relational)
		if self.args.wordnetFeature:
			lf = self.loadFeature("../data/features/wordnet.f")
			self.relFeatures.append(lf.relational)

		(self.trainID, self.trainX, self.trainY) = self.createDataForCCNN(helper.trainingDirs, featureHandler.trainMUIDs, useRelationalFeatures)
		(self.devID, self.devX, self.devY) = self.createDataForCCNN(helper.devDirs, featureHandler.devMUIDs, useRelationalFeatures)

		'''
		(self.trainID, self.trainX, self.trainY) = self.createData(helper.trainingDirs, featureHandler.trainMUIDs, useRelationalFeatures)
		(self.devID, self.devX, self.devY) = self.createData(helper.devDirs, featureHandler.devMUIDs, useRelationalFeatures)
		'''
	def loadFeature(self, file):
		print("loading",file)
		return pickle.load(open(file, 'rb'))

	# return a list of MUID pairs, where each pair comes from the same dirHalf
	def createMUIDPairs(self, MUIDs):
		muidPairs = set()
		dirHalfToMUIDs = defaultdict(set)
		for muid in MUIDs:
			mention = self.corpus.XUIDToMention[muid]
			dirHalfToMUIDs[mention.dirHalf].add(muid)
		# we sort to ensure consistency
		for dirHalf in sorted(dirHalfToMUIDs.keys()):
			for muid1 in sorted(dirHalfToMUIDs[dirHalf]):
				for muid2 in sorted(dirHalfToMUIDs[dirHalf]):
					if muid2 <= muid1:
						continue
					inSameDoc = False
					if self.corpus.XUIDToMention[muid1].doc_id == self.corpus.XUIDToMention[muid2].doc_id:
						inSameDoc = True
					if self.useWD and inSameDoc:
						muidPairs.add((muid1, muid2))
					elif not self.useWD and not inSameDoc:
						muidPairs.add((muid1, muid2))
		return muidPairs
	
	def createDataForCCNN(self, dirs, MUIDs, useRelationalFeatures):
		pairs = []
		X = []
		Y = []
		labels = []
		numFeatures = 0
		muidPairs = self.createMUIDPairs(MUIDs)
		for (muid1, muid2) in muidPairs:
			m1_features = []
			m2_features = []
			(uid1, uid2) = sorted([self.corpus.XUIDToMention[muid1].UID, self.corpus.XUIDToMention[muid2].UID])
			# loops through each feature (e.g., BoW, lemma) for the given uid pair
			for feature in self.singleFeatures:
				for i in feature[uid1]:  # loops through each val of the given feature
					m1_features.append(i)
				for i in feature[uid2]:
					m2_features.append(i)
			# loops through each feature (e.g., BoW, lemma) for the given uid pair
			if useRelationalFeatures:
				for feature in self.relFeatures:
					if (uid1, uid2) not in feature:
						print("not in")
						exit(1)
					for i in feature[(uid1, uid2)]:
						m1_features.append(i)
						m2_features.append(i)
			if len(m1_features) != numFeatures and numFeatures != 0:
				print("* ERROR: # features diff:", len(m1_features), "and", numFeatures)
			numFeatures = len(m1_features)

			if len(m1_features) != len(m2_features):
				print("* ERROR: m1 and m2 have diff feature emb lengths")
				exit(1)
			# make the joint embedding
			m1Matrix = np.zeros(shape=(1, len(m1_features)))
			m2Matrix = np.zeros(shape=(1, len(m2_features)))
			m1Matrix[0] = m1_features
			m2Matrix[0] = m2_features
			m1Matrix = np.asarray(m1Matrix).reshape(1,len(m1_features),1)
			m2Matrix = np.asarray(m2Matrix).reshape(1, len(m2_features),1)
			pair = np.asarray([m1Matrix, m2Matrix])
			X.append(pair)

			# makes label
			if self.corpus.XUIDToMention[muid1].REF == self.corpus.XUIDToMention[muid2].REF:
				labels.append(1)
			else:
				labels.append(0)

			# makes xuid (aka muid) pairs
			pairs.append((muid1, muid2))

		X = np.asarray(X)
		Y = np.asarray(labels)
		print("features have a length of:", numFeatures)
		print("* createData() loaded", len(pairs), "pairs")
		return (pairs, X, Y)

	# creates data for FFNN and SVM:
	# [(muid1,muid2), [features], [1,0]]
	def createData(self, dirs, MUIDs, useRelationalFeatures):
		pairs = []
		X = []
		Y = []
		numFeatures = 0
		numPosAdded = 0
		numNegAdded = 0
		muidPairs = self.createMUIDPairs(MUIDs)
		for (muid1, muid2) in muidPairs:

			label = [1, 0]
			if self.corpus.XUIDToMention[muid1].REF == self.corpus.XUIDToMention[muid2].REF:
				label = [0, 1]
				numPosAdded += 1
			else:
				if numNegAdded > numPosAdded*self.numNegPerPos:
					continue
				numNegAdded += 1

			features = []
			(uid1, uid2) = sorted([self.corpus.XUIDToMention[muid1].UID, self.corpus.XUIDToMention[muid2].UID])
			# loops through each feature (e.g., BoW, lemma) for the given uid pair
			for feature in self.singleFeatures:
				for i in feature[uid1]: # loops through each val of the given feature
					features.append(i)
				for i in feature[uid2]:
					features.append(i)
			# loops through each feature (e.g., BoW, lemma) for the given uid pair
			if useRelationalFeatures:
				for feature in self.relFeatures:
					if (uid1, uid2) not in feature:
						print("not in")
						exit(1)
					for i in feature[(uid1, uid2)]:
						features.append(i)
			if len(features) != numFeatures and numFeatures != 0:
				print("* ERROR: # features diff:",len(features),"and",numFeatures)
			numFeatures = len(features)

			pairs.append((muid1, muid2))
			X.append(features)
			Y.append(label)
		print("features have a length of:",numFeatures)
		print("* createData() loaded",len(pairs), "pairs")
		return (pairs, X, Y)
