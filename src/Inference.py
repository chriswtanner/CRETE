import pickle
import numpy as np
from collections import defaultdict
class Inference:
	def __init__(self, featureHandler, helper, useRelationalFeatures, useWD, useCCNN):
		self.featureHandler = featureHandler
		self.helper = helper
		self.corpus = helper.corpus
		self.useWD = useWD
		self.args = featureHandler.args

		self.badPOS = ["‘’", "``", "POS", "$", "''"] # TMP FOR SAMELEMMA TEST
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

		# FOR CCNN
		if useCCNN:
			(self.trainID, self.trainX, self.trainY) = self.createDataForCCNN(helper.trainingDirs, featureHandler.trainMUIDs, useRelationalFeatures, True)
			(self.devID, self.devX, self.devY) = self.createDataForCCNN(helper.devDirs, featureHandler.devMUIDs, useRelationalFeatures, False)
			#(self.testID, self.testX, self.testY) = self.createDataForCCNN(helper.testingDirs, featureHandler.testMUIDs, useRelationalFeatures, False)
		else:
			# FOR FFNN
			(self.trainID, self.trainX, self.trainY) = self.createData(helper.trainingDirs, featureHandler.trainMUIDs, useRelationalFeatures, True)
			(self.devID, self.devX, self.devY) = self.createData(helper.devDirs, featureHandler.devMUIDs, useRelationalFeatures, False)
			
	def loadFeature(self, file):
		print("loading",file)
		return pickle.load(open(file, 'rb'))

	# return a list of MUID pairs, where each pair comes from either:
	# (1) the same doc (within-dic); or,
	# (2) the same dirHalf (cross-doc)
	def createMUIDPairs(self, MUIDs):
		muidPairs = set()
		dirHalfToMUIDs = defaultdict(set)
		docToMUIDs = defaultdict(set)
		for muid in MUIDs:
			mention = self.corpus.XUIDToMention[muid]
			dirHalfToMUIDs[mention.dirHalf].add(muid)

		# we sort to ensure consistency across multiple runs
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
	
	# almost identical to createData() but it re-shapes the vectors to be 5D -- pairwise.
	# i could probably combine this into 1 function and have a boolean flag isCCNN=True
	def createDataForCCNN(self, dirs, MUIDs, useRelationalFeatures, negSubsample):
		pairs = []
		X = []
		Y = []
		labels = []
		numFeatures = 0
		numPosAdded = 0
		numNegAdded = 0
		muidPairs = self.createMUIDPairs(MUIDs)

		# TMP ADDED FOR SAME LEMMA TEST
		'''
		TN = 0.0
		TP = 0.0
		FN = 0.0
		FP = 0.0
		'''
		for (muid1, muid2) in muidPairs:
			if muid1 == muid2:
				print("whaaaaa: muidPairs:", muidPairs)
			
			# TMP ADDED FOR SAME LEMMA TEST
			'''
			sameLemma = True
			lemmaList1 = []
			lemmaList2 = []
			for t1 in self.corpus.XUIDToMention[muid1].tokens:
				lemma = self.getBestStanToken(t1.stanTokens).lemma.lower()
				lemmaList1.append(lemma)
			for t2 in self.corpus.XUIDToMention[muid2].tokens:
				lemma = self.getBestStanToken(t2.stanTokens).lemma.lower()
				lemmaList2.append(lemma)
			if len(lemmaList1) != len(lemmaList2):
				sameLemma = False
			else:
				for _ in range(len(lemmaList1)):
					if lemmaList1[_] != lemmaList2[_]:
						sameLemma = False
						break
			'''
			if self.corpus.XUIDToMention[muid1].REF == self.corpus.XUIDToMention[muid2].REF:
				
				# TMP ADDED FOR SAME LEMMA TEST
				'''
				if sameLemma:
					TP += 1
				else:
					FN += 1
				'''
				labels.append(1)
				numPosAdded += 1
			else:

				# TMP ADDED FOR SAME LEMMA TEST
				'''
				if not sameLemma:
					TN += 1
				else:
					FP += 1
				'''
				if negSubsample and numNegAdded > numPosAdded*self.args.numNegPerPos:
					continue
				numNegAdded += 1
				labels.append(0)
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

			# makes xuid (aka muid) pairs
			pairs.append((muid1, muid2))

		X = np.asarray(X)
		#print("labels:",labels)
		Y = np.asarray(labels)
		print("features have a length of:", numFeatures)
		print("* createData() loaded", len(pairs), "pairs")

		# TMP ADDED FOR SAME LEMMA TEST
		'''
		recall = 0
		if (TP + FN) > 0:
			recall = float(TP / (TP + FN))
		prec = 0
		if (TP + FP) > 0:
			prec = float(TP / (TP + FP))
		f1 = 0
		if (recall + prec) > 0:
			f1 = 2*(recall*prec) / (recall + prec)
		print("samelamma f1:",f1, prec, recall)
		'''
		return (pairs, X, Y)

	# creates data for FFNN and SVM:
	# [(muid1,muid2), [features], [1,0]]
	def createData(self, dirs, MUIDs, useRelationalFeatures, negSubsample):
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
				if negSubsample and numNegAdded > numPosAdded*self.args.numNegPerPos:
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
		pp = float(numPosAdded / (numPosAdded+numNegAdded))
		pn = float(numNegAdded / (numPosAdded+numNegAdded))
		print("* createData() loaded", len(pairs), "pairs (",pp,"% pos, ",pn,"% neg)")
		return (pairs, X, Y)

	# TMP, REMOVE THIS.  only used for testing samelemma in a rush
	# this method normally resides in a diff class, just we didn't have access to it from this class
	def getBestStanToken(self, stanTokens, token=None):
		longestToken = ""
		bestStanToken = None
		for stanToken in stanTokens:
			if stanToken.pos in self.badPOS:
				# only use the badPOS if no others have been set
				if bestStanToken == None:
					bestStanToken = stanToken
			else:  # save the longest, nonBad POS tag
				if len(stanToken.text) > len(longestToken):
					longestToken = stanToken.text
					bestStanToken = stanToken
		if len(stanTokens) > 1 and token != None:
			print("token:", str(token.text), "=>", str(bestStanToken))
		if bestStanToken == None:
			print("* ERROR: our bestStanToken is empty!")
			exit(1)
		return bestStanToken
