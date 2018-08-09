import pickle
import numpy as np
from collections import defaultdict
class DataHandler:
	def __init__(self, featureHandler, helper):
		self.featureHandler = featureHandler
		self.helper = helper
		self.corpus = helper.corpus
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

	def loadNNData(self, useRelationalFeatures, useCCNN, scope):
		if useCCNN:
			(self.trainID, self.trainX, self.trainY) = self.createDataForCCNN(self.helper.trainingDirs, self.featureHandler.trainXUIDs, useRelationalFeatures, True, scope)
			(self.devID, self.devX, self.devY) = self.createDataForCCNN(self.helper.devDirs, self.featureHandler.devXUIDs, useRelationalFeatures, False, scope)
			#(self.testID, self.testX, self.testY) = self.createDataForCCNN(helper.testingDirs, featureHandler.testXUIDs, useRelationalFeatures, False)
		else: # FOR FFNN and SVM
			(self.trainID, self.trainX, self.trainY) = self.createDataForFFNN(self.helper.trainingDirs, self.featureHandler.trainXUIDs, useRelationalFeatures, True, scope)
			(self.devID, self.devX, self.devY) = self.createDataForFFNN(self.helper.devDirs, self.featureHandler.devXUIDs, useRelationalFeatures, False, scope)


	def loadFeature(self, file):
		print("loading",file)
		return pickle.load(open(file, 'rb'))

	# return a list of XUID pairs, based on what was passed-in,
	# where each pair comes from either:
	# (1) 'doc' = the same doc (within-dic); or,
	# (2) 'dirHalf' = the same dirHalf (but not same doc)
	# (3) 'dir' = the same dir (but not same doc)
	def createXUIDPairs(self, XUIDs, scope):
		if scope != "doc" and scope != "dirHalf" and scope != "dir":
			print("* ERROR: scope must be doc, dirHalf, or dir")
			exit(1)
		xuidPairs = set()
		
		dirHalfToXUIDs = defaultdict(set)
		ECBDirToXUIDs = defaultdict(set)
		
		for xuid in XUIDs:
			mention = self.corpus.XUIDToMention[xuid]
			dirHalfToXUIDs[mention.dirHalf].add(xuid)
			ECBDirToXUIDs[mention.dir_num].add(xuid)

			# NOTE: left off here
		
		# we sort to ensure consistency across multiple runs
		print("# mentions passed-in:", len(XUIDs))
		for ecb_dir in sorted(ECBDirToXUIDs.keys()):

		#for dirHalf in sorted(dirHalfToXUIDs.keys()):
			for xuid1 in sorted(ECBDirToXUIDs[ecb_dir]):
				for xuid2 in sorted(ECBDirToXUIDs[ecb_dir]):
					if xuid2 <= xuid1:
						continue
					inSameDoc = False
					if self.corpus.XUIDToMention[xuid1].doc_id == self.corpus.XUIDToMention[xuid2].doc_id:
						inSameDoc = True
					if scope == "doc" and inSameDoc:
						xuidPairs.add((xuid1, xuid2))
					elif scope == "dirHalf" and not inSameDoc:
						xuidPairs.add((xuid1, xuid2))
					elif scope == "dir" and not inSameDoc:
						xuidPairs.add((xuid1, xuid2))
		return xuidPairs
	
	# almost identical to createData() but it re-shapes the vectors to be 5D -- pairwise.
	# i could probably combine this into 1 function and have a boolean flag isCCNN=True.
	# we pass in XUID because the mentions could be from any Stan, HDDCRP, or ECB; however,
	# we need to remember that the co-reference REF tags only exist in the output file that we compare against
	def createDataForCCNN(self, dirs, XUIDs, useRelationalFeatures, negSubsample, scope):
		pairs = []
		X = []
		Y = []
		labels = []
		numFeatures = 0
		numPosAdded = 0
		numNegAdded = 0
		xuidPairs = self.createXUIDPairs(XUIDs, scope)

		print("# pairs: ", len(xuidPairs))
		# TMP ADDED FOR SAME LEMMA TEST
		'''
		TN = 0.0
		TP = 0.0
		FN = 0.0
		FP = 0.0
		'''
		for (xuid1, xuid2) in xuidPairs:
			if xuid1 == xuid2:
				print("whaaaaa: xuidPairs:", xuidPairs)
			
			# TMP ADDED FOR SAME LEMMA TEST
			'''
			sameLemma = True
			lemmaList1 = []
			lemmaList2 = []
			for t1 in self.corpus.XUIDToMention[xuid1].tokens:
				lemma = self.getBestStanToken(t1.stanTokens).lemma.lower()
				lemmaList1.append(lemma)
			for t2 in self.corpus.XUIDToMention[xuid2].tokens:
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

			# NOTE: if this is for HDDCRP or Stan mentions, the REFs will always be True
			# because we don't have such info for them, so they are ""
			if self.corpus.XUIDToMention[xuid1].REF == self.corpus.XUIDToMention[xuid2].REF:
				
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
			(uid1, uid2) = sorted([self.corpus.XUIDToMention[xuid1].UID, self.corpus.XUIDToMention[xuid2].UID])
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
				exit(1)
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

			# makes xuid (aka xuid) pairs
			pairs.append((xuid1, xuid2))

		X = np.asarray(X)
		#print("labels:",labels)
		Y = np.asarray(labels)
		pp = float(numPosAdded / (numPosAdded+numNegAdded))
		pn = float(numNegAdded / (numPosAdded+numNegAdded))
		print("* createData() loaded", len(pairs), "pairs (", pp, "% pos, ",
		      pn, "% neg); features' length = ", numFeatures)
			  		
		if len(pairs) == 0:
			exit(1)
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
	# [(xuid1,xuid2), [features], [1,0]]
	def createDataForFFNN(self, dirs, XUIDs, useRelationalFeatures, negSubsample, scope):
		pairs = []
		X = []
		Y = []
		numFeatures = 0
		numPosAdded = 0
		numNegAdded = 0
		xuidPairs = self.createXUIDPairs(XUIDs, scope)
		for (xuid1, xuid2) in xuidPairs:
			label = [1, 0]
			# NOTE: if this is for HDDCRP or Stan mentions, the REFs will always be True
			# because we don't have such info for them, so they are ""
			if self.corpus.XUIDToMention[xuid1].REF == self.corpus.XUIDToMention[xuid2].REF:
				label = [0, 1]
				numPosAdded += 1
			else:
				if negSubsample and numNegAdded > numPosAdded*self.args.numNegPerPos:
					continue
				numNegAdded += 1

			features = []
			(uid1, uid2) = sorted([self.corpus.XUIDToMention[xuid1].UID, self.corpus.XUIDToMention[xuid2].UID])
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

			pairs.append((xuid1, xuid2))
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