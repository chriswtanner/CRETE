import pickle
from collections import defaultdict
class Inference:
	def __init__(self, numRuns, featureHandler, helper, useRelationalFeatures, useWD):
		self.numRuns = numRuns
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

		(self.trainID, self.trainX, self.trainY) = self.createData(helper.trainingDirs, featureHandler.trainMUIDs, useRelationalFeatures)
		(self.devID, self.devX, self.devY) = self.createData(helper.devDirs, featureHandler.devMUIDs, useRelationalFeatures)

	def evaluate(self, preds):
		print(1.0)

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
	
	# creates data for FFNN and SVM:
	# [(muid1,muid2), [features], [1,0]]
	def createData(self, dirs, MUIDs, useRelationalFeatures):
		pairs = []
		X = []
		Y = []
		muidPairs = self.createMUIDPairs(MUIDs)
		for (muid1, muid2) in muidPairs:
			features = []
			uid1 = self.corpus.XUIDToMention[muid1].UID
			uid2 = self.corpus.XUIDToMention[muid2].UID
			# loops through each feature (e.g., BoW, lemma) for the given uid pair
			for feature in self.singleFeatures:
				for i in feature[uid1]: # loops through each val of the given feature
					features.append(i)
				for i in feature[uid2]:
					features.append(i)
			# loops through each feature (e.g., BoW, lemma) for the given uid pair
			for feature in self.relFeatures:
				for i in feature[(uid1, uid2)]:
					features.append(i)
			label = [1,0]
			if self.corpus.XUIDToMention[muid1].REF == self.corpus.XUIDToMention[muid2].REF:
				label = [0,1]
			pairs.append((muid1, muid2))
			X.append(features)
			Y.append(label)
		print("* createData() loaded",len(pairs), "pairs")
		return [pairs, X, Y]
