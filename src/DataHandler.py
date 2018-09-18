import time
import pickle
import numpy as np
import random
from sklearn.metrics import accuracy_score # TMP -- dependency parse features
from sklearn.neural_network import MLPClassifier # TMP -- dependency parse features
from sklearn.metrics import classification_report # TMP -- dependency parse features
from itertools import chain
from collections import defaultdict
class DataHandler:
	def __init__(self, helper, trainXUIDs, devXUIDs, testXUIDs):
		self.helper = helper
		self.args = helper.args
		self.corpus = helper.corpus

		self.trainXUIDs = trainXUIDs
		self.devXUIDs = devXUIDs
		self.testXUIDs = testXUIDs

		# we are passing in 3 sets of XUIDs, and these are the ones we
		# actually want to use for our model, so this is where we
		# should keep track of which docs -> XUIDs (only for the sake of
		# knowing if we have singletons -- docs with only 1 XUID, and thus would
		# have no pairs constructed for training/testing)
		self.docToXUIDsWeWantToUse = defaultdict(set)
		for xuid in chain(self.trainXUIDs, self.devXUIDs, self.testXUIDs):
			doc_id = self.corpus.XUIDToMention[xuid].doc_id
			self.docToXUIDsWeWantToUse[doc_id].add(xuid)

		self.badPOS = ["‘’", "``", "POS", "$", "''"] # TMP FOR SAMELEMMA TEST
		self.singleFeatures = []
		self.relFeatures = []

		if self.args.useECBTest:
			f_suffix = "ecb"
		else:
			f_suffix = "hddcrp"

		if self.args.wordFeature:
			lf = self.loadFeature("../data/features/" + str(f_suffix) + "/word.f")
			self.singleFeatures.append(lf.singles)
			self.relFeatures.append(lf.relational)
		if self.args.lemmaFeature:
			lf = self.loadFeature("../data/features/" + str(f_suffix) + "/lemma.f")
			self.singleFeatures.append(lf.singles)
			self.relFeatures.append(lf.relational)
		if self.args.charFeature:
			lf = self.loadFeature("../data/features/" + str(f_suffix) + "/char.f")
			self.singleFeatures.append(lf.singles)
			self.relFeatures.append(lf.relational)
		if self.args.posFeature:
			lf = self.loadFeature("../data/features/" + str(f_suffix) + "/pos.f")
			self.singleFeatures.append(lf.singles)
			self.relFeatures.append(lf.relational)
		if self.args.dependencyFeature:
			lf = self.loadFeature("../data/features/" + str(f_suffix) + "/dependency.f")
			self.singleFeatures.append(lf.singles)
			self.relFeatures.append(lf.relational)
		if self.args.bowFeature:
			lf = self.loadFeature("../data/features/" + str(f_suffix) + "/bow.f")
			self.singleFeatures.append(lf.singles)
			self.relFeatures.append(lf.relational)
		if self.args.wordnetFeature:
			lf = self.loadFeature("../data/features/" + str(f_suffix) + "/wordnet.f")
			self.relFeatures.append(lf.relational)

	def loadNNData(self, useRelationalFeatures, useCCNN, scope):
		print("[dh] loading ...")
		start_time = time.time()
		if useCCNN:
			(self.trainID, self.trainX, self.trainY) = self.createDataForCCNN(self.helper.trainingDirs, self.trainXUIDs, useRelationalFeatures, True, scope)
			(self.devID, self.devX, self.devY) = self.createDataForCCNN(self.helper.devDirs, self.devXUIDs, useRelationalFeatures, False, scope)

			# TMP dependency features -- train and test a NN over the dependency features
			print("len:", len(self.trainX), len(self.trainY))
			print("len:", len(self.devX), len(self.devY))
			'''
			alphas = [0.0001, 0.001]
			num_epochs = [10, 50]
			for a in alphas:
				for ne in num_epochs:
					#for _ in range(3):
					clf = MLPClassifier(alpha = a, max_iter=ne, verbose=False)
					clf.fit(self.trainX, self.trainY)
					preds = clf.predict(self.devX)
					print("ALPHA:", a, "ne:") #, ne, "run:", int(_))
					print(classification_report(self.devY, preds))
					print(accuracy_score(self.devY, preds))
					print("self.devY:", self.devY)
					print("preds:", preds)
			exit(1)
			'''
			(self.testID, self.testX, self.testY) = self.createDataForCCNN(self.helper.testingDirs, self.testXUIDs, useRelationalFeatures, False, scope)
		else: # FOR FFNN and SVM
			(self.trainID, self.trainX, self.trainY) = self.createDataForFFNN(self.helper.trainingDirs, self.trainXUIDs, useRelationalFeatures, True, scope)
			(self.devID, self.devX, self.devY) = self.createDataForFFNN(self.helper.devDirs, self.devXUIDs, useRelationalFeatures, False, scope)
			(self.testID, self.testX, self.testY) = self.createDataForFFNN(self.helper.testingDirs, self.testXUIDs, useRelationalFeatures, False, scope)
		print("[dh] done loading -- took ", str((time.time() - start_time)), "seconds")

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

		# we sort to ensure consistency across multiple runs
		print("# mentions passed-in:", len(XUIDs))
		#print("createXUIDPairs() created ECBDirToXUIDs to have this many [real] dirs:",len(ECBDirToXUIDs.keys()))
		tmp_xuids_reclaimed = set()
		tmp_ecbtoxuids = set()
		for ecb_dir in sorted(ECBDirToXUIDs.keys()):
		#for dirHalf in sorted(dirHalfToXUIDs.keys()):
			for xuid1 in sorted(ECBDirToXUIDs[ecb_dir]):
				tmp_ecbtoxuids.add(xuid1)
				for xuid2 in sorted(ECBDirToXUIDs[ecb_dir]):
					if xuid2 <= xuid1:
						continue
					inSameDoc = False
					inSameDirHalf = False
					if self.corpus.XUIDToMention[xuid1].doc_id == self.corpus.XUIDToMention[xuid2].doc_id:
						inSameDoc = True
					if self.corpus.XUIDToMention[xuid1].dirHalf == self.corpus.XUIDToMention[xuid2].dirHalf:
						inSameDirHalf = True
					if scope == "doc" and inSameDoc:
						xuidPairs.add((xuid1, xuid2))
						tmp_xuids_reclaimed.add(xuid1)
						tmp_xuids_reclaimed.add(xuid2)
					elif scope == "dirHalf" and not inSameDoc and inSameDirHalf:
						xuidPairs.add((xuid1, xuid2))
						tmp_xuids_reclaimed.add(xuid1)
						tmp_xuids_reclaimed.add(xuid2)
					elif scope == "dir" and not inSameDoc:
						xuidPairs.add((xuid1, xuid2))
		#print("tmp_xuids_reclaimed:", len(tmp_xuids_reclaimed))
		print("tmp_ecbtoxuids:", len(tmp_ecbtoxuids))
		return xuidPairs
	
	# almost identical to createData() but it re-shapes the vectors to be 5D -- pairwise.
	# i could probably combine this into 1 function and have a boolean flag isCCNN=True.
	# we pass in XUID because the mentions could be from any Stan, HDDCRP, or ECB; however,
	# we need to remember that the co-reference REF tags only exist in the output file that we compare against
	def createDataForCCNN(self, dirs, XUIDs, useRelationalFeatures, negSubsample, scope):

		# TMP for dependency features testing for adding entities
		X_dep = []
		Y_dep = []

		pairs = []
		X = []
		Y = []
		labels = []
		numFeatures = 0
		numPosAdded = 0
		numNegAdded = 0
		xuidPairs = self.createXUIDPairs(XUIDs, scope)
		print("* [createDataForCCNN] # XUIDs passed-in:", len(XUIDs), "; # pairs made from these: ", len(xuidPairs))

		# TMP ADDED FOR SAME LEMMA TEST
		'''
		TN = 0.0
		TP = 0.0
		FN = 0.0
		FP = 0.0
		'''

		# just for dependency sanity check
		'''
		for xuid in XUIDs:
			m = self.corpus.XUIDToMention[xuid]
			if m.dirHalf == "1ecb.xml":
				print(m)
				t1textp = [_.text for _ in m.parentTokens]
				e1REFsp = [_.REF for _ in m.parentEntities]
				t1textc = [_.text for _ in m.childrenTokens]
				e1REFsc = [_.REF for _ in m.childrenEntities]
				print("\tt1textp:", t1textp)
				print("\te1REFsp:", e1REFsp)
				print("\tt1textc:", t1textc)
				print("\te1REFsc:", e1REFsc)
		exit(1)
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

			# TMP ADDED TO CHECK DEPENDENCY FEATURES
			# PARENTS, CHILDREN, MIXED (do any tokens exist)
			# 1 do the mentions have a token in common?
			# 2 do the mentions have an entity mention in common?
			# 3 do the mentions have their 1st entity mention in common?
			m1 = self.corpus.XUIDToMention[xuid1]
			m2 = self.corpus.XUIDToMention[xuid2]
			features = []


			# prints the dependency stuff, to see if it makes sense
			#if m1.REF == m2.REF or random.random() < 0.2: # 1/5 of the negatives

			# 1
			tokenShared = False
			t1text = [_.text for _ in m1.parentTokens]
			t2text = [_.text for _ in m2.parentTokens]

			e1REFs = [_.REF for _ in m1.parentEntities]
			e2REFs = [_.REF for _ in m2.parentEntities]
			for t1 in t1text:
				if t1 in t2text:
					tokenShared = True
					break
						
			# 2
			entityInCommon = False
			for p1 in e1REFs:
				if p1 in e2REFs:
					entityInCommon = True
					break
			# 3
			firstEntitiesEqual = False
			if len(e1REFs) > 0 and len(e2REFs) > 0 and e1REFs[0] == e2REFs[0]:
				firstEntitiesEqual = True
			features.append(tokenShared)
			features.append(entityInCommon)
			features.append(firstEntitiesEqual)

			# children
			# 1
			tokenShared = False
			t1text = [_.text for _ in m1.childrenTokens]
			t2text = [_.text for _ in m2.childrenTokens]

			e1REFs = [_.REF for _ in m1.childrenEntities]
			e2REFs = [_.REF for _ in m2.childrenEntities]
			for t1 in t1text:
				if t1 in t2text:
					tokenShared = True
					break
			
			# 2
			entityInCommon = False
			for p1 in e1REFs:
				if p1 in e2REFs:
					entityInCommon = True
					break
			# 3
			firstEntitiesEqual = False
			if len(e1REFs) > 0 and len(e2REFs) > 0 and e1REFs[0] == e2REFs[0]:
				firstEntitiesEqual = True
			features.append(tokenShared)
			features.append(entityInCommon)
			features.append(firstEntitiesEqual)

			# MIXED
			# 1
			tokenShared = False
			#parents
			t1textp = [_.text for _ in m1.parentTokens]
			t2textp = [_.text for _ in m2.parentTokens]
			e1REFsp = [_.REF for _ in m1.parentEntities]
			e2REFsp = [_.REF for _ in m2.parentEntities]
			#children
			t1textc = [_.text for _ in m1.childrenTokens]
			t2textc = [_.text for _ in m2.childrenTokens]
			e1REFsc = [_.REF for _ in m1.childrenEntities]
			e2REFsc = [_.REF for _ in m2.childrenEntities]
			for t1 in [t1textp, t1textc]:
				if t1 in t2textp or t1 in t2textc:
					tokenShared = True
					break
			# 2
			entityInCommon = False
			for p1 in [e1REFsp, e1REFsc]:
				if p1 in e2REFsp or p1 in e2REFsc:
					entityInCommon = True
					break
			# 3
			firstEntitiesEqual = False
			if features[2] or features[5]:
				firstEntitiesEqual = True
			if not firstEntitiesEqual:
				if len(e1REFsc) > 0 and len(e2REFsp) > 0:
					if e1REFsc[0] == e2REFsp[0]:
						firstEntitiesEqual = True
				if len(e1REFsp) > 0 and len(e2REFsc) > 0:
					if e1REFsp[0] == e2REFsc[0]:
						firstEntitiesEqual = True

			features.append(tokenShared)
			features.append(entityInCommon)
			features.append(firstEntitiesEqual)

			if len(t1textp) > 0:
				features.append(1)
			else:
				features.append(0)
			if len(t1textc) > 0:
				features.append(1)
			else:
				features.append(0)

			if len(t2textp) > 0:
				features.append(1)
			else:
				features.append(0)
			if len(t2textc) > 0:
				features.append(1)
			else:
				features.append(0)

			#features.append(m1.REF == m2.REF)
			#if m1.dirHalf == "1ecb.xml":
			if m1.REF == m2.REF or random.random() < 0.2: # 1/5 of the negatives:
			#print("p1:", m1.parentTokens, "p2:", m2.parentTokens)
				X_dep.append([int(_) for _ in features])
				Y_dep.append(int(m1.REF == m2.REF))
					#print(str(int(m1.REF == m2.REF)), t1text, t2text, features)

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
				#if m1.dirHalf == "1ecb.xml":
				#	print(features)
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
				#if m1.dirHalf == "1ecb.xml":
				#	print(features)
			m1_features = []
			m2_features = []
			(uid1, uid2) = sorted([self.corpus.XUIDToMention[xuid1].UID, self.corpus.XUIDToMention[xuid2].UID])
			# loops through each feature (e.g., BoW, lemma) for the given uid pair
			
			for feature in self.singleFeatures:
				#print("feature keys:", feature.keys())
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
			print("* ERROR: no pairs!")
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
		#print("shape:", X.shape, "len:", len(X), len(X[0]), len(X[0][0]), len(X[0][0][0]), len(X[0][0][0][0]))
		#print("shapeY:", Y.shape) #, "len:", len(Y), len(Y[0]), len(Y[0][0]), len(Y[0][0][0]))
		#exit(1)
		#return (pairs, X_dep, Y_dep)
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
