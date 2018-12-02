import time
import pickle
import numpy as np
import random

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

	def loadNNData(self, supp_features, useCCNN, scope):
		print("[dh] loading ...")
		start_time = time.time()
		if useCCNN:
			(self.trainID, self.trainX, self.supplementalTrain, self.trainY) = self.createDataForCCNN(self.helper.trainingDirs, self.trainXUIDs, supp_features, True, scope)
			(self.devID, self.devX, self.supplementalDev, self.devY) = self.createDataForCCNN(self.helper.devDirs, self.devXUIDs, supp_features, False, scope)			
			(self.testID, self.testX, self.supplementalTest, self.testY) = self.createDataForCCNN(self.helper.testingDirs, self.testXUIDs, supp_features, False, scope)
		else: # FOR FFNN and SVM
			(self.trainID, self.trainX, self.trainY) = self.createDataForFFNN(self.helper.trainingDirs, self.trainXUIDs, supp_features, True, scope)
			(self.devID, self.devX, self.devY) = self.createDataForFFNN(self.helper.devDirs, self.devXUIDs, supp_features, False, scope)
			(self.testID, self.testX, self.testY) = self.createDataForFFNN(self.helper.testingDirs, self.testXUIDs, supp_features, False, scope)
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
	def createDataForCCNN(self, dirs, XUIDs, supp_features_type, negSubsample, scope):

		# TMP, to ensure we correctly make pairs of entities or events but not entities-event pairs
		mentionTypeToCount = defaultdict(int)

		# TMP for dependency features testing for adding entities
		X_dep = []
		Y_dep = []

		pairs = []
		X = []
		Y = []
		supp_features = []
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
		tmp_pairs_with = 0
		tmp_pairs_without = 0

		tmp_dir_to_with = defaultdict(int)
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
			m1 = self.corpus.XUIDToMention[xuid1]
			m2 = self.corpus.XUIDToMention[xuid2]
			if m1.isPred and not m2.isPred:
				continue
			elif not m1.isPred and m2.isPred:
				continue
			
			features = []

			# TMP: if it's ent or events
			if supp_features_type == "type":
				if m1.isPred and m2.isPred: # events 
					features.append(0)
					features.append(1)
				elif not m1.isPred and not m2.isPred: # entities
					features.append(1)
					features.append(0)
			
			'''
			if supp_features_type == "none":
				if len(m1.levelToChildrenEntities) == 0 or len(m2.levelToChildrenEntities) == 0:
					tmp_pairs_without += 1
					continue
				else:
					tmp_pairs_with += 1
			'''
			# TMP added: tests adding entity coref info			
			if supp_features_type == "one" or supp_features_type == "shortest":
				
				# checks if one of the events doesn't have a path to an entity
				m1_full_paths = None
				m2_full_paths = None
				if m1.isPred and m2.isPred: # both are events, so let's use paths to entities
					m1_full_paths = m1.levelToChildren
					m2_full_paths = m2.levelToChildren
				elif not m1.isPred and not m2.isPred: # both are entities, so let's use paths to events
					m1_full_paths = m1.levelToParents
					m2_full_paths = m2.levelToParents
				
				if len(m1_full_paths) == 0 or len(m2_full_paths) == 0:
				#if len(m1.levelToChildrenEntities) == 0 or len(m2.levelToChildrenEntities) == 0:
					
					# at least one mention doesn't have a path to an Entity
					#features.append(0)
					#features.append(1)

					# golden entity info (at shortest level)
					#features.append(0)
					#features.append(0)
					#features.append(1)

					# do not have identical path 
					'''
					features.append(0)
					features.append(0)
					features.append(1)
					'''
					tmp_pairs_without += 1
					continue
				else: # both events have a path to entities
					#features.append(1)
					#features.append(0)
					tmp_pairs_with += 1
					'''
					if supp_features_type == "one":
						if 1 not in m1.levelToChildrenEntities or 1 not in m2.levelToChildrenEntities:
							continue
					'''
					m1_shortests = [] if len(m1_full_paths) == 0 else [x for x in m1_full_paths[next(iter(sorted(m1_full_paths)))]]
					m2_shortests = [] if len(m2_full_paths) == 0 else [x for x in m2_full_paths[next(iter(sorted(m2_full_paths)))]]

					#m1_shortests = set([] if len(m1.levelToChildrenEntities) == 0 else [x for x in m1.levelToChildrenEntities[next(iter(sorted(m1.levelToChildrenEntities)))]])
					#m2_shortests = set([] if len(m2.levelToChildrenEntities) == 0 else [x for x in m2.levelToChildrenEntities[next(iter(sorted(m2.levelToChildrenEntities)))]])

					entcoref = False
					max_coref_score = 0
					same_paths = False
					for (ment1, path1) in m1_shortests:
						for (ment2, path2) in m2_shortests:
							cur_score = 0
							if (ment1.XUID, ment2.XUID) in self.helper.predictions:
								cur_score = self.helper.predictions[(ment1.XUID, ment2.XUID)]
								#print("\tgot it:", cur_score)
							elif (ment2.XUID, ment1.XUID) in self.helper.predictions:
								cur_score = self.helper.predictions[(ment2.XUID, ment1.XUID)]
								#print("\tgot it:", cur_score)
							else:
								#print("dont have mentions:", ment1, ment2)
								tmp_dir_to_with[ment1.dir_num] += 1
								if ment1.dir_num in self.helper.testingDirs:
									#print("MISSING A PAIR THAT WE SHOULD TESTING PREDS FOR", ment1, ment2)
									#exit(1)
									cur_score = 0
								#print("we don't have it! but we have:", self.helper.predictions)
								#exit(1)
							cur_score = 1 - min(cur_score, 1)
							if cur_score > max_coref_score:
								max_coref_score = cur_score
								
								cur_paths_same = True
								for p1 in path1:
									for p2 in path2:
										if p1.relationship != p2.relationship:
											cur_paths_same = False
											break
								same_paths = cur_paths_same # resets it

							if ment1.REF == ment2.REF:
								entcoref = True
								#break
					
					#if m1.dir_num not in self.helper.testingDirs:
					#	print("TRAIN: pred:", str(max_coref_score), " gold:", str(int(entcoref)))

					# PREDICTED ENTITY INFO (at shortest level)
					if m1.dir_num in self.helper.testingDirs:
						if max_coref_score > self.args.entity_threshold:
							max_coref_score = 1
						else:
							max_coref_score = 0
						#features.append(0.5)
						#features.append(0.5)
						features.append(max_coref_score)
						features.append(1 - max_coref_score)
						#print("TEST: pred:", str(max_coref_score), " gold:", str(int(entcoref)))
					else: # GOLDEN ENTITY INFO (at shortest level)
						if entcoref:
							features.append(1)
							features.append(0)
						else:
							features.append(0)
							features.append(1)
					
					# checks paths
					dep1_relations = set()
					dep2_relations = set()

					# looks at path info
					m1_paths = [p for level in m1.levelToEntityPath.keys() for p in m1.levelToEntityPath[level]]
					m2_paths = [p for level in m2.levelToEntityPath.keys() for p in m2.levelToEntityPath[level]]
					
					m1_paths = []
					if 1 in m1.levelToEntityPath.keys():
						for p in m1.levelToEntityPath[1]:
							m1_paths.append(p)
							dep1_relations.add(p[0])

					m2_paths = []
					if 1 in m2.levelToEntityPath.keys():
						for p in m2.levelToEntityPath[1]:
							m2_paths.append(p)
							dep2_relations.add(p[0])
					
					''' # adds dependency path relation info
					for rel in sorted(self.helper.relationToIndex):
						if rel in dep1_relations and entcoref:
							features.append(1)
						else:
							features.append(0)
					for rel in sorted(self.helper.relationToIndex):
						if rel in dep2_relations and entcoref:
							features.append(1)
						else:
							features.append(0)
					'''
					#print("m1_paths:", str(m1_paths))
					
					haveIdenticalPath = False
					for m1p in m1_paths:
						for m2p in m2_paths:
							if m1p == m2p:
							#if m1p[0] == m2p[0]:
								haveIdenticalPath = True
								break
					
					if same_paths: #haveIdenticalPath:
						features.append(1)
						features.append(0)
						#features.append(0)
					else:
						features.append(0)
						features.append(1)
						#features.append(0)
					
				# OPTIONAL GOLD INFO
				'''
				if m1.REF == m2.REF:
					features.append(1)
					features.append(0)
				else:
					features.append(0)
					features.append(1)
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

			#features = [] # TODO: do not keep this this way; it represents NO SUPP INFO

			if supp_features_type != "none":
				supp_features.append(np.asarray(features))

			(uid1, uid2) = sorted([self.corpus.XUIDToMention[xuid1].UID, self.corpus.XUIDToMention[xuid2].UID])
			
			# loops through each feature (e.g., BoW, lemma) for the given uid pair
			for feature in self.singleFeatures:
				for i in feature[uid1]:  # loops through each val of the given feature
					m1_features.append(i)
				for i in feature[uid2]:
					m2_features.append(i)

			# loops through each feature (e.g., BoW, lemma) for the given uid pair
			'''
			if useRelationalFeatures: # this never seems to help
				for feature in self.relFeatures:
					if (uid1, uid2) not in feature:
						print("not in")
						exit(1)
					for i in feature[(uid1, uid2)]:
						m1_features.append(i)
						m2_features.append(i)
			'''
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

			# make the joint dependency embedding
			X_dep.append([int(_) for _ in features])
			'''
			dep1Matrix = np.zeros(shape=(1, len(features)))
			dep2Matrix = np.zeros(shape=(1, len(features)))
			dep1Matrix[0] = features
			dep2Matrix[0] = features
			dep1Matrix = np.asarray(dep1Matrix).reshape(1, len(features), 1)
			dep2Matrix = np.asarray(dep2Matrix).reshape(1, len(features), 1)
			'''
			X.append(pair)

			# makes xuid pairs
			pairs.append((xuid1, xuid2))
			
			mentionType = str(m1.isPred) + "_" + str(m2.isPred)
			mentionTypeToCount[mentionType] += 1

		print("we dont have these:", tmp_dir_to_with)
		
		X = np.asarray(X)
		supp_features = np.asarray(supp_features)
		#print("labels:",labels)
		Y = np.asarray(labels)
		print("numPosAdded:", str(numPosAdded))
		print("numNegAdded:", str(numNegAdded))
		pp = float(numPosAdded / (numPosAdded+numNegAdded))
		pn = float(numNegAdded / (numPosAdded+numNegAdded))
		print("* createData() loaded", len(pairs), "pairs (", \
			pp, "% pos, ", pn, "% neg); features' length = ", \
			numFeatures, "; supp length:", str(len(supp_features)))

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
		print("mentionTypeToCount:",str(mentionTypeToCount))
		print("tmp_pairs_with:", tmp_pairs_with, "tmp_pairs_without", tmp_pairs_without)
		return (pairs, X, supp_features, Y)

	# creates data for FFNN and SVM:
	# [(xuid1,xuid2), [features], [1,0]]
	def createDataForFFNN(self, dirs, XUIDs, supp_features, negSubsample, scope):
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
			'''
			if useRelationalFeatures: # this never seems to help
				for feature in self.relFeatures:
					if (uid1, uid2) not in feature:
						print("not in")
						exit(1)
					for i in feature[(uid1, uid2)]:
						features.append(i)
			'''
			if len(features) != numFeatures and numFeatures != 0:
				print("* ERROR: # features diff:",len(features),"and",numFeatures)
			numFeatures = len(features)

			pairs.append((xuid1, xuid2))
			X.append(features)
			Y.append(label)
		print("features have a length of:", numFeatures)
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
