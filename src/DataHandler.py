import time
import pickle
import numpy as np
import random
from MiniPred import MiniPred
from itertools import chain
from collections import defaultdict
class DataHandler:
	def __init__(self, helper, trainXUIDs, devXUIDs, testXUIDs):

		# keeps track of how many event pairs coref and their entity ones too
		self.tmp_coref_counts = defaultdict(lambda: defaultdict(int)) 
		self.tmp_minipreds = {}

		self.tmp_count_in = 0
		self.tmp_count_out = 0

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

	def loadFeature(self, file):
		print("loading",file)
		return pickle.load(open(file, 'rb'))

	def load_xuid_pairs(self, supp_features, scope):
		self.trainXUIDPairs = self.createXUIDPairs(self.trainXUIDs, scope, supp_features)
		self.devXUIDPairs = self.createXUIDPairs(self.devXUIDs, scope, supp_features)
		self.testXUIDPairs = self.createXUIDPairs(self.testXUIDs, scope, supp_features)

	def construct_tree_files_(self):
		self.construct_tree_files(self.trainXUIDPairs, "tree_lstm/data/sick/train/ecb/")
		self.construct_tree_files(self.devXUIDPairs, "tree_lstm/data/sick/dev/ecb/")
		self.construct_tree_files(self.testXUIDPairs, "tree_lstm/data/sick/test/ecb/")

	# produces files that TreeLSTM can read
	def construct_tree_files(self, xuid_pairs, dir_path):
	
		fout_a = open(dir_path + "a.toks", 'w')
		fout_b = open(dir_path + "b.toks", 'w')
		fout_sim = open(dir_path + "sim.txt", 'w')
		fout_a_deps = open(dir_path + "a.parents", 'w')
		fout_b_deps = open(dir_path + "b.parents", 'w')

		# gets unique IDs
		valid_xuids = set()
		for xuid1, xuid2 in xuid_pairs:
			valid_xuids.add(xuid1)
			valid_xuids.add(xuid2)
		print("# unique valid_xuids:", len(valid_xuids))
		
		# constructs ecb tokens -> mentions
		doc_ids = set()
		sentenceTokenToMention = defaultdict(lambda: defaultdict(set))
		for xuid in valid_xuids:
			m = self.corpus.EUIDToMention[xuid]
			sentNum = m.globalSentenceNum
			doc_ids.add(m.doc_id)
			for t in m.tokens:
				sentenceTokenToMention[sentNum][t].add(m)

		# creates stan -> ecb tokens
		stanTokenToECBTokens = defaultdict(set)
		for doc_id in doc_ids:
			for t in self.corpus.doc_idToDocs[doc_id].tokens:
				for stan in t.stanTokens:
					stanTokenToECBTokens[stan].add(t)

		# writes out each sentence in plain text
		for xuid1, xuid2 in xuid_pairs:
			m1 = self.corpus.XUIDToMention[xuid1]
			m2 = self.corpus.XUIDToMention[xuid2]
			label = "0"
			if m1.REF == m2.REF:
				label = "1"
			text1, dependency_chains1 = self.construct_tree_file(xuid1, stanTokenToECBTokens)
			text2, dependency_chains2 = self.construct_tree_file(xuid2, stanTokenToECBTokens)
			
			fout_a.write(text1 + "\n")
			fout_b.write(text2 + "\n")
			fout_sim.write(label + "\n")
			fout_a_deps.write(dependency_chains1 + "\n")
			fout_b_deps.write(dependency_chains2 + "\n")
		fout_a.close()
		fout_b.close()
		fout_sim.close()
		fout_a_deps.close()
		fout_b_deps.close()

	def construct_tree_file(self, xuid, stanTokenToECBTokens):
		m = self.corpus.XUIDToMention[xuid]
		#sent = " ".join([t.text for t in self.corpus.globalSentenceNumToTokens[m.globalSentenceNum]])

		sent = ""
		# constructs token -> index (1-based)
		token_to_index = {}
		token_index = 1
		for token in self.corpus.globalSentenceNumToTokens[m.globalSentenceNum]:
			if token.tokenID == "-1":
				continue
			token_to_index[token] = token_index
			sent += token.text + " "
			token_index += 1
		sent = sent.strip()

		# constructs dependency chain
		dependency_chain = []
		for t in self.corpus.globalSentenceNumToTokens[m.globalSentenceNum]:
			if t.tokenID == "-1":
				continue
			bestStan = self.getBestStanToken(t.stanTokens)
			if len(bestStan.parentLinks[self.helper.dependency_parse_type]) > 1:
				print("* more than 1 parent:", len(bestStan.parentLinks[self.helper.dependency_parse_type]))
				exit()

			pl = next(iter(bestStan.parentLinks[self.helper.dependency_parse_type]))
			parentToken = pl.parent
			#print("pl:", pl)
			if parentToken.isRoot:
				dependency_chain.append(0)
			else:
				ecb_parent_tokens = stanTokenToECBTokens[parentToken]
				if len(ecb_parent_tokens) != 1:
					print("* not 1 ecb parents", ecb_parent_tokens)
					exit()
				token = next(iter(ecb_parent_tokens))
				dependency_chain.append(token_to_index[token])
		if dependency_chain.count(0) != 1:
			print("* have != 1 dependency parent")
			exit()
		if len(sent.split(" ")) != len(dependency_chain):
			print("* sent len != dependency length")
			exit()

		return sent, " ".join([str(d) for d in dependency_chain])

	def loadNNData(self, supp_features, useCCNN, scope):
		print("[dh] loading ...")
		start_time = time.time()

		if useCCNN:
			(self.trainID, self.trainX, self.supplementalTrain, self.trainY) = self.createDataForCCNN(self.helper.trainingDirs, self.trainXUIDPairs, supp_features, True, scope, "train")
			(self.devID, self.devX, self.supplementalDev, self.devY) = self.createDataForCCNN(self.helper.devDirs, self.devXUIDPairs, supp_features, False, scope, "dev")
			(self.testID, self.testX, self.supplementalTest, self.testY) = self.createDataForCCNN(self.helper.testingDirs, self.testXUIDPairs, supp_features, False, scope, "test")
		else: # FOR FFNN and SVM
			(self.trainID, self.trainX, self.trainY) = self.createDataForFFNN(self.helper.trainingDirs, self.trainXUIDs, supp_features, True, scope)
			(self.devID, self.devX, self.devY) = self.createDataForFFNN(self.helper.devDirs, self.devXUIDs, supp_features, False, scope)
			(self.testID, self.testX, self.testY) = self.createDataForFFNN(self.helper.testingDirs, self.testXUIDs, supp_features, False, scope)
		print("[dh] done loading -- took ", str((time.time() - start_time)), "seconds")

	# return a list of XUID pairs, based on what was passed-in,
	# where each pair comes from either:
	# (1) 'doc' = the same doc (within-dic); or,
	# (2) 'dirHalf' = the same dirHalf (but not same doc)
	# (3) 'dir' = the same dir (but not same doc)
	def createXUIDPairs(self, XUIDs, scope, supp_features_type):
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
		tmp_xuids_reclaimed = set()
		tmp_ecbtoxuids = set()
		for ecb_dir in sorted(ECBDirToXUIDs.keys()):
		#for dirHalf in sorted(dirHalfToXUIDs.keys()):
			for xuid1 in sorted(ECBDirToXUIDs[ecb_dir]):
				tmp_ecbtoxuids.add(xuid1)
				m1 = self.corpus.XUIDToMention[xuid1]

				for xuid2 in sorted(ECBDirToXUIDs[ecb_dir]):
					if xuid2 <= xuid1:
						continue
					m2 = self.corpus.XUIDToMention[xuid2]

					if m1.isPred and m2.isPred:  # both are events, so let's use paths to entities
						m1_full_paths = m1.levelToChildren
						m2_full_paths = m2.levelToChildren
					elif not m1.isPred and not m2.isPred:  # both are entities, so let's use paths to events
						m1_full_paths = m1.levelToParents
						m2_full_paths = m2.levelToParents

					# assume its good.  only if we care about links do we
					# optionally set it to false
					have_valid_links = True
					if supp_features_type == "shortest":
						if len(m1_full_paths) == 0 or len(m2_full_paths) == 0:
							have_valid_links = False

					inSameDoc = False
					inSameDirHalf = False
					if self.corpus.XUIDToMention[xuid1].doc_id == self.corpus.XUIDToMention[xuid2].doc_id:
						inSameDoc = True
					if self.corpus.XUIDToMention[xuid1].dirHalf == self.corpus.XUIDToMention[xuid2].dirHalf:
						inSameDirHalf = True

					to_add = False
					if scope == "doc" and inSameDoc and have_valid_links:
						to_add = True
					elif scope == "dirHalf" and not inSameDoc and inSameDirHalf and have_valid_links:
						to_add = True
					elif scope == "dir" and not inSameDoc and have_valid_links:
						xuidPairs.add((xuid1, xuid2))

					# we passed all of our filters (appropriate scope and is 'shortest' or not)
					if to_add:
						xuidPairs.add((xuid1, xuid2))
						tmp_xuids_reclaimed.add(xuid1)
						tmp_xuids_reclaimed.add(xuid2)
		#print("tmp_xuids_reclaimed:", len(tmp_xuids_reclaimed))
		#print("tmp_ecbtoxuids:", len(tmp_ecbtoxuids))
		print("\t# xuidPairs:", len(xuidPairs))
		rooted_xuids = self.construct_rooted_trees(xuidPairs)

		# TODO: need to filter the pairs now
		filtered_xuid_pairs = set()
		for xuid1, xuid2 in xuidPairs:
			if xuid1 in rooted_xuids and xuid2 in rooted_xuids:
				filtered_xuid_pairs.add((xuid1, xuid2))
		print("#filtered_xuid_pairs:", len(filtered_xuid_pairs))

		#TODO: print the sentences for these pairs.  can measure treelstm performance and CCNN perf
		return filtered_xuid_pairs
		#return xuidPairs
		
	def construct_rooted_trees(self, xuid_pairs):
		rooted_xuids = set() # returns this
		valid_sentences = set()
		valid_xuids = set()
		sentenceTokenToMention = defaultdict(lambda: defaultdict(set))
		stanTokenToECBTokens = defaultdict(set)

		for xuid1, xuid2 in xuid_pairs:
			valid_xuids.add(xuid1)
			valid_xuids.add(xuid2)

		print("# unique valid_xuids:", len(valid_xuids))
		for xuid in valid_xuids:
			m = self.corpus.EUIDToMention[xuid]
			sentNum = m.globalSentenceNum

			# constructs ecb tokens -> mentions
			for t in m.tokens:
				sentenceTokenToMention[sentNum][t].add(m)
				# construct stan -> ecb tokens.

				#for stan in t.stanTokens:
				#	stanTokenToECBTokens[stan].add(t)

			for t in self.corpus.doc_idToDocs[m.doc_id].tokens:
				for stan in t.stanTokens:
					stanTokenToECBTokens[stan].add(t)

			valid_sentences.add(sentNum)
		print("# sentences:", len(valid_sentences))

		has_mention = 0
		has_no_mention = 0
		for sent_num in valid_sentences:
			#print("sent_num:", sent_num)
			root_stan = None
			sent_text = ""
			for t in self.corpus.globalSentenceNumToTokens[sent_num]:
				if t.tokenID == "-1":
					continue
				bestStan = self.getBestStanToken(t.stanTokens)
				for pl in bestStan.parentLinks[self.helper.dependency_parse_type]:
					parentToken = pl.parent
					if parentToken.isRoot:
						root_stan = parentToken
				sent_text += t.text + " "
			#print("\tsent:", sent_text, "; root:", root_stan)
			actual_root = root_stan.childLinks[self.helper.dependency_parse_type][0].child

			if actual_root in stanTokenToECBTokens:
				ecb_tokens = stanTokenToECBTokens[actual_root]
				rooted_mentions = set()
				for ecb in ecb_tokens:
					if ecb in sentenceTokenToMention[sent_num]:
						rooted_mentions = sentenceTokenToMention[sent_num][ecb]
				#print("ment:", m)
				if len(rooted_mentions) == 0:
					has_no_mention += 1
					#print("* missing a mention. here's the dfs:")
					#helper.dfs_tree(root_stan, [], [], "", sentenceTokenToMention[sent_num], stanTokenToECBTokens)
				else:
					#print("rooted_mentions:", rooted_mentions)
					if len(rooted_mentions) > 1:
						print("whoa:", len(rooted_mentions))
						exit()
					has_mention += 1
					m = next(iter(rooted_mentions))
					rooted_xuids.add(m.XUID)
			else:
				print("** ERROR: don't have an ecb token for the stantoken")
				exit()
		print("has_mention:", has_mention, "; has_no_mention:", has_no_mention)
		return rooted_xuids

	# almost identical to createData() but it re-shapes the vectors to be 5D -- pairwise.
	# i could probably combine this into 1 function and have a boolean flag isCCNN=True.
	# we pass in XUID because the mentions could be from any Stan, HDDCRP, or ECB; however,
	# we need to remember that the co-reference REF tags only exist in the output file that we compare against
	def createDataForCCNN(self, dirs, xuidPairs, supp_features_type, negSubsample, scope, split):

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

		numPosNotAdded = 0
		numNegNotAdded = 0

		#xuidPairs = self.createXUIDPairs(XUIDs, scope)
		print("*B ",split,"[createDataForCCNN] # pairs made from these: ", len(xuidPairs))

		tmp_pairs_with = 0
		tmp_pairs_without = 0
		xuids_used_for_pairs = set()
		tmp_dir_to_with = defaultdict(int)
		for (xuid1, xuid2) in xuidPairs:
			if xuid1 == xuid2:
				print("whaaaaa: xuidPairs:", xuidPairs)
				exit(1)

			m1 = self.corpus.XUIDToMention[xuid1]
			m2 = self.corpus.XUIDToMention[xuid2]
			if m1.isPred and not m2.isPred:
				println("* ERROR: mismatched mention types (event and entities)")
				exit(1)
				continue
			elif not m1.isPred and m2.isPred:
				println("* ERROR: mismatched mention types (event and entities)")
				exit(1)
				continue
			
			features = []

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
				
				# actually means paths to the other mention-type, thanks to the code above
				if len(m1_full_paths) == 0 or len(m2_full_paths) == 0:
					if self.corpus.XUIDToMention[xuid1].REF == self.corpus.XUIDToMention[xuid2].REF:
						numPosNotAdded += 1
					else:
						numNegNotAdded += 1

					tmp_pairs_without += 1
					continue
				else: # both events have a path to entities
					tmp_pairs_with += 1

					m1_shortests = [] if len(m1_full_paths) == 0 else [x for x in m1_full_paths[next(iter(sorted(m1_full_paths)))]]
					m2_shortests = [] if len(m2_full_paths) == 0 else [x for x in m2_full_paths[next(iter(sorted(m2_full_paths)))]]

					entcoref = False
					same_paths = False

					# gold info
					for (ment1, path1) in m1_shortests:
						for (ment2, path2) in m2_shortests:
							if ment1.REF == ment2.REF:
								entcoref = True

					if entcoref:
						features.append(0)
					else:
						features.append(1)

					'''
					#THIS WAS UNCOMMENTED
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
					'''

					# used for checking the types of errors we make
					'''
					tmp_key = (xuid1, xuid2)
					if xuid2 < xuid1:
						tmp_key = (xuid2, xuid1)
					# COUNTS STATISTICS OF EVENT AND ENTITY COREF'ing
					if self.corpus.XUIDToMention[xuid1].REF == self.corpus.XUIDToMention[xuid2].REF:
						self.tmp_coref_counts["event_coref"][entcoref] += 1
					else:
						self.tmp_coref_counts["event_NOcoref"][entcoref] += 1

					# TMP COUNTS COREF STATS
					event_gold = False
					if self.corpus.XUIDToMention[xuid1].REF == self.corpus.XUIDToMention[xuid2].REF:
						event_gold = True

					mp = MiniPred(tmp_key, event_gold, entcoref)
					self.tmp_minipreds[tmp_key] = mp
					'''

				# OPTIONAL GOLD INFO
				'''
				if m1.REF == m2.REF:
					features.append(0)
					#features.append(0)
					#features.append(0)
				else:
					features.append(1)
					#features.append(1)
				'''

			# NOTE: if this is for HDDCRP or Stan mentions, the REFs will always be True
			# because we don't have such info for them, so they are ""
			if self.corpus.XUIDToMention[xuid1].REF == self.corpus.XUIDToMention[xuid2].REF:
				labels.append(0)
				numPosAdded += 1
			else:
				if negSubsample and numNegAdded > numPosAdded*self.args.numNegPerPos:
					continue
				numNegAdded += 1
				labels.append(1)

			# TMP: merely displays statistics just like checkDependencyRelations()
			# namely, how many events coref, and of these 			
			m1_features = []
			m2_features = []

			#if supp_features_type != "none":
			supp_features.append(np.asarray(features))

			(uid1, uid2) = sorted([self.corpus.XUIDToMention[xuid1].UID, self.corpus.XUIDToMention[xuid2].UID])
			
			# loops through each feature (e.g., BoW, lemma) for the given uid pair
			for feature in self.singleFeatures:
				for i in feature[uid1]:  # loops through each val of the given feature
					m1_features.append(i)
				for i in feature[uid2]:
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

			# makes xuid pairs
			pairs.append((xuid1, xuid2))

			# keeps track of all unique XUIDs that were used for constructing pairs
			xuids_used_for_pairs.add(xuid1)
			xuids_used_for_pairs.add(xuid2)
			mentionType = str(m1.isPred) + "_" + str(m2.isPred)
			mentionTypeToCount[mentionType] += 1

		print("\t", split, "E # unique XUIDs used in pairs:", len(xuids_used_for_pairs))
		print("\t", split, "we dont have these:", tmp_dir_to_with)
		
		X = np.asarray(X)
		
		# TODO: uncomment out this line!
		supp_features = np.asarray(supp_features)

		Y = np.asarray(labels)
		print("\t", split, "numPosAdded:", str(numPosAdded))
		print("\t", split, "numNegAdded:", str(numNegAdded))
		pp = float(numPosAdded / (numPosAdded+numNegAdded))
		pn = float(numNegAdded / (numPosAdded+numNegAdded))

		npp = 0
		npn = 0
		if numPosNotAdded > 0 or numNegNotAdded > 0:
			npp = float(numPosNotAdded / (numPosNotAdded + numNegNotAdded))
			npn = float(numNegNotAdded / (numPosNotAdded + numNegNotAdded))

		print("\t", split, "* createData() loaded", len(pairs), "pairs (# pos:",numPosAdded, \
			pp, "% pos; neg:", numNegAdded, "(%", pn, "); didn't add: # pos:", numPosNotAdded, \
			"(%", npp, "); didn't add: # neg:", numNegNotAdded, " (%", npn,"); features' length = ", \
			numFeatures, "; supp length:", str(len(supp_features)), str(len(supp_features[0])))

		if len(pairs) == 0:
			print("* ERROR: no pairs!")
			exit(1)
		
		print("ALL CORPUS tmp_coref_counts:", self.tmp_coref_counts)
		print("shortest stats: tmp xuids (# pairs w/ 2 relations):", len(self.tmp_minipreds.keys()))
		print("\t", split, "mentionTypeToCount:",str(mentionTypeToCount))
		print("\t", split, "C tmp_pairs_with paths:", tmp_pairs_with, "D tmp_pairs_without paths", tmp_pairs_without)
		print("**** tmp_count_in:", self.tmp_count_in)
		print("tmp_count_out:", self.tmp_count_out)
		return (pairs, X, supp_features, Y)

	#def createMiniFFNN(self, )

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
			print("* ERROR: our bestStanToken is empty! stanTokens", stanTokens)
			exit(1)
		return bestStanToken
