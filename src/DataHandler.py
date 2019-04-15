import time
import pickle
import numpy as np
import random
from MiniPred import MiniPred
from itertools import chain
from collections import defaultdict

class SentTree:
	def __init__(self, sent, dependency_chain, mention_token_indices, root_XUID):
		self.sent = sent
		self.dependency_chain = " ".join([str(d) for d in dependency_chain])
		self.mention_token_indices = mention_token_indices
		self.root_XUID = root_XUID

	def __str__(self):
		return("sent:" + self.sent + "; dependency_chain:" + str(self.dependency_chain) + \
		"mention_token_indices:" + str(self.mention_token_indices) + "ROOT:" + str(self.root_XUID))

class TreeSet:
	def __init__(self, sent_legend, sent_labels, sent_key_to_index, xuid_pair_and_sent_key):
		self.sent_legend = sent_legend
		self.sent_labels = sent_labels
		self.sent_key_to_index = sent_key_to_index
		self.xuid_pair_and_sent_key = xuid_pair_and_sent_key

	def __str__(self):
		print("# sent pairs:", len(self.sent_labels), "; # pos:", str(sent_labels.count(2)), "; # neg:", str(sent_labels.count(1)))
		
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
		self.allXUIDs = trainXUIDs | devXUIDs | testXUIDs

		# to be filled in by the Construct Trees functions
		self.train_tree_set = None
		self.dev_tree_set = None
		self.test_tree_set = None
		self.sent_num_to_obj = {}
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
			lf = self.loadFeature(self.args.baseDir + "data/features/" + str(f_suffix) + "/word.f")
			self.singleFeatures.append(lf.singles)
			self.relFeatures.append(lf.relational)
		if self.args.lemmaFeature:
			lf = self.loadFeature(self.args.baseDir + "data/features/" + str(f_suffix) + "/lemma.f")
			self.singleFeatures.append(lf.singles)
			self.relFeatures.append(lf.relational)
		if self.args.charFeature:
			lf = self.loadFeature(self.args.baseDir + "data/features/" + str(f_suffix) + "/char.f")
			self.singleFeatures.append(lf.singles)
			self.relFeatures.append(lf.relational)
		if self.args.posFeature:
			lf = self.loadFeature(self.args.baseDir + "data/features/" + str(f_suffix) + "/pos.f")
			self.singleFeatures.append(lf.singles)
			self.relFeatures.append(lf.relational)
		if self.args.dependencyFeature:
			lf = self.loadFeature(self.args.baseDir + "data/features/" + str(f_suffix) + "/dependency.f")
			self.singleFeatures.append(lf.singles)
			self.relFeatures.append(lf.relational)
		if self.args.bowFeature:
			lf = self.loadFeature(self.args.baseDir + "data/features/" + str(f_suffix) + "/bow.f")
			self.singleFeatures.append(lf.singles)
			self.relFeatures.append(lf.relational)
		if self.args.wordnetFeature:
			lf = self.loadFeature(self.args.baseDir + "data/features/" + str(f_suffix) + "/wordnet.f")
			self.relFeatures.append(lf.relational)

	def loadFeature(self, file):
		print("loading",file)
		return pickle.load(open(file, 'rb'))

	def load_xuid_pairs(self, supp_features, scope):
		# TRUE represents we should filter the xuids to being only root events
		self.trainXUIDPairs = self.createXUIDPairs(self.trainXUIDs, scope, supp_features)
		self.devXUIDPairs = self.createXUIDPairs(self.devXUIDs, scope, supp_features)
		self.testXUIDPairs = self.createXUIDPairs(self.testXUIDs, scope, supp_features)

	def construct_tree_files_(self, is_wd):
		#(train_mt1, train_mt2) = 
		dir_path = "ecb_wd/"
		if not is_wd:
			dir_path = "ecb_cd/"

		evaluate_all_pairs = True

		# the True passed-in below corresponds to if we should construct only Trees that are rooted w/ events or not
		# (so, training should always be True, but dev and test can be False)
		self.sent_num_to_obj = {}
		self.trainXUIDPairs, self.train_tree_set = self.construct_tree_files(True, self.trainXUIDPairs, self.args.baseDir + "src/tree_lstm/data/sick/train/" + dir_path, True, evaluate_all_pairs)
		self.devXUIDPairs, self.dev_tree_set = self.construct_tree_files(False, self.devXUIDPairs, self.args.baseDir + "src/tree_lstm/data/sick/dev/" + dir_path, False, evaluate_all_pairs)
		self.testXUIDPairs, self.test_tree_set = self.construct_tree_files(False, self.testXUIDPairs, self.args.baseDir + "src/tree_lstm/data/sick/test/" + dir_path, False, evaluate_all_pairs)

	# produces files that TreeLSTM can read
	def construct_tree_files(self, is_training, xuid_pairs, dir_path, filter_only_roots, evaluate_all_pairs):
		print("dir_path:", dir_path)
		fout_a = open(dir_path + "a.toks", 'w')
		fout_b = open(dir_path + "b.toks", 'w')
		fout_sim = open(dir_path + "sim.txt", 'w')
		fout_a_deps = open(dir_path + "a.parents", 'w')
		fout_b_deps = open(dir_path + "b.parents", 'w')

		tmp_sentence_pairs = set()
		xuid_to_sentence_num = {}
		candidate_sentences = set()

		# gets unique IDs
		candidate_xuids = set()
		for xuid1, xuid2 in xuid_pairs:
			candidate_xuids.add(xuid1)
			candidate_xuids.add(xuid2)
		print("# unique candidate_xuids:", len(candidate_xuids))
		
		# constructs ecb tokens -> mentions
		# constructs xuid -> sentence #
		doc_ids = set()
		sentenceTokenToMention = defaultdict(lambda: defaultdict(set))
		for xuid in candidate_xuids:
			m = self.corpus.XUIDToMention[xuid]
			sentNum = m.globalSentenceNum
			doc_ids.add(m.doc_id)
			for t in m.tokens:
				sentenceTokenToMention[sentNum][t].add(m)
			xuid_to_sentence_num[xuid] = sentNum
			candidate_sentences.add(sentNum)

		# creates stan -> ecb tokens
		stanTokenToECBTokens = defaultdict(set)
		for doc_id in doc_ids:
			for t in self.corpus.doc_idToDocs[doc_id].tokens:
				for stan in t.stanTokens:
					stanTokenToECBTokens[stan].add(t)

		# creates sent -> XUIDs
		num_rootless_sent = 0
		for sent_num in candidate_sentences:
			# saves the SentTree, even if the Sentence doesn't have a proper root
			# bc this is only an issue if we need to filter by root (which would affect
			# the SentTrees we write out and the xuid pairs)
			st = self.construct_sent_tree(sent_num, candidate_xuids, stanTokenToECBTokens)
			if st.root_XUID == -1:
				num_rootless_sent += 1
			self.sent_num_to_obj[sent_num] = st
		print("# candidate_sentences:", len(candidate_sentences), " ==?== # sent:", len(self.sent_num_to_obj))
		print("num_rootless_sent:", num_rootless_sent)

		# gather the legit XUID pairs to use for train/test
		ret_xuid_pairs = []
		num_xuids_not_root = 0
		num_xuid_pairs_have_no_roots = 0
		for xuid1, xuid2 in xuid_pairs:

			sent_num1 = xuid_to_sentence_num[xuid1]
			sent_num2 = xuid_to_sentence_num[xuid2]
			# at a bare minimum, we can only train or test xuids if we
			# have both sentences in which they're contained
			if sent_num1 not in self.sent_num_to_obj or sent_num2 not in self.sent_num_to_obj:
				num_xuids_not_root += 1
				print("* ERROR: don't have the sentence corresponding to an XUID")
				exit()
				continue

			# we need to ignore all xuid pairs that are from sentences that don't contain valid roots
			if filter_only_roots:
				if self.sent_num_to_obj[sent_num1].root_XUID == -1 or self.sent_num_to_obj[sent_num2].root_XUID == -1:
					num_xuid_pairs_have_no_roots += 1
					continue

			# only look at the XUIDs which are roots
			if not evaluate_all_pairs:
				# current xuids must be roots
				if self.sent_num_to_obj[sent_num1].root_XUID == xuid1 and self.sent_num_to_obj[sent_num2].root_XUID == xuid2:
					ret_xuid_pairs.append((xuid1, xuid2))
			else: # if we don't require the xuid pairs to be of sentence roots, then we don't need to check anything
				ret_xuid_pairs.append((xuid1, xuid2))

		# AT THIS POINT: WE KNOW THAT THE SENTENCES FOR EVERY XUID PAIR IS FINE.  WE CAN WRITE IT OUT
		print("num_xuid_pairs_have_no_roots:", num_xuid_pairs_have_no_roots)
		# construct the TreeLSTM training based on the unique trees
		sent_legend = [] # keeps track of which sentences are on each line
		sent_labels = [] # only used for debugging
		sent_key_to_index = {} # maps the above so we don't need to search it
		xuid_pair_and_sent_key = [] # ((xuid1,xuid2), sent_key)
		
		numNegAdded = 0
		numPosAdded = 0
		num_same_sentences = 0
		num_pairs_belonging_to_null_sents = 0
		num_neg_not_added = 0
		for xuid1, xuid2 in ret_xuid_pairs:

			if xuid1 == xuid2:
				print("* ERROR: xuids are the same")
				exit()
			sent_num1 = xuid_to_sentence_num[xuid1]
			sent_num2 = xuid_to_sentence_num[xuid2]

			if is_training and sent_num1 == sent_num2:
				num_same_sentences += 1
				continue

			sent_label = 1
			root_xuid1 = self.sent_num_to_obj[sent_num1].root_XUID
			root_xuid2 = self.sent_num_to_obj[sent_num2].root_XUID
			if root_xuid1 == -1 or root_xuid2 == -1:
				num_pairs_belonging_to_null_sents += 1
				sent_label = 1
				if is_training:
					print("* ERROR: in training but sent have no xuid as root")
					exit()
			else:
				if self.corpus.XUIDToMention[root_xuid1].REF == self.corpus.XUIDToMention[root_xuid2].REF:
					sent_label = 2

			key = str(sent_num1) + "_" + str(sent_num2)
			tu = (sent_num1, sent_num2)
			if sent_num2 < sent_num1:
				key =  str(sent_num2) + "_" + str(sent_num1)
				tu = (sent_num2, sent_num1)
				sent_num3 = sent_num1
				sent_num1 = sent_num2
				sent_num2 = sent_num3

			# NOW: sent_num1 is always the lesser one
			if key not in sent_key_to_index:
				if sent_label == 1: # negative
					if is_training and numNegAdded > numPosAdded*self.args.numNegPerPos:
						num_neg_not_added += 1
						continue
					numNegAdded += 1
				else:
					numPosAdded += 1

				sent_key_to_index[key] = len(sent_key_to_index.keys())
				sent_legend.append(tu)
				sent_labels.append(sent_label)
				# actually write out the files
				fout_a.write(self.sent_num_to_obj[sent_num1].sent + "\n")
				fout_b.write(self.sent_num_to_obj[sent_num2].sent + "\n")
				fout_sim.write(str(sent_label) + "\n")
				fout_a_deps.write(self.sent_num_to_obj[sent_num1].dependency_chain + "\n")
				fout_b_deps.write(self.sent_num_to_obj[sent_num2].dependency_chain + "\n")
			# even if it's not a unique tree, we still need to store the xuid info
			xuid_pair_and_sent_key.append(((xuid1, xuid2), key))
		fout_a.close()
		fout_b.close()
		fout_sim.close()
		fout_a_deps.close()
		fout_b_deps.close()
		print("num_same_sentences:", num_same_sentences, "; num_pairs_belonging_to_null_sents:", num_pairs_belonging_to_null_sents)
		print("num_neg_not_added:", num_neg_not_added, "num_same_sentences:", num_same_sentences)
		print("# sent pairs:", len(sent_legend))
		i = 0
		for sent_num1, sent_num2 in sent_legend:
			if filter_only_roots:
				#print(sent_num1, sent_num2)
				#print(str(i), "s1:", " ".join([t.text for t in self.corpus.globalSentenceNumToTokens[sent_num1]]), "s2:", " ".join([t.text for t in self.corpus.globalSentenceNumToTokens[sent_num2]]), "label:", sent_labels[i])
				i += 1
				if self.sent_num_to_obj[sent_num1].root_XUID not in self.allXUIDs or \
					self.sent_num_to_obj[sent_num2].root_XUID not in self.allXUIDs:
					print("* one of our sentences doesn't have a root.  wtf")
					exit()
		print("# numPosAdded:", numPosAdded, "; numNegAdded:", numNegAdded)
		print("* orig xuid pairs:", len(xuid_pairs), "; # refined:", len(ret_xuid_pairs), "# sent pairs:", len(sent_legend), "# saved xuid pairs:", len(xuid_pair_and_sent_key))
		tree_set = TreeSet(sent_legend, sent_labels, sent_key_to_index, xuid_pair_and_sent_key)
		return ret_xuid_pairs, tree_set

	# we only care about the XUIDs that fit our scope (doc, dir, dirHalf)
	def construct_sent_tree(self, sent_num, candidate_xuids, stanTokenToECBTokens):

		# SAVED AS A TREE_SENT object
		sent = ""
		mention_token_indices = defaultdict(list) 
		dependency_chain = []
		root_XUID = -1

		# constructs token -> index (1-based)
		token_to_index = {} # only used locally
		token_index = 1
		for t in self.corpus.globalSentenceNumToTokens[sent_num]:
			if t.tokenID == "-1":
				continue

			for m in t.mentions:
				xuid = m.XUID
				if xuid in candidate_xuids: # we care about it
					mention_token_indices[xuid].append(token_index)
			token_to_index[t] = token_index
			sent += t.text + " "
			token_index += 1
		sent = sent.strip()

		# constructs dependency chain
		for t in self.corpus.globalSentenceNumToTokens[sent_num]:
			if t.tokenID == "-1":
				continue
			bestStan = self.getBestStanToken(t.stanTokens)
			if len(bestStan.parentLinks[self.helper.dependency_parse_type]) > 1:
				print("* more than 1 parent:", len(bestStan.parentLinks[self.helper.dependency_parse_type]))
				exit()

			pl = next(iter(bestStan.parentLinks[self.helper.dependency_parse_type]))
			parentToken = pl.parent
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
			print("* have != 1 dependency root")
			exit()
		if len(sent.split(" ")) != len(dependency_chain):
			print("* sent len != dependency length")
			exit()

		root_index = dependency_chain.index(0)
		root_token = self.corpus.globalSentenceNumToTokens[sent_num][root_index]
		for m in root_token.mentions:
			xuid = m.XUID
			# as long as the root is any mention in our corpus (could be a
			# singleton, doesn't have to belong to an xuid-pair)
			if xuid in self.allXUIDs:
				root_XUID = xuid
				break
		st = SentTree(sent, dependency_chain, mention_token_indices, root_XUID)
		return st

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
		return xuidPairs

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
