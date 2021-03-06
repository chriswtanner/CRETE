import pickle
import operator
import sys
import copy
from Mention import Mention
from StanDB import StanDB
from StanToken import StanToken
from collections import defaultdict
class ECBHelper:
	def __init__(self, args, event_pronouns, entity_pronouns):
		self.args = args
		self.corpus = None # should be passed-in
		self.dependency_parse_type = 'basic-dependencies'
		self.tmp_ref_to_abbr = {} # tmp -- just for printing dependency parse trees
		# data splits
		self.trainingDirs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 19, 20, 21, 22] #, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
		
		# immediately curtails the training dirs to being however many we specified at run-time passed-in params
		self.trainingDirs = self.trainingDirs[:self.args.num_dirs]

		self.devDirs = [23, 24, 25]
		#self.testingDirs = [36, 37, 38, 39, 40, 41, 42, 43, 44, 45]
		self.testingDirs = [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]

		self.pronouns = self.loadPronouns(args.pronounsFile) # load the regardless
		self.event_pronouns = event_pronouns
		self.entity_pronouns = entity_pronouns

		self.UIDToMention = defaultdict(list) # used only for ECBMentions

		# filled in via createHDDCRPMentions()
		# (maps text UID lines from CoNLL file to the XUID (aka HMUID) of the created HDDCRP Mention
		self.UIDToHMUID = defaultdict(list)

		# filled in via createStanMentions() (maps text UID
		# lines from CoNLL file to the created Stan Mention)
		# where StanMention is a regular Mention w/ regular Tokens
		# (which contain lists of StanTokens)
		self.UIDToSUID = defaultdict(list)
		self.docToVerifiedSentences = self.loadVerifiedSentences(args.verifiedSentencesFile)

		# TMP: created in addDependenciesToMention(), which maps each dep. relation to a unique #
		self.relationToIndex = {}

		self.predictions = defaultdict(float) # filled in with entity coref predictions

		# TMP, manually defines the relations we care about
		self.valid_relations = ['nsubj', 'dobj']

	# prints the entire corpus (regardless of if entities or events) to a file
	def printCorpus(self, filename_out):
		print("WRITING OUT CORPUS: ", str(filename_out))
		num_events = [m.isPred for m in self.corpus.ecb_mentions].count(True)
		print("\t# total mentions: ", str(len(self.corpus.ecb_mentions)), "(events = ", str(num_events), ")")
		
		fout = open(filename_out, 'w')
		'''
		for ref in self.corpus.refToEUIDs:
			print("---------------\nREF:", ref, "\n---------------")
			for euid in self.corpus.refToEUIDs[ref]:
		'''
		sentToTokens = defaultdict(list)
		for t in self.corpus.corpusTokens:
			sentNum = t.globalSentenceNum
			sentToTokens[sentNum].append(t)

		for dh in self.corpus.dirHalves.keys():
			print("\n=====================================")
			print("    DIR " + str(dh) + " (has " +
			      str(len(self.corpus.dirHalves[dh].REFToEUIDs.keys())) + " unique entity clusters, which follow below)")
			print("=====================================")
			for ref in self.corpus.dirHalves[dh].REFToEUIDs.keys():
				print("\n-------------\nREF: " + ref + "\n-------------")
				numEvents = 0
				numEntities = 0
				for euid in self.corpus.dirHalves[dh].REFToEUIDs[ref]:
					if euid in self.corpus.EUIDToMention.keys():
						mention = self.corpus.EUIDToMention[euid]
						if mention.isPred:
							numEvents += 1
						else:
							numEntities += 1
						firstMentionToken = mention.tokens[0]
						sentNum = firstMentionToken.globalSentenceNum
						tmpOut = ""
						inMention = False
						for t in sentToTokens[sentNum]:
							if t in mention.tokens:
								if inMention == False:
									tmpOut = tmpOut + "**[ "
									inMention = True
							else:
								if inMention == True:
									tmpOut = tmpOut + "]** "
									inMention = False
							tmpOut = tmpOut + t.text + " "
						print("\n",tmpOut)
					else:
						print("* didn't have", str(euid))
				print("\n\t--( # events:", str(numEvents), " # ents:", str(numEntities), ")")
		fout.close()

	# saves the entity coref predictions
	def addPredictions(self, ids, preds):
		if ids == None and preds == None:
			self.predictions = None
		else:
			for ((id1, id2), pred) in zip(ids, preds):
				self.predictions[(id1, id2)] = pred[0]

	# gathers all of the mentions we want, based on the passed-in type(s)
	def getCorpusMentions(self, mention_types):
		trainXUIDs = set()
		devXUIDs = set()
		testXUIDs = set()
		has_pronoun_count = 0
		has_no_pronoun_count = 0
		excluded_pronoun_mentions = []

		for m_types in mention_types:
			if m_types != "events" and m_types != "entities":
				print("* ERROR: incorrect mention type")
				exit(1)

		all_dirs = set(self.trainingDirs) | set(self.devDirs) | set(self.testingDirs)
		for m in self.corpus.ecb_mentions:
			if m.dir_num not in all_dirs:
				continue

			# only return the mentions that are the type(s) we care about
			if m.isPred and "events" not in mention_types:
				continue
			if not m.isPred and "entities" not in mention_types:
				continue

			# determines if it has a pronoun or not (and if we care)
			has_pronoun = False
			for t in m.tokens:
				for pronoun in self.pronouns:
					if pronoun == t.text and len(m.tokens) == 1:
						has_pronoun = True
						#print("has pronoun (bad count):", m)

			if has_pronoun:
				has_pronoun_count += 1
			else:
				has_no_pronoun_count += 1

			# possibly throw it out only if it's a pronoun and we don't want pronouns
			if has_pronoun:
				if m.isPred and not self.event_pronouns:
					excluded_pronoun_mentions.append(m.text)
					continue
				elif not m.isPred and not self.entity_pronouns:
					excluded_pronoun_mentions.append(m.text)
					continue

			# figures out which set we add it to
			if m.dir_num in self.trainingDirs: # training
				trainXUIDs.add(m.XUID)
			elif m.dir_num in self.devDirs:
				devXUIDs.add(m.XUID)
			elif self.args.useECBTest and m.dir_num in self.testingDirs:
				testXUIDs.add(m.XUID)
			else:
				print("* we should have already dismissed the mention")
				exit()
				continue
					
		print("has_pronoun_count:", has_pronoun_count)
		print("has_no_pronoun_count:", has_no_pronoun_count)
		print("# excluded mentions:", len(excluded_pronoun_mentions))

		#for i in excluded_pronoun_mentions:
		#	print("bad:", str(i))
		
		# conditionally add HDDCRP Mentions (as the test set)
		if not self.args.useECBTest:
			for xuid in self.corpus.HMUIDToMention:
				testXUIDs.add(xuid)

		return (trainXUIDs, devXUIDs, testXUIDs)

	def loadPronouns(self, filename):
		input_file = open(filename, 'r')
		return set(input_file.read().lower().strip().split("\n"))	

	def getEnsemblePreds(self, ensemblePreds):
		preds = []
		for i in range(len(ensemblePreds)):
			preds.append([sum(ensemblePreds[i]) / len(ensemblePreds[i])])
		return preds

	# takes pairwise predictions and adds to the dictionary's list
	def addEnsemblePredictions(self, withinDoc, relevant_dirs, ids, preds, ensemblePreds):
		i = 0
		print("lenids:", str(len(ids)), "# preds:", str(len(ensemblePreds)))
		for ((xuid1, xuid2), pred) in zip(ids, preds):
			
			m1 = self.corpus.XUIDToMention[xuid1]
			m2 = self.corpus.XUIDToMention[xuid2]
			# NOTE: the lower the score, the more likely they are the same.  it's a dissimilarity score
			pred = pred[0]
			doc_id = m1.doc_id
			if m1.dir_num not in relevant_dirs:
				print("* ERROR: passed in predictions which belong to a dir other than what we specify")
				exit(1)
			if withinDoc and m2.doc_id != doc_id:
				print("* ERROR1: xuids are from diff docs!")
				exit(1)

			if len(ensemblePreds) == len(ids): # not empty
				ensemblePreds[i].append(pred)
			else:
				ensemblePreds.append([pred])
			i += 1
		print("len(ensemblePreds):", str(len(ensemblePreds)))

	# evaluates the CCNN pairwise predictions,
	# returning the F1, PREC, RECALL scores
	def evaluatePairwisePreds(self, ids, preds, golds, dh, rev=False):

		A = set()
		B = set()
		C = set()
		D = set()

		#print("* in evaluatePairwisePreds()")
		numGoldPos = 0
		scoreToGoldTruth = defaultdict(list)
		
		acc = 0
		#for p, g in zip(preds[0:15], golds[0:15]):
		#	print("pred:", p, "gold:", g)

		for _ in range(len(preds)):
			#print("looking at pred:", preds[_][0], "and gold:", golds[_])
			if golds[_] == 0:
				numGoldPos += 1
				scoreToGoldTruth[preds[_][0]].append(1)
				#print("a")
			else:
				scoreToGoldTruth[preds[_][0]].append(0)
				#print("")

		
		s = sorted(scoreToGoldTruth.keys(), reverse=rev)
		'''
		for _ in s:
			print("score:", _, "trutH:", scoreToGoldTruth[_])
		'''

		#print("numGoldPos:", numGoldPos)
		TP = 0.0
		FP = 0.0
		bestF1 = 0
		bestVal = -1
		bestR = 0
		bestP = 0
		numReturnedSoFar = 0
		score_to_index_rank = {}
		for eachVal in s:
			for _ in scoreToGoldTruth[eachVal]:
				if _ == 1:
					TP += 1
				else:
					FP += 1

			numReturnedSoFar += len(scoreToGoldTruth[eachVal])
			#print("numReturnedSoFar:", numReturnedSoFar)
			score_rounded = str(round(eachVal,7))
			score_to_index_rank[score_rounded] = numReturnedSoFar

			recall = float(TP / numGoldPos)
			prec = float(TP / numReturnedSoFar)
			f1 = 0
			if (recall + prec) > 0:
				f1 = 2*(recall*prec) / (recall + prec)

			if f1 > bestF1:
				bestF1 = f1
				bestVal = eachVal
				bestR = recall
				bestP = prec
				#print("new besT: thres:", eachVal, "prec:", prec, "rec:", recall, "f1:", f1)
		if numReturnedSoFar != len(preds):
			print("* ERROR: we didn't look at preds correctly")
			exit(1)
		if bestF1 <0:
			print("* ERROR: our F1 was < 0")
			exit(1)

		# given the best threshold, now let's check the individual performance
		# of both entities and events
		mentionStats = defaultdict(lambda: defaultdict(int))
		the_pairs = defaultdict(lambda: defaultdict(int))
		TP = 0.0
		FP = 0.0
		num_gold = 0
		num_predicted = 0
		num_wrong = 0
		num_right = 0
		pred_to_gold_features = defaultdict(list)
		pred_to_ids_error  = defaultdict(list)
		wrong_pairs = set()
		fp_pairs = []
		fn_pairs = []

		#print("# fp_pairs:", len(fp_pairs), "# fn_pairs:", len(fn_pairs))
		return (bestF1, bestP, bestR, bestVal)

	def get_all_valid_1hops(self, dh, token, valid_1hops, valid_relations):
		#print("get_all_valid_1hops() on token:", token)
		#print("init valid_1hops:", valid_1hops)
		bestStan = dh.getBestStanToken(token.stanTokens)
		#print("\tbestStan:", bestStan)
		for cl in bestStan.childLinks[self.dependency_parse_type]:
			#print("\t\tcl:", cl)
			ecbTokens = self.stanTokenToECBTokens[cl.child]
			#print("\t\t\tecbtokens:", ecbTokens)

			found_valid_relation = ""
			for rel in valid_relations:
				#print("\tvalid relation:", rel)
				if cl.relationship.startswith(rel):
					found_valid_relation = rel
					#print("\tfound_valid_relation!!! = ", rel)
					break
			if found_valid_relation != "":
				#print("\twe found a valid relation, ", rel)
				for child_token in ecbTokens:
					#print("\t\tlooking at child:", child_token)
					valid_1hops[rel].add(child_token)
					#print("\t\t\tvalid_1hops nwo:", valid_1hops)
		#print("* returning valid_1hops:", valid_1hops)

	# returns a list of all paths to an entity mention (a list of lists)
	def getAllChildrenMentionPaths(self, dh, tokenToMentions, originalMentionStans, token, curPath, allPaths):
		foundEntity = False
		if token in tokenToMentions and len(curPath) > 0: # we hit a mention of some sort (event or entity)
			#print("\t [ found a mention!]")
			foundMentions = tokenToMentions[token]
			for m in foundMentions:
				if not m.isPred:
					allPaths.append(curPath)
					foundEntity = True
					#print("\t\t and it's an entity!")
		if not foundEntity:
			#print("\t [ NO MENTION]")
			bestStan = dh.getBestStanToken(token.stanTokens)
			for cl in bestStan.childLinks[self.dependency_parse_type]:
				if cl.child not in originalMentionStans and cl not in curPath:
					ecbTokens = self.stanTokenToECBTokens[cl.child]
					for ecbToken in ecbTokens:
						
						#new_visited = copy.copy(visited) # creates its own copy of the visited
						new_path = copy.copy(curPath) # creates its own copy of the visited
						new_path.append(cl) # adds our new edge
						#print("\tnew_path:", str(new_path), " with token we'll explore: ", str(ecbToken))
						self.getAllChildrenMentionPaths(dh, tokenToMentions, originalMentionStans, ecbToken, new_path, allPaths)

	def checkDependencyRelations(self):
		distancesCounts = defaultdict(int)
		distanceZero = 0
		distanceNonZero = 0
		entitiesCoref = 0
		entitiesNoCoref = 0

		howManyEntities = defaultdict(int) # Q0
		howManyLevelEntitiesAppear = defaultdict(int) # Q1
		depthOfFirstEntity = defaultdict(int) # Q2
		distOfEntitiesAtFirstLevel = defaultdict(int) # Q3
		bothEventAndEntitiesExist = defaultdict(lambda: defaultdict(int)) # Q4
		bothEventAndEntitiesCoref = defaultdict(lambda: defaultdict(int)) # Q5
		#bothEventAndEntitiesCoref
		num_pairs = 0

		eventsConsidered = set()

		for dh in sorted(self.corpus.dirHalves):
			#print("dh:",dh)

			for euid1 in sorted(self.corpus.dirHalves[dh].EUIDs):
				m1 = self.corpus.EUIDToMention[euid1]
				if not m1.isPred:
					continue
				#print("m1:", m1)
				if m1 in eventsConsidered:
					print("ALREADY HAVE IT")
					exit(1)
				eventsConsidered.add(m1)

				# TODO tmp: only look at pairs where both of their entities occurred at 1 hope
				# for now, let's just store the 1-hop ones
				m1_paths = []
				
				if 1 in m1.levelToEntityPath.keys():
					for p in m1.levelToEntityPath[1]:
						m1_paths.append(p)
				else:
					continue
				
				# Q1
				num_levels = len(m1.levelToChildrenMentions.keys())
				howManyLevelEntitiesAppear[num_levels] += 1

				# Q2 depthOfFirstEntity
				if len(m1.levelToChildrenMentions) > 0:
					shortest_level = sorted(m1.levelToChildrenMentions)[0]
					depthOfFirstEntity[shortest_level] += 1
				else:
					depthOfFirstEntity[0] += 1

				# Q3
				# print("*** M1 ents: ", str(m1.childrenLinked))
				m1_shortests = set()
				if len(m1.levelToChildrenMentions) > 0:
					shortest_level = sorted(m1.levelToChildrenMentions)[0]
					for ents in m1.levelToChildrenMentions[shortest_level]:
						m1_shortests.add(ents)

					num_found = len(m1.levelToChildrenMentions[shortest_level])
					distOfEntitiesAtFirstLevel[num_found] += 1
				else:
					distOfEntitiesAtFirstLevel[0] += 1

				# Q6: all entities
				m1_allEntities = set()
				for level in m1.levelToChildrenMentions:
					for ents in m1.levelToChildrenMentions[level]:
						m1_allEntities.add(ents)

				# Q0: how many entities
				howManyEntities[len(m1_allEntities)] += 1

				for euid2 in sorted(self.corpus.dirHalves[dh].EUIDs):
					if euid1 >= euid2:
						continue

					m2 = self.corpus.EUIDToMention[euid2]
					if not m2.isPred:
						continue

					# TODO: tmp: only look at pairs which both have entities 1 hop away
					m2_paths = []
					
					if 1 in m2.levelToEntityPath.keys():
						for p in m2.levelToEntityPath[1]:
							m2_paths.append(p)
					else:
						continue
					
					#if m1.REF != m2.REF:
					#	continue

					# WITHIN DOC
					if m1.doc_id != m2.doc_id: # == means cross
						continue

					# Q6
					m2_allEntities = set()
					for level in m2.levelToChildrenMentions:
						for ents in m2.levelToChildrenMentions[level]:
							m2_allEntities.add(ents)

					num_pairs += 1

					# Q4
					entBothExist = False
					if len(m1.levelToChildrenMentions) > 0 and len(m2.levelToChildrenMentions) > 0:
						entBothExist = True
					eventsCoref = False
					if m1.REF == m2.REF:
						eventsCoref = True
					bothEventAndEntitiesExist[eventsCoref][entBothExist] += 1

					# no man's land.  some other basic stat i was doing
					m2_shortests = set()
					if len(m2.levelToChildrenMentions) > 0:
						shortest_level = sorted(m2.levelToChildrenMentions)[0]
						for ents in m2.levelToChildrenMentions[shortest_level]:
							m2_shortests.add(ents)
				
					entcoref = False
					for ment1 in m1_shortests:
						for ment2 in m2_shortests:
							if ment1.REF == ment2.REF:
								entcoref = True
								break
					
					haveIdenticalPath = False
					for m1p in m1_paths:
						for m2p in m2_paths:
							if m1p[0] == m2p[0]:
								haveIdenticalPath = True
								break

					if entcoref:
						entitiesCoref += 1
					else:
						entitiesNoCoref += 1
					
					#######

					flag = haveIdenticalPath and entcoref

					# Q5
					if entBothExist:
						bothEventAndEntitiesCoref[eventsCoref][flag] += 1
						#bothEventAndEntitiesCoref[eventsCoref][entcoref] += 1

					#print("m2_shortests:", str(m2_shortests))
					# only compare m1,m2 if they're both events
					if len(m1_shortests) == 0 or len(m2_shortests) == 0:
						distanceZero += 1
					else:
						distanceNonZero += 1
					if len(m1_shortests) <= len(m2_shortests):
						distancesCounts[(len(m1_shortests),len(m2_shortests))] += 1
					else:
						distancesCounts[(len(m2_shortests),len(m1_shortests))] += 1
		

		print("distancesCounts:", str(distancesCounts))
		print("distanceNonZero:", str(distanceNonZero))
		print("distanceZero:", str(distanceZero))
		print("% non-zero:", str(float(distanceNonZero / (distanceNonZero + distanceZero))))
		print("entitiesCoref:", str(entitiesCoref))
		print("entitiesNoCoref:", str(entitiesNoCoref))
		print("Q0: howManyEntities:", str(howManyEntities))
		print("Q1: howManyLevelEntitiesAppear: " + str(howManyLevelEntitiesAppear))
		print("Q2: depthOfFirstEntity:" + str(depthOfFirstEntity))
		print("sum:"  + str(sum([depthOfFirstEntity[key] for key in depthOfFirstEntity.keys()])))
		print("Q3: distOfEntitiesAtFirstLevel:", distOfEntitiesAtFirstLevel)
		print("Q4: bothEventAndEntitiesExist:", bothEventAndEntitiesExist)
		print("Q5: bothEventAndEntitiesCoref:", bothEventAndEntitiesCoref)
		print("num_pairs:", str(num_pairs))
		print("eventsConsidered:", str(len(eventsConsidered)))

	def dfs_tree(self, stan_node, text_branches, cur_nodes, cur_rel, sentenceTokenToMention, stanTokenToECBTokens):
		#print("* dfs received:", stan_node, text_branches, cur_nodes)
		new_cur_nodes = cur_nodes.copy()
		if not stan_node.isRoot:
			ecb_tokens = stanTokenToECBTokens[stan_node]
			#print("ecb_tokenszz:", ecb_tokens)
			ecb_token = next(iter(ecb_tokens))
			#print("\tadding ecb_token:", ecb_token)
			new_cur_nodes.append(ecb_token)
			str_line = ""
			for _ in range(len(text_branches)-1):
				if text_branches[_] == 1:
					str_line += "|                "
				else:
					str_line += "                 "
			if len(cur_nodes) > 0:
				str_line += "|--(" + cur_rel[0:6] + ")--> "
			if ecb_token in sentenceTokenToMention:

				mention = next(iter(sentenceTokenToMention[ecb_token]))
				ref_id = -1
				if mention.REF in self.tmp_ref_to_abbr:
					ref_id = self.tmp_ref_to_abbr[mention.REF]
				else:
					ref_id = len(self.tmp_ref_to_abbr.keys())
					self.tmp_ref_to_abbr[mention.REF] = ref_id
				found_event = False
				for _ in sentenceTokenToMention[ecb_token]:
					if _.isPred:
						found_event = True
				if found_event:
					str_line += "*R" + str(ref_id) + "[" + ecb_token.text + "]"
				else:
					str_line += "R" + str(ref_id) + "[" + ecb_token.text + "]"
			else:
				str_line += ecb_token.text
			print(str_line)
			#self.fout.write(str_line + "\n")
		
		for _ in range(len(stan_node.childLinks[self.dependency_parse_type])):
			child_link = stan_node.childLinks[self.dependency_parse_type][_]
			#print("\t", str(_), "child_lnk:", child_link)
			new_text_branches = text_branches.copy()
			if _ == len(stan_node.childLinks[self.dependency_parse_type]) - 1 and not stan_node.isRoot:
				new_text_branches.append(0)
			elif not stan_node.isRoot:
				new_text_branches.append(1)
			child_stan = child_link.child
			#print("\tchild:", child_stan)
			self.dfs_tree(child_stan, new_text_branches, new_cur_nodes, child_link.relationship, sentenceTokenToMention, stanTokenToECBTokens)

	def addDependenciesToMentions(self, dh):
		# TMP: keeps track of how many event mentions have entities attached or not
		have_ent = 0
		not_have_ent = 0
		sentences_we_looked_at = set()
		sentences_with_both_connections = set()

		eventsConsidered = set()
		relation_to_count = defaultdict(int)
		shortest_path_to_ent = defaultdict(int) # keeps track of how many events had their shortest path to be of depth N
		for doc_id in self.corpus.doc_idToDocs:
			#self.fout = open(doc_id + "_parsetree.txt", 'w')
			# maps ECB Token -> StanToken and vice versa
			self.stanTokenToECBTokens = defaultdict(set)
			curdoctokens = ""
			for t in self.corpus.doc_idToDocs[doc_id].tokens:
				curdoctokens += t.text + " "
				for s in t.stanTokens:
					self.stanTokenToECBTokens[s].add(t)

			# checks if a given stan token maps to multiple ECB tokens
			'''
			for k in self.stanTokenToECBTokens:
				if len(self.stanTokenToECBTokens[k]) > 1:
					print("woops, we have len of :", len(self.stanTokenToECBTokens[k]), ":", k)
			'''

			# looks through each mention, to print the most immediate governor
			# and modifier mentions of the opposite type
			# maps each mention to a SENTENCE
			sentenceToEventMentions = defaultdict(set)
			sentenceToEntityMentions = defaultdict(set)
			sentenceTokenToMention = defaultdict(lambda: defaultdict(set))
			#self.fout.write("\n-----------------------------------------\ndocid:" + doc_id + "\n-----------------------------------------\n")
			num_entities = 0
			num_events = 0

			for euid in self.corpus.doc_idToDocs[doc_id].EUIDs:

				m = self.corpus.EUIDToMention[euid]
				sentNum = m.globalSentenceNum
				if m.isPred:
					num_events += 1
				else:
					num_entities += 1

				#print("\tmention:", m)
				for t in m.tokens:
					sentenceTokenToMention[sentNum][t].add(m)
					#print("* setting", sentNum, ":", t, " to be", m)

				if m.isPred:
					sentenceToEventMentions[sentNum].add(m)
				else:
					sentenceToEntityMentions[sentNum].add(m)

			for s in sentenceToEventMentions:
				
				sentences_we_looked_at.add(s)

				tokenText = ""
				first_token = None
				for t in self.corpus.globalSentenceNumToTokens[s]:
					if t.tokenID == "-1":
						continue
					bestStan = dh.getBestStanToken(t.stanTokens)
					for pl in bestStan.parentLinks[self.dependency_parse_type]:
						parentToken = pl.parent
						if parentToken.isRoot:
							first_token = parentToken
							break
					tokenText += t.text + " "
		
				for m in sentenceToEventMentions[s]:

					# TMP, eugene's idea of using just a few relations (1-hop away)
					#print("\tentities in this sent:", sentenceToEntityMentions[s])
					valid_1hops = defaultdict(set)
					for mention_token in m.tokens:
						self.get_all_valid_1hops(dh, mention_token, valid_1hops, self.valid_relations)
					m.set_valid1hops(valid_1hops, sentenceTokenToMention[s])

					# we do this here, but the main time is below.  this is just for debugging purposes
					mentionStanTokens = set()
					for t in m.tokens:
						bestStan = dh.getBestStanToken(t.stanTokens)
						mentionStanTokens.add(bestStan)

					eventsConsidered.add(m)
				
					# gets the StanTokens for the current mention, so that we
					# never explore any of them as parents or chilren
					mentionStanTokens = set()
					for t in m.tokens:
						bestStan = dh.getBestStanToken(t.stanTokens)
						mentionStanTokens.add(bestStan)

					# finds its children (modifiers)
					# was unreliable in getChildren()
					# bc i dont explore all ecbtokens, only the first.  now it should be good, bc i explore all paths
					for t in m.tokens:
						allPaths = []
						curPath = []

						# constructs all paths that have a way to entities
						allPaths = []
						self.getAllChildrenMentionPaths(dh, sentenceTokenToMention[s], mentionStanTokens, t, curPath, allPaths)

						one_hop_relations = [] # stores the dependency relation that took us to an entity in just 1 hop
						for path in allPaths:
							m.pathsToChildrenEntities.append(path)
							level = 1
							for p in path:
								relation = p.relationship
								cur_stan = p.child
								ecbTokens = self.stanTokenToECBTokens[cur_stan]
								for ecb in ecbTokens:
									if ecb in sentenceTokenToMention[s]:
										foundMentions = sentenceTokenToMention[s][ecb]
										for mfound in foundMentions:
											if not mfound.isPred: # mfound is an ENTITY

												if not m.isPred or mfound.isPred:
													print("** wrong types")
													exit(1)

												one_hop_relations.append(relation)

												# ADDS LINKS FROM EVENT DOWN TO ENTITY
												m.childrenLinked.add(mfound) # A
												m.levelToChildrenMentions[level].add(mfound) # B
												if (mfound, path) not in m.levelToChildren[level]: # C
													m.levelToChildren[level].append((mfound, path))

												# ADDS THE REVERSE LINKS FROM ENTITY UP TO EVENTS
												mfound.parentsLinked.add(m) # A
												mfound.levelToParentMentions[level].add(m) # B
												if (m, path.reverse()) not in mfound.levelToParents[level]: # C
													mfound.levelToParents[level].append((m, path))

								level += 1

						'''
						print("m:", m)
						print("\tpathsToChildrenEntities:", m.pathsToChildrenEntities)
						print("\tlevelToChildrenMentions:", m.levelToChildrenMentions)
						for l in m.levelToChildrenMentions:
							for m2 in m.levelToChildrenMentions[l]:
								print("\tchlid:", m2)
						exit()
						'''
						# the shortest level/depth at which an entity appears
						shortest_level = 0 if len(m.levelToChildrenMentions) == 0 else \
							next(iter(sorted(m.levelToChildrenMentions)))

						shortest_path_to_ent[shortest_level] += 1
						
						for rel in one_hop_relations:
							relation_to_count[rel] += 1

					'''
					print("event:", m)
					print("\tchildrenLinked:", [str(_) for _ in m.childrenLinked])
					print("\tlevelToChildrenMentions:", m.levelToChildrenMentions)
					print("\tlevelToChildren:", m.levelToChildren)
					'''

			# END OF SENTENCE	
		# END OF ALL DOCS
		print("eventsConsidered:", str(len(eventsConsidered)))
		sorted_x = sorted(relation_to_count.items(), key=operator.itemgetter(1), reverse=True)
		rel_num = 0
		for rel in sorted_x:
			print("rel:", rel)
			self.relationToIndex[rel] = rel_num
			rel_num += 1

		print("have_ent:", have_ent)
		print("not_have_ent:", not_have_ent)
		print("# sentences_we_looked_at:", len(sentences_we_looked_at))
		print("# which have both:", len(sentences_with_both_connections))

		# optionally prints a shortened REF key so that i can display the dependency parse trees in a sane manner
		'''
		for dir_half in self.corpus.dirHalves:
			print("dirhalf:", dir_half, "\n---------------")
			for ref in self.corpus.dirHalves[dir_half].REFToEUIDs:
				if ref not in self.tmp_ref_to_abbr:
					print("***** WHY isn't:", ref, "in self.tmp_ref_to_abbr.  size:", len(self.tmp_ref_to_abbr))
				else:
					print("\tREF:", self.tmp_ref_to_abbr[ref])
					for uid in self.corpus.dirHalves[dir_half].REFToEUIDs[ref]:
						men = self.corpus.EUIDToMention[uid]
						print("\t\tuid:", uid, "men:", men)
		'''
		#print(self.relationToIndex)
		#prints coutns of dependency relations
		#for x in sorted_x:
		#	print(x[0], ",", x[1])

		#for level in shortest_path_to_ent.keys():
		#	print("level", level, " = ", str(shortest_path_to_ent[level]))
		

	def writeCoNLLFile(self, predictedClusters, suffix): # suffix should be wd_"${sp}"_"${run#}".txt" or cd instead of wd
		hmuidToClusterID = {}
		for c_id in predictedClusters.keys():
			for hmuid in predictedClusters[c_id]:
				hmuidToClusterID[hmuid] = c_id

		#print("# hmuid:", str(len(hmuidToClusterID.keys())))
		# sanity check
		'''
		for hmuid in self.hddcrp_parsed.hm_idToHMention.keys():
			if hmuid not in hmuidToClusterID.keys():
				print("ERROR: hmuid:",str(hmuid),"NOT FOUND within our clusters, but it's parsed!")
				exit(1)
		'''
		# constructs output file
		fileOut = str(self.args.baseDir) + "results/hddcrp_pred_" + str(suffix) + ".txt"
		print("ECBHelper writing out:", str(fileOut))
		fout = open(fileOut, 'w')

		#print("self.UIDToHMUID:", self.UIDToHMUID)

		# reads the original CoNLL prediction file (not golden), while writing each line
		f = open(self.args.hddcrpFullFile, 'r')
		tokenIndex = 0
		REFToStartTuple = defaultdict(list)
		for line in f:
			line = line.rstrip()
			tokens = line.split("\t")
			if line.startswith("#") and "document" in line:
				sentenceNum = 0
				fout.write(line + "\n")
			elif line == "":
				sentenceNum += 1
				fout.write(line + "\n")
			elif len(tokens) == 5:
				doc, _, tokenNum, text, ref_ = tokens
				UID = str(doc) + ";" + str(sentenceNum) + ";" + str(tokenNum)

				#print("\tfound a line:",UID)
				# reconstructs the HMention(s) that exist on this line, for the
				# sake of being able to now look up what cluster assignent it/they belong to
				hmentions = set()
				for hmuid in self.UIDToHMUID[UID]:
					#print("hmuid:",hmuid)
					hmentions.add(self.corpus.XUIDToMention[hmuid])
					#print("corpus.XUIDToMention:", self.corpus.XUIDToMention)
				#print("line:", UID, ": hmentions: ", hmentions)
				refs = []
				if ref_.find("|") == -1:
					refs.append(ref_)
				else:  # we at most have 1 "|""
					refs.append(ref_[0:ref_.find("|")])
					refs.append(ref_[ref_.find("|")+1:])
					#print("***** FOUND 2:",str(line))

				if (len(refs) == 1 and refs[0] == "-"):
					# just output it, since we want to keep the same mention going
					fout.write(line + "\n")
				else:
					ref_section = ""
					isFirst = True
					for ref in refs:
						if ref[0] == "(" and ref[-1] != ")":  # i.e. (ref_id
							ref_id = int(ref[1:])
							REFToStartTuple[ref_id].append((tokenIndex, isFirst))
							startTuple = (tokenIndex, isFirst)
							foundMention = False
							for hmention in hmentions:
								if hmention.REF == ref_id and hmention.startTuple == startTuple:  # we found the exact mention
									foundMention = True
									hmuid = hmention.XUID
									if hmuid in hmuidToClusterID:
										clusterID = hmuidToClusterID[hmuid]
										ref_section += "(" + str(clusterID)
										break
							if not foundMention:
								print("* ERROR #1, we never found the mention for this line:", str(line))
								ref_section = "-"
								#exit(1)

						# represents we are ending a mention
						elif ref[-1] == ")":  # i.e., ref_id) or (ref_id)
							ref_id = -1

							endTuple = (tokenIndex, isFirst)
							startTuple = ()
							# we set ref_if, tokens, UID
							if ref[0] != "(":  # ref_id)
								ref_id = int(ref[:-1])
								startTuple = REFToStartTuple[ref_id].pop()
							else:  # (ref_id)
								ref_id = int(ref[1:-1])
								startTuple = (tokenIndex, isFirst)
								ref_section += "("

							#print("starttuple:",str(startTuple))
							#print("endTuple:",str(endTuple))

							foundMention = False
							for hmention in hmentions:
								#print("looking at hmention:",str(hmention))
								if hmention.REF == ref_id and hmention.startTuple == startTuple and hmention.endTuple == endTuple:  # we found the exact mention
									foundMention = True
									hmuid = hmention.XUID
									if hmuid in hmuidToClusterID:
										clusterID = hmuidToClusterID[hmuid]
										ref_section += str(clusterID) + ")"
										break
							if not foundMention:
								print("* ERROR #2, we never found the mention for this line:", str(line))
								ref_section = "-"
								#exit(1)

						if len(refs) == 2 and isFirst:
							ref_section += "|"
						isFirst = False
					fout.write(str(doc) + "\t" + str(_) + "\t" + str(tokenNum) +
											"\t" + str(text) + "\t" + str(ref_section) + "\n")
					# end of current token line
				tokenIndex += 1  # this always increases whenever we see a token
		f.close()
		fout.close()

	def addECBCorpus(self, corpus):
		self.corpus = corpus
		for m in corpus.ecb_mentions:
			for t in m.tokens:
				self.UIDToMention[t.UID] = m

	# ECB+ only makes guarantees that the following sentences are correctly annotated
	def loadVerifiedSentences(self, sentFile):
		docToVerifiedSentences = defaultdict(set)
		f = open(sentFile, 'r')
		f.readline()
		for line in f:
			tokens = line.rstrip().split(",")
			doc_id = tokens[0] + "_" + tokens[1] + ".xml"
			sentNum = tokens[2]
			docToVerifiedSentences[doc_id].add(int(sentNum))
		f.close()
		return docToVerifiedSentences

	# pickles the StanTokens
	def saveStanTokens(self):
		s = StanDB()
		s.UIDToStanTokens = {}  # UID -> StanToken[]
		for t in self.corpus.corpusTokens:
			s.UIDToStanTokens[t.UID] = t.stanTokens
		print("* writing out", len(s.UIDToStanTokens), "UIDs' StanTokens")
		pickle_out = open(self.args.stanTokensFile, 'wb')
		pickle.dump(s, pickle_out)

	# reads in the pickled StanTokens
	def loadStanTokens(self):
		pickle_in = open(self.args.stanTokensFile, 'rb')
		stan_db = pickle.load(pickle_in)
		for uid in stan_db.UIDToStanTokens:
			if uid in self.corpus.UIDToToken:
				self.corpus.UIDToToken[uid].stanTokens = stan_db.UIDToStanTokens[uid]
		print("* [StanDB] loaded", len(stan_db.UIDToStanTokens), "UIDs' StanTokens")

	def addStanfordAnnotations(self, stanfordParser):
		stanDocSet = set()
		for doc_id in stanfordParser.docToSentenceTokens.keys():
			stanDocSet.add(doc_id)
		# adds stan links on a per doc basis
		for doc_id in sorted(stanDocSet):
			stanTokens = []  # builds list of stanford tokens
			for sent_num in sorted(stanfordParser.docToSentenceTokens[doc_id].keys()):
				for token_num in stanfordParser.docToSentenceTokens[doc_id][sent_num]:
					sToken = stanfordParser.docToSentenceTokens[doc_id][sent_num][token_num]
					if sToken.isRoot == False:
						stanTokens.append(sToken)

			# for readability, make a new var
			ourTokens = self.corpus.doc_idToDocs[doc_id].tokens
					
			j = 0
			i = 0
			while i < len(ourTokens):
				if j >= len(stanTokens):
					if i == len(ourTokens) - 1 and stanTokens[-1].text == "...":
						ourTokens[i].addStanTokens([stanTokens[-1]])
						break
					elif i == len(ourTokens) - 1 and ourTokens[i].text == ".":
						print("ADDING a final pseudo-Stan of .")
						prevSentenceNum = stanTokens[j-1].sentenceNum
						prevEndIndex = stanTokens[j-1].endIndex
						periodToken = StanToken(False, prevSentenceNum, 0, ".", ".", prevEndIndex, prevEndIndex, ".", "O")
						ourTokens[i].addStanTokens([periodToken])
						break
					else:
						print("ran out of stan tokens")
						exit(1)

				stanToken = stanTokens[j]
				ourToken = ourTokens[i]

				curStanTokens = [stanToken]
				curOurTokens = [ourToken]

				stan = stanToken.text
				ours = ourToken.text

				# pre-processing fixes since the replacements file can't handle spaces
				if stan == "''":
					stan = "\""
				elif stan == "2 1/2":
					stan = "2 1/2"
				elif stan == "3 1/2":
					stan = "3 1/2"
				elif stan == "877 268 9324":
					stan = "8772689324"
				elif stan == "0845 125 2222":
					stan = "08451252222"
				elif stan == "0800 555 111":
					stan = "0800555111"
				elif stan == "0800 555111":
					stan = "0800555111"
				elif stan == "0845 125 222":
					stan = "0845125222"

				# pre-check for unalignable tokens
				# ours: "." stan misses it completely
				# NOTE: if we're here, it means we are aligned up
				# to this point, so the only options are:
				# (1) stan replaced . with a :
				isBlank = False
				if ours == "." or ours.strip(" ") == "":
					isBlank = True
				if isBlank and stan != "." and stan != "...": 
					ours = "."
					if stan == ":":
						stanToken.text = "."
						stan = "."
					else:
						#print("ADDING a pseudo-Stan of .")
						stan = "."
						j = j - 1
						prevSentenceNum = stanTokens[j].sentenceNum
						prevEndIndex = stanTokens[j].endIndex
						periodToken = StanToken(False, prevSentenceNum, 0, ".", ".", prevEndIndex, prevEndIndex, ".", "O")
						curStanTokens = [periodToken]
				elif stan == "." and ours != "." and ours != "...":
					print("** SKIPPING OVER THE STAN TOKEN .")
					j += 1
					continue

				# get the words to equal lengths first
				while len(ours) != len(stan):
					#print("ne!")
					sys.stdout.flush()
					while len(ours) > len(stan):
						#print("\tstan length is shorter:", str(ours)," vs:",str(stan)," stanlength:",str(len(stan)))
						if j+1 < len(stanTokens):
							if stanTokens[j+1].text == "''":
								stanTokens[j+1].text = "\""
								print("TRYING TO FIX THE UPCOMING STAN TOKEN!")
							stan += stanTokens[j+1].text

							curStanTokens.append(stanTokens[j+1])
							if stan == "71/2":
								stan = "7 ½"
							elif stan == "31/2":
								stan = "3½"
							j += 1
						else:
							print("\tran out of stanTokens: ours",ours,"stan:",stan)
							exit(1)

					while len(ours) < len(stan):
						if i+1 < len(ourTokens):
							ours += ourTokens[i+1].text
							curOurTokens.append(ourTokens[i+1])
							if ours == "31/2":
								ours = "3 1/2"
							elif ours == "21/2":
								ours = "2 1/2"
							elif ours == "31/2-inch":
								ours = "3 1/2-inch"
							elif ours == "3 1/2":
								ours = "3 1/2"
							i += 1
						else:
							print("\t** ran out of ourTokens")
							exit(1)

				if ours.lower() != stan.lower():
					print("\tMISMATCH: [", str(ours), "] [", str(stan), "]")
					exit(1)
				else:  # texts are identical, so let's set the stanTokens
					for t in curOurTokens:
						t.addStanTokens(curStanTokens)

				j += 1
				i += 1
			# ensures every Token in the doc has been assigned at least 1 StanToken
			for t in self.corpus.doc_idToDocs[doc_id].tokens:
				if len(t.stanTokens) == 0:
					print("Token:", str(t), " never linked w/ a stanToken!")
					exit(1)
			#print("finished w/ doc:",doc_id)
		print("we've successfully added stanford links to every single token within our", str(len(self.corpus.doc_idToDocs)), "docs")

	def createStanMentions(self):
		last_ner = ""
		toBeMentions = [] # list of StanToken Lists
		curTokens = []
		tokenToNER = {} # only used for creating a 'type' field in Mention
		lastSentenceNum = -1
		for each_token in self.corpus.corpusTokens:
			if each_token.sentenceNum != lastSentenceNum:
				last_ner = "" # reset
				if lastSentenceNum != -1 and len(curTokens) > 0: # leftover from last sentence
					toBeMentions.append(curTokens)
			if self.args.onlyValidSentences and each_token.sentenceNum not in self.docToVerifiedSentences[each_token.doc_id]:
				continue
			cur_ner = ""
			for st in each_token.stanTokens:
				if st.ner != "O" and st.ner != "0":
					cur_ner = st.ner
			if cur_ner != "":
				tokenToNER[each_token] = cur_ner

			if cur_ner != last_ner: # possibly ending, possibly starting
				if last_ner != "": # ending
					toBeMentions.append(curTokens)
					curTokens = []  # new clean slate
				if cur_ner != "": # starting
					curTokens = [] # reinforcement
					curTokens.append(each_token)
			elif cur_ner == last_ner and cur_ner != "": # keeping it going
				curTokens.append(each_token)
			lastSentenceNum = each_token.sentenceNum
			last_ner = cur_ner
		if len(curTokens) > 0: # had some left, let's make a mention
			toBeMentions.append(curTokens)
		
		for m in toBeMentions:
			doc_id = m[0].doc_id 
			dir_num = int(doc_id.split("_")[0])
			extension = doc_id[doc_id.find("ecb"):]
			dirHalf = str(dir_num) + extension
			text = []
			SUIDs = []
			#print("new mention")
			for t in m:
				#sentenceNum = t.sentenceNum
				#tokenNum = t.tokenNum
				SUID = t.UID
				SUIDs.append(SUID)
				#print("t.text:",t.text)
				#print("t.UID:",t.UID,"SUID:",SUID)
				#print("uidtotoken:",self.corpus.UIDToToken[SUID],"t:",t)
				if self.corpus.UIDToToken[SUID] != t:
					print("ERROR: Token mismatch")
					exit(1)
				text.append(t.text)

			curMention = Mention(dirHalf, dir_num, doc_id, m, text, False, tokenToNER[m[0]])
			self.corpus.addStanMention(curMention)
			for SUID in SUIDs:
				self.UIDToSUID[SUID].append(curMention.XUID)

		print("* Created",len(self.corpus.stan_mentions),"Stan Mentions")
	
	# creates HDDCRP Mentions
	def createHDDCRPMentions(self, hddcrp_mentions):

		tmpECBTokens = set()
		for m in self.corpus.ecb_mentions:
			if m.dir_num in self.testingDirs:
				for t in m.tokens:
					tmpECBTokens.add(t)

		tmpHDDCRPTokens = set()
		numMismatch = 0

		# these are correctly 'HUIDs' (i.e., text fields concatenated, not a XUID)
		for i in range(len(hddcrp_mentions)):
			HUIDs, ref_id, startTuple, endTuple = hddcrp_mentions[i]
			tokens = []
			text = []
			if len(HUIDs) == 0:
				print("ERROR: empty HDDCRP Mention")
				exit(1)

			for HUID in HUIDs:
				HUID_minus_text = HUID[0:HUID.rfind(";")]

				doc_id = HUID.split(";")[0]
				dir_num = int(doc_id.split("_")[0])
				extension = doc_id[doc_id.find("ecb"):]
				dirHalf = str(dir_num) + extension

				token = self.corpus.UIDToToken[HUID_minus_text]
				tokens.append(token)
				text.append(token.text)
				if HUID.split(";")[-1] != token.text:
					print("WARNING: TEXT MISMATCH: HUID:", HUID, "ECB token:", token)
					numMismatch += 1
			if self.args.onlyValidSentences and tokens[0].sentenceNum not in self.docToVerifiedSentences[tokens[0].doc_id]:
				continue
			else:
				for t in tokens:
					tmpHDDCRPTokens.add(t)
				curMention = Mention(dirHalf, dir_num, doc_id, tokens, text, True, "unknown")

				# sets the HDDCRP Mention's REF to being what was listed in the prediction file
				# this is done just so that we can later match it w/ the prediction file again, as there may be multiple
				# HMentions on the same line of the file, and we need a unique identifier
				curMention.setREF(ref_id)
				curMention.setStartTuple(startTuple)
				curMention.setEndTuple(endTuple)
				self.corpus.addHDDCRPMention(curMention)
				for HUID in HUIDs: # UID == HUID always, because we checked above (in the warning check)
					HUID_minus_text = HUID[0:HUID.rfind(";")]
					self.UIDToHMUID[HUID_minus_text].append(curMention.XUID)
		print("# ecb testing tokens (from mentions)", len(tmpECBTokens))
		print("# hddcrp testing tokens (from mentions):",len(tmpHDDCRPTokens))
		print("# HDDCRP Mentions created:",len(self.corpus.hddcrp_mentions))
		print("# numMismatch:", numMismatch)
		'''
		tp = 0
		fn = 0
		for t in tmpECBTokens:
			if t in tmpHDDCRPTokens:
				tp += 1
			else:
				fn += 1
		prec = float(tp / len(tmpHDDCRPTokens))
		recall = float(tp / len(tmpECBTokens))
		f1 = (2*prec*recall) / (prec + recall)
		print("p,r,f1:",prec,recall,f1)
		'''
	def printCorpusStats(self):
		mentionStats = defaultdict(int)
		numEntities = 0
		numEvents = 0
		for m in self.corpus.ecb_mentions:

			# logs if it's an entity or event
			if m.isPred:
				numEvents += 1
			else:
				numEntities += 1

			if m.dir_num in self.trainingDirs:
				mentionStats["train"] += 1
			elif m.dir_num in self.devDirs:
				mentionStats["dev"] += 1
			elif m.dir_num in self.testingDirs:
				mentionStats["test"] += 1
		print("mentionStats:", mentionStats)
	
		mentionStats = defaultdict(lambda: defaultdict(int))
		for m in self.corpus.hddcrp_mentions:
			for t in m.tokens:
				if t.UID in self.UIDToMention:
					menType = self.UIDToMention[t.UID].mentionType
					if m.dir_num in self.trainingDirs:
						mentionStats["train"][menType] += 1
					elif m.dir_num in self.devDirs:
						mentionStats["dev"][menType] += 1
					elif m.dir_num in self.testingDirs:
						mentionStats["test"][menType] += 1
					else:
						print("* ERROR: wrong dir")
						exit(1)
		
		print("[ CORPUS STATS ]")
		print("\t# dirHalves:", str(len(self.corpus.dirHalves)))
		print("\t# docs:", len(self.corpus.doc_idToDocs))
		print("\t# REFs:", len(self.corpus.refToEUIDs.keys()))
		numS = 0
		numC = 0
		for ref in self.corpus.refToEUIDs:
			if len(self.corpus.refToEUIDs[ref]) == 1:
				numS +=1
			else:
				numC += 1
		numTrain = 0
		numDev = 0
		numTest = 0
		trainDirs = set()
		devDirs = set()
		testDirs = set()
		for m in self.corpus.ecb_mentions:
			if m.dir_num in self.trainingDirs:
				numTrain += 1
				trainDirs.add(m.dir_num)
			elif m.dir_num in self.devDirs:
				numDev += 1
				devDirs.add(m.dir_num)
			elif m.dir_num in self.testingDirs:
				numTest += 1
				testDirs.add(m.dir_num)

		#print("\t# HDDCRP Mentions (test): NON-ACTION:", mentionStats["test"])
		print("\t# ECB Tokens:", len(self.corpus.corpusTokens))
		print("\t# ECB Mentions:", len(self.corpus.EUIDToMention))
		print("\t\t# entities:", str(numEntities), "# events:", str(numEvents))
		print("\t\ttrain (", numTrain, " mentions) dirs: ", sorted(trainDirs), sep="")
		print("\t\tdev (", numDev, " mentions) dirs: ", sorted(devDirs), sep="")
		print("\t\ttest (", numTest, " mentions) dirs: ", sorted(testDirs), sep="")
		print("\t# Stan Mentions:", len(self.corpus.SUIDToMention))
		print("\t# HDDCRP Mentions:", len(self.corpus.HMUIDToMention))
		print("\t== # Total Mentions:", len(self.corpus.XUIDToMention))

	# calculates the prec, recall, F1 of tokens
	# w.r.t. (a) Stan; (b) HDDCRP; (c) Stan+HDDCRP
	# across (1) events; (2) non-events; (3) all
	def printHDDCRPMentionCoverage(self):
		event_ecb_tokens = set()
		non_event_ecb_tokens = set()
		all_ecb_tokens = set()

		ecb_uids = set()
		stan_uids = set()

		for m in self.corpus.ecb_mentions:
			ecb_uids.add(m.UID)
			for t in m.tokens:
				dir_num = int(t.doc_id.split("_")[0])
				if dir_num not in self.testingDirs:
					continue
				if m.isPred:
					event_ecb_tokens.add(t)
				else:
					non_event_ecb_tokens.add(t)
				all_ecb_tokens.add(t)
		
		both_tokens = set() # stan + hddcrp

		# gathers stan tokens
		stan_tokens = set()
		for m in self.corpus.stan_mentions:
			ecb_uids.add(m.UID)
			for t in m.tokens:
				dir_num = int(t.doc_id.split("_")[0])
				if dir_num not in self.testingDirs:
					continue
				stan_tokens.add(t)
				both_tokens.add(t)

		# gathers HDDCRP tokens
		hddcrp_tokens = set()
		for hm in self.corpus.hddcrp_mentions:
			for t in hm.tokens:
				dir_num = int(t.doc_id.split("_")[0])
				if dir_num not in self.testingDirs:
					continue
				hddcrp_tokens.add(t)
				both_tokens.add(t)

		# collects mention-performance
		perfects = set()
		partials = set()
		falseNegatives = set()
		falsePositives = set()
		for m in self.corpus.ecb_mentions:
			if m.dir_num not in self.testingDirs or m.isPred:
				continue
			uid = m.UID
			if uid in stan_uids:
				perfects.add(m)
			else:
				# check if partial coverage or none
				isPartial = False
				for t in m.tokens:
					if t in stan_tokens:
						isPartial = True
						break
				if isPartial:
					partials.add(m)
				else:
					falseNegatives.add(m)
		for m in self.corpus.stan_mentions:
			if m.dir_num not in self.testingDirs:
				continue
			uid = m.UID
			if uid not in ecb_uids: # check if partial or none
				isPartial = False
				for t in m.tokens:
					if t in all_ecb_tokens:
						isPartial = True
						break
				if not isPartial: # no shared tokens; false positive
					falsePositives.add(m)

		print("# perfect ECB_MENTIONS:",len(perfects))
		print("# partials ECB_MENTIONS:", len(partials))
		print("# false Negatives (misses) ECB_MENTIONS:", len(falseNegatives))
		print("# false Positives ECB_MENTIONS:", len(falsePositives))
		self.printSet("perfects",perfects)
		self.printSet("partials", partials)
		self.printSet("falseNegatives", falseNegatives)
		self.printSet("falsePositives", falsePositives)
		self.printMentionCoverage("HDDCRP", hddcrp_tokens, event_ecb_tokens, non_event_ecb_tokens, all_ecb_tokens)
		self.printMentionCoverage("STAN", stan_tokens, event_ecb_tokens, non_event_ecb_tokens, all_ecb_tokens)
		self.printMentionCoverage("STAN+HDDCRP", both_tokens, event_ecb_tokens, non_event_ecb_tokens, all_ecb_tokens)

	def printSet(self, label, set):
		print(label,":")
		for i in set:
			print(i)

	def printMentionCoverage(self, label, our_tokens, event_ecb_tokens, non_event_ecb_tokens, all_ecb_tokens):
		# events
		numETP = 0
		numEFP = 0
		# non-events
		numNETP = 0
		numNEFP = 0
		# all
		numATP = 0
		numAFP = 0
		for t in our_tokens:
			# event
			if t in event_ecb_tokens:
				numETP += 1
			else:
				numEFP += 1
			# non-event
			if t in non_event_ecb_tokens:
				numNETP += 1
			else:
				numNEFP += 1
			# all
			if t in all_ecb_tokens:
				numATP += 1
			else:
				numAFP += 1

		event_prec = numETP / len(our_tokens)
		event_recall = numETP / len(event_ecb_tokens)
		event_f1 = 2*(event_prec * event_recall) / (event_prec + event_recall)
		
		non_event_prec = numNETP / len(our_tokens)
		non_event_recall = numNETP / len(non_event_ecb_tokens)
		non_event_f1 = 2*(non_event_prec * non_event_recall) / (non_event_prec + non_event_recall)
		
		all_prec = numATP / len(our_tokens)
		all_recall = numATP / len(all_ecb_tokens)
		all_f1 = 2*(all_prec * all_recall) / (all_prec + all_recall)
		print("** ",label,"MENTIONS **")
		print("[event] p:",event_prec,"r:",event_recall,"f1:",event_f1)
		print("[non-event] p:",non_event_prec,"r:",non_event_recall,"f1:",non_event_f1)
		print("[all] p:",all_prec,"r:",all_recall,"f1:",all_f1)
