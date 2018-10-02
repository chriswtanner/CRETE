import pickle
import operator
import sys
import copy
from Mention import Mention
from StanDB import StanDB
from StanToken import StanToken
from collections import defaultdict
class ECBHelper:
	def __init__(self, args):
		self.args = args
		self.corpus = None # should be passed-in

		# data splits
		self.trainingDirs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 19, 20, 21, 22]
		self.devDirs = [23, 24, 25]
		self.testingDirs = [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]

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


	# given a regular ECB-token, find its ecb-governor tokens
	# just, we have to use stan tokens as the intermediatary, since that's
	# where our dependency information comes from.  this gets tricky because
	# there's sometimes a non-1-to-1 mapping from ecb-token to stan-token
	def getParents(self, mentionStans, dh, token, depth):
		if token not in self.tokensVisited:
			bestStan = dh.getBestStanToken(token.stanTokens)
			prefix = "\t"
			for i in range(depth):
				prefix += "\t"
			#print(prefix, token, "; bestStan:", bestStan)
			
			# grab its parent(s)
			#print(prefix,"# parentLinks:",len(bestStan.parentLinks))
			self.tokensVisited.add(token)
			for p in bestStan.parentLinks:
				#print(prefix,"\tstanparent:", p.parent)
				
				# only explore the parent if it's not part of the original Mention!
				if p.parent not in mentionStans:
					ecbTokens = self.stanTokenToECBTokens[p.parent]
					self.levelToParentLinks[depth].add(p)

					#print(prefix,"\tecbTokens:", ecbTokens)
					if len(ecbTokens) > 0:
						ecbParentToken = next(iter(self.stanTokenToECBTokens[p.parent]))
						self.levelToParents[depth].add(ecbParentToken)
						self.getParents(mentionStans, dh, ecbParentToken, depth+1)

	def getChildren(self, mentionStans, dh, token, depth):
		if token not in self.tokensVisited:
			bestStan = dh.getBestStanToken(token.stanTokens)
			prefix = "\t"
			for i in range(depth):
				prefix += "\t"
			#print(prefix, token, "; bestStan:", bestStan)
			
			# grab its parent(s)
			#print(prefix,"# parentLinks:",len(bestStan.parentLinks))
			self.tokensVisited.add(token)
			for p in bestStan.childLinks:
				#print(prefix,"\tstanparent:", p.parent)

				if p.child not in mentionStans:
					ecbTokens = self.stanTokenToECBTokens[p.child]
					self.levelToChildrenLinks[depth].add(p)

					#print(prefix,"\tecbTokens:", ecbTokens)
					if len(ecbTokens) > 0:
						ecbChildToken = next(iter(self.stanTokenToECBTokens[p.child]))
						self.levelToChildren[depth].add(ecbChildToken)
						self.getChildren(mentionStans, dh, ecbChildToken, depth+1)

	def addDependenciesToMentions(self, dh):
		for doc_id in self.corpus.doc_idToDocs:
			#if not doc_id.startswith("1_") or "plus" in doc_id:
			#if doc_id != "1_10ecb.xml":
			#	continue
			#print("[DOC: ", doc_id)
			# maps ECB Token -> StanToken and vice versa
			self.stanTokenToECBTokens = defaultdict(set)
			curdoctokens = ""
			for t in self.corpus.doc_idToDocs[doc_id].tokens:
				curdoctokens += t.text + " "
				for s in t.stanTokens:
					self.stanTokenToECBTokens[s].add(t)
					#print("t:",t.text,"s:",s.text)
			#print("curdoctokens:", curdoctokens)
			for k in self.stanTokenToECBTokens:
				if len(self.stanTokenToECBTokens[k]) > 1:
					print("woops, we have len of :", len(self.stanTokenToECBTokens[k]), ":", k)
					#exit(1)

			# looks through each mention, to print the most immediate governor
			# and modifier mentions of the opposite type
			# maps each mention to a SENTENCE
			sentenceToEventMentions = defaultdict(set)
			sentenceToEntityMentions = defaultdict(set)
			sentenceTokenToMention = defaultdict(lambda: defaultdict(set))
			for euid in self.corpus.doc_idToDocs[doc_id].EUIDs:
				m = self.corpus.EUIDToMention[euid]
				sentNum = m.globalSentenceNum

				for t in m.tokens:
					sentenceTokenToMention[sentNum][t].add(m)

				if m.isPred:
					sentenceToEventMentions[sentNum].add(m)
				else:
					sentenceToEntityMentions[sentNum].add(m)

			for s in sentenceToEventMentions:
				#print("sentence #:", s)
				#print("\tevents:", sentenceToEventMentions)
				for m in sentenceToEventMentions[s]:
					print("\t", m)

					# gets the StanTokens for the current mention, so that we \
					# never explore any of them as parents or chilren
					mentionStanTokens = set()
					for t in m.tokens:
						bestStan = dh.getBestStanToken(t.stanTokens)
						mentionStanTokens.add(bestStan)

					# finds its parents (governors)
					self.levelToParents = defaultdict(set)
					self.levelToParentLinks = defaultdict(set)

					self.tokensVisited = set()
					for t in m.tokens:
						#print("\t\ttoken:", t)
						self.getParents(mentionStanTokens, dh, t, 1)
					print("\tmention yielded following governor structure:", self.levelToParentLinks)
					
					m.addParentLinks(self.levelToParentLinks)
					print("m:")
					for level in m.levelToParentLinks:
						print("level:", level)
						for pl in m.levelToParentLinks[level]:
							print("\t", str(pl))
					if len(self.levelToParents) > 0:
						#print("\n\tgovernors:")
						for level in sorted(self.levelToParents):
							#print("\t\tlevel:", level)
							for t in self.levelToParents[level]:
								#print("\t\t\t", t)

								# adds dependency parent tokens
								if t not in m.parentTokens:
									m.parentTokens.append(t)
								for parent_mention in sentenceTokenToMention[s][t]:
									#print("\t\t\t\tparent mention:", parent_mention)
									if not parent_mention.isPred:
										if parent_mention not in m.parentEntities:
											m.parentEntities.append(parent_mention)
										#print("\t\t\t\t****** WE HAVE AN ENTITY MENTION!!! w/ ref:", parent_mention.REF)

					# finds its children (modifiers)
					self.levelToChildren = defaultdict(set)
					self.levelToChildrenLinks = defaultdict(set)
					self.tokensVisited = set()
					for t in m.tokens:
						#print("\t\ttoken:", t)
						self.getChildren(mentionStanTokens, dh, t, 1)
					#print("\tmention yielded following governor structure:", self.levelToParents)
					
					m.addChildrenLinks(copy.deepcopy(self.levelToChildrenLinks))

					if len(self.levelToChildren) > 0:
						#print("\n\tdependents:")
						for level in sorted(self.levelToChildren):
							#print("\t\tlevel:", level)
							for t in self.levelToChildren[level]:
								#print("\t\t\t", t)

								# adds dependency parent tokens
								if t not in m.childrenTokens:
									m.childrenTokens.append(t)
								for child_mention in sentenceTokenToMention[s][t]:
									#print("\t\t\tchild mention:", child_mention)
									if not child_mention.isPred:
										if child_mention not in m.childrenEntities:
											m.childrenEntities.append(child_mention)
										#print("\t\t\t\t****** WE HAVE AN ENTITY MENTION!!! w/ ref:", child_mention.REF)


				'''
				print("\tentities:")
				for m in sentenceToEntityMentions[s]:
					print("\t", m)
				'''
			print("done w/ current doc:", str(doc_id))
			exit(1)
			# prints tokens and their dependencies
			'''
			for t in self.corpus.doc_idToDocs[doc_id].tokens:
				print("\n" + str(t))
				bestStan = dh.getBestStanToken(t.stanTokens)
				print("bestStan:", bestStan)
			'''
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
		for m in self.corpus.ecb_mentions:
			if m.dir_num in self.trainingDirs:
				mentionStats["train"] += 1
			elif m.dir_num in self.devDirs:
				mentionStats["dev"] += 1
			elif m.dir_num in self.testingDirs:
				mentionStats["test"] += 1
		print(mentionStats)
	
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
