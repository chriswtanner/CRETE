from collections import defaultdict
from DirHalf import DirHalf
from ECBDir import ECBDir
class Corpus:
	def __init__(self):

		# NOTE: about the naming convention, XUID is the label
		# to address Mentions, whether they are ECB or HDDCRP
		# but when they are stored in data structures that are specifically/
		# exclusively intended for 1 or the other, we call them EUID, HMUID, SUID
		self.numCorpusTokens = 0
		self.corpusTokens = []
		self.corpusTokensToCorpusIndex = {}
		self.UIDToToken = {}
		self.curXUID = 0 # counter used for both ECB and HDDCRP, hence the X

		self.XUIDToMention = {} # used for *all* mentions, regardless if ECB, HDDCRP, Stan

		# only used for ECB Corpus mentions
		self.ecb_mentions = []
		self.EUIDToMention = {}
		self.EUIDToREF = {}
		self.refToEUIDs = defaultdict(set)
		# while self.refToEUIDs contains all refs (entities and events)
		# self.eventREFs keeps track of if the ref stores entities or events
		self.refToMentionTypes = defaultdict(set)
		self.docSentToEMentions = defaultdict(list)

		# only used for HDDCRP mentions (we have no REF info)
		self.hddcrp_mentions = []
		self.HMUIDToMention = {}
		self.docSentToHMentions = defaultdict(list)

		# only used for Stan mentions (we have no REF info)
		self.stan_mentions = []
		self.SUIDToMention = {}
		self.docSentToSMentions = defaultdict(list)

		self.ECBDirs = defaultdict(ECBDir) # a full DIR, aka encompasses both dirHalves
		self.dirHalves = defaultdict(DirHalf) # half a DIR

		# Docs are stored within dirHalves,
		# but also accessible from here, by its name
		self.doc_idToDocs = {}

		# to easily parse the original sentence which contains each Mention
		self.globalSentenceNumToTokens = defaultdict(list)

	# TMP: just displays how many times each event-pair is co-ref or not, w.r.t.
	# belonging to sentences which contain co-ref entities
	def calculateEntEnvAgreement(self):
		withinDocPairs = defaultdict(lambda: defaultdict(int))
		crossDocPairs = defaultdict(lambda: defaultdict(int))
		withinDocParentLinks = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
		withinDocChildrenLinks = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

		# keeps track of sentence -> entities
		sentenceNumToEntities = defaultdict(set)
		for euid in self.EUIDToMention:
			m = self.EUIDToMention[euid]
			if not m.isPred:
				sentenceNumToEntities[m.globalSentenceNum].add(m)

		# iterates over all event mention pairs
		for dh in sorted(self.dirHalves):
			#dprint("dh:",dh)
			for euid1 in sorted(self.dirHalves[dh].EUIDs):
				m1 = self.EUIDToMention[euid1]
				if not m1.isPred:
					continue
				print("m1:", m1)
				sentence1 = m1.globalSentenceNum
				entities1 = sentenceNumToEntities[sentence1]
				for euid2 in sorted(self.dirHalves[dh].EUIDs):
					if euid1 >= euid2:
						continue

					m2 = self.EUIDToMention[euid2]
					if not m2.isPred:
						continue

					# at this point, we have an m1 and m2, both of which are events
					print("m2:", m2)
					
					sameParentLink = False
					m1_parentDep = ""
					m2_parentDep = ""
					if 1 in m.levelToParentLinks.keys():
						print("levelToParentLinks[1]:", m.levelToParentLinks[1])


					# checks if our entities coref

					sentence2 = m2.globalSentenceNum
					entities2 = sentenceNumToEntities[sentence2]

					eventCoref = "eventCoref_no"
					if self.EUIDToREF[euid1] == self.EUIDToREF[euid2]:
						eventCoref = "eventCoref_yes"

					entityCoref = "entCoref_no"
					for ent1 in entities1:
						REF1 = ent1.REF
						for ent2 in entities2:
							if ent1 == ent2:
								continue
							if ent2.REF == REF1:
								entityCoref = "entCoref_yes"
								break
					
					isSameDoc = False
					if self.EUIDToMention[euid1].doc_id == self.EUIDToMention[euid2].doc_id:
						isSameDoc = True
					
					if isSameDoc:
						withinDocPairs[eventCoref][entityCoref] += 1
						#withinDocParentLinks[eventCoref][entityCoref][]
					else:
						crossDocPairs[eventCoref][entityCoref] += 1
		print("withinDocPairs:")
		for _ in withinDocPairs:
			print(_, withinDocPairs[_])
		
		print("crossDocPairs:")
		for _ in crossDocPairs:
			print(_, crossDocPairs[_])

	# ensures we've created ECB/HDDCRP/Stan Mentions all from the same sentences
	def checkMentions(self):
		allKeys = set()
		allKeys.update(self.docSentToEMentions.keys())
		allKeys.update(self.docSentToHMentions.keys())
		allKeys.update(self.docSentToSMentions.keys())
		print("# keys:",len(allKeys))
		docToSent = defaultdict(set)
		for doc, _ in allKeys:
			docToSent[doc].add(_)
		for doc in docToSent.keys():
			#print(doc)
			for sent in docToSent[doc]:
				necb = len(self.docSentToEMentions[(doc, sent)])
				nh = len(self.docSentToHMentions[(doc, sent)])
				ns = len(self.docSentToSMentions[(doc, sent)])
				if necb == 0 and ns > 0:
					print("doc:",doc, "sent:",sent)
					print(necb,",",nh,",",ns)

	# adds a Token to the corpus
	def addToken(self, token):
		self.corpusTokens.append(token)
		self.corpusTokensToCorpusIndex[token] = self.numCorpusTokens
		self.numCorpusTokens = self.numCorpusTokens + 1

	# adds a Stan Mention to the corpus (no REF info)
	def addStanMention(self, mention):
		(doc_id, sentenceNum) = self.getDocAndSentence(mention)
		mention.setXUID(self.curXUID)  # updates the mention w/ XUID info
		self.stan_mentions.append(mention)
		self.SUIDToMention[self.curXUID] = mention
		#self.dirs[mention.dir_num].assignStanMention(self.curXUID, mention.doc_id)
		self.dirHalves[mention.dirHalf].assignStanMention(self.curXUID, mention.doc_id)
		self.docSentToSMentions[(doc_id, sentenceNum)].append(mention)
		self.XUIDToMention[self.curXUID] = mention
		self.curXUID += 1

	# adds a HDDCRP Mention to the corpus (no REF info)
	def addHDDCRPMention(self, mention):
		(doc_id, sentenceNum) = self.getDocAndSentence(mention)
		mention.setXUID(self.curXUID)  # updates the mention w/ XUID info
		self.hddcrp_mentions.append(mention)
		self.HMUIDToMention[self.curXUID] = mention
		self.dirHalves[mention.dirHalf].assignHDDCRPMention(self.curXUID, mention.doc_id)
		self.ECBDirs[mention.dir_num].assignHDDCRPMention(self.curXUID, mention.doc_id)
		self.docSentToHMentions[(doc_id, sentenceNum)].append(mention)
		self.XUIDToMention[self.curXUID] = mention
		self.curXUID += 1

	# adds a Mention to the corpus
	def addMention(self, mention, REF):

		# TMP -- just for annotation viewing
		# (we iterate through all tokens this mention spans, and point those tokens to this mention)
		for t in mention.tokens:
			t.addMentionSpan(mention)

		(doc_id, sentenceNum) = self.getDocAndSentence(mention)

		# updates the mention w/ REF and XUID info
		mention.setXUID(self.curXUID)
		mention.setREF(REF)

		self.ecb_mentions.append(mention)
		self.EUIDToMention[self.curXUID] = mention
		self.EUIDToREF[self.curXUID] = REF
		self.refToEUIDs[REF].add(self.curXUID)
		if mention.isPred:
			self.refToMentionTypes[REF].add("event")
		else:
			self.refToMentionTypes[REF].add("entity")

		self.dirHalves[mention.dirHalf].assignECBMention(self.curXUID, mention.doc_id, REF)
		self.ECBDirs[mention.dir_num].assignECBMention(self.curXUID, mention.doc_id, REF)
		self.docSentToEMentions[(doc_id, sentenceNum)].append(mention)
		self.XUIDToMention[self.curXUID] = mention
		self.curXUID += 1

	def assignGlobalSentenceNums(self):
		for t in self.corpusTokens:
			self.globalSentenceNumToTokens[int(t.globalSentenceNum)].append(t)

	# adds access to the Docs via their names
	def addDocPointer(self, doc_id, curDoc):
		self.doc_idToDocs[doc_id] = curDoc

	def getDocAndSentence(self, mention):
		doc_id = mention.doc_id
		sentenceNum = mention.tokens[0].sentenceNum
		return (doc_id, sentenceNum)
