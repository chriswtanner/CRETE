from collections import defaultdict
class Mention:
	def __init__(self, dirHalf, dir_num, doc_id, tokens, text, isPred, mentionType):
		self.dirHalf = dirHalf
		self.dir_num = dir_num
		self.doc_id = doc_id
		self.tokens = tokens
		# both Tokens and Mentions keeps track of the globalSentenceNum
		# (so that we don't have to do messy lookup within the Token everytime)
		self.globalSentenceNum = tokens[0].globalSentenceNum 
		self.text = text
		self.isPred = isPred
		self.relativeTokenIndices = []
		self.suffix = doc_id[doc_id.find("ecb"):]
		self.mentionType = mentionType
		self.UID = "" # a unique concatenation of its Tokens' UIDs
		self.startTuple = ()
		self.endTuple = ()
		# gets filled in by Corpus.add*Mention()
		# robust to handle ECB or HDDCRP (if we named it MUID, then it
		# could look like it's only used for ECB Mentions, which isn't true)
		self.XUID = -1
		self.REF = ""

		#### TMP -- used for auxilary dependency info
		self.parentTokens = []
		self.parentEntities = []
		self.childrenTokens = []
		self.childrenEntities = []
		
		self.pathsToChildrenEntities = []
		self.entitiesLinked = set()
		self.levelToParentLinks = defaultdict(set)
		self.levelToChildrenEntities = defaultdict(set)

		self.levelToChildren = defaultdict(list) # NEW ONE, which stores tuples (mention, path)
		self.levelToParents = defaultdict(list) # NEW ONE, which stores tuples (mention, path)

		self.levelToEntityPath = defaultdict(list)
		self.parentRel = "None"
		self.childRel = "None"
		
		for t in self.tokens:
			self.UID += t.UID + ";"

	# only used for HDDCRP Mentions
	def setStartTuple(self, st):
		self.startTuple = st

	def setEndTuple(self, et):
		self.endTuple = et

	def setREF(self, REF):
		self.REF = REF

	def setXUID(self, XUID):
		self.XUID = XUID

	#### TMP -- used for auxilary dependency info
	def addParentLinks(self, levelToParentLinks):
		self.levelToParentLinks = levelToParentLinks
		if 1 in levelToParentLinks:
			for pl in levelToParentLinks[1]:
				self.parentRel = pl.relationship.lower()
				break

	def addEntityPath(self, level, path):
		relations = []
		for p in path:
			relations.append(p.relationship)
		self.levelToEntityPath[level].append(relations)

	def __str__(self):
		return "MENTION: " + str(self.XUID) + " (dir " + str(self.dir_num) + "; doc: " + str(self.doc_id) + "): text: " + str(self.text) + " type: " + str(self.mentionType)
