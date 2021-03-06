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

		# DON'T need these going from entities to events
		self.pathsToChildrenEntities = []
		self.pathsToParentEntities = []

		# WE DONT NEED TO SPECIFY EVENTS OR ENTITIES bc we're careful in constructing these
		# so that we're only linked to mentions of the opposite type
		self.childrenLinked = set() # STORES MENTIONS   A 
		self.parentsLinked = set() # STORES MENTIONS

		self.levelToChildrenMentions = defaultdict(set)   # B
		self.levelToParentMentions = defaultdict(set)

		self.levelToChildren = defaultdict(list) # NEW ONE, which stores tuples (mention, path) to mentions of the opposite type
		self.levelToParents = defaultdict(list) # NEW ONE, which stores tuples (mention, path) to mentions of hte opposite type

		self.parentRel = "None"
		self.childRel = "None"
		
		for t in self.tokens:
			self.UID += t.UID + ";"

	# only used for trying eugene's idea of the immediate hops
	def set_valid1hops(self, valid_hops, sentenceTokenToMention):
		self.valid_rel_to_entities = defaultdict(set)
		for rel in valid_hops:
			for token in valid_hops[rel]:
				if token in sentenceTokenToMention:
					foundMentions = sentenceTokenToMention[token]
					#print("\tfoundMentions:", foundMentions)
					for mfound in foundMentions:
						if not mfound.isPred: # mfound is an entity
							self.valid_rel_to_entities[rel].add(mfound)
		self.valid_hops = valid_hops

	# only used for HDDCRP Mentions
	def setStartTuple(self, st):
		self.startTuple = st

	def setEndTuple(self, et):
		self.endTuple = et

	def setREF(self, REF):
		self.REF = REF

	def setXUID(self, XUID):
		self.XUID = XUID

	def __str__(self):
		#return str(self.XUID) + ": " + str(self.text)
		return "MENTION: " + str(self.XUID) + " (dir " + str(self.dir_num) + "; doc: " + str(self.doc_id) + "; sent:" + str(self.globalSentenceNum) + "): text: " + str(self.text) + " type: " + str(self.mentionType)
