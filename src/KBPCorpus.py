from collections import defaultdict
# Corpus is created first by the Source file, which creates
# Tokens and Docs (list of Tokens by lines)
# later, the event_files are read and Mentions are created, since Mentions
# are comprised of Tokens
class KBPCorpus:
	def __init__(self):
		# filled in from Source file
		self.corpusTokens = []
		self.corpusTokensToCorpusIndex = {}

		self.eventToSourceFile = {}
		self.sourceToEventFile = {}
		self.docs = []

		# mention stuff is filled in from Event file
		self.curKUID = 0  # KBPMention index counter
		self.kbp_mentions = []
		self.KUIDToKBPMention = {} 
		self.KUIDToREF = {}
		self.refToKUIDs = defaultdict(set)

	def addMention(self, mention):
		print("adding mention")

	def addDoc(self, doc):
		self.docs.append(doc)
