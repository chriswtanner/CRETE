import pickle
class Token:
	def __init__(self, tokenID, sentenceNum, globalSentenceNum, tokenNum, doc_id, hSentenceNum, hTokenNum, text):

		# TMP -- so that we can easily the corpus w/ annotations
		self.mentions = set()

		# NOTE, this is relative to the given sentence <start> = 1  the = 2 (1-based)
		self.tokenNum = tokenNum
		self.hSentenceNum = hSentenceNum
		self.hTokenNum = hTokenNum
		self.doc_id = doc_id

		# list of the stan tokens that map to this (USUALLY this is just 1 stan token, though)
		self.stanTokens = []
		self.tokenID = tokenID  # given in the XML
		self.sentenceNum = sentenceNum
		self.globalSentenceNum = globalSentenceNum
		self.text = text

		self.UID = str(self.doc_id) + ";" + str(self.hSentenceNum) + ";" + str(self.hTokenNum)

	def addStanTokens(self, stanTokens):
		self.stanTokens = stanTokens

	# only used if ECBParser.printCorpusTokens = True (which prints out our Mentions in a readable format)
	# if we don't care about printing those things, then this can be removed,
	# as i added this code just to for that function
	def addMentionSpan(self, m):
		self.mentions.add(m)

	def __str__(self):
		return("TOKEN: ID:" + str(self.tokenID) + "; TOKEN#: " + str(self.tokenNum) + "; SENTENCE#:" + str(self.sentenceNum) + " globalSentenceNum: " + str(self.globalSentenceNum) + "; TEXT:" + str(self.text))
