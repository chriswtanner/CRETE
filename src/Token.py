class Token:
    def __init__(self, tokenID, sentenceNum, globalSentenceNum, tokenNum, doc_id, hSentenceNum, hTokenNum, text, tokens=[]):
    
        self.tokens = tokens

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

        self.UID = str(self.doc_id) + ";" + str(self.hSentenceNum) + \
            ";" + str(self.hTokenNum)

    def addStanTokens(self, stanTokens):
        self.stanTokens = stanTokens

    def __str__(self):
        return("TOKEN: ID:" + str(self.tokenID) + "; TOKEN#: " + str(self.tokenNum) + "; SENTENCE#:" + str(self.sentenceNum) + " globalSentenceNum: " + str(self.globalSentenceNum) + "; TEXT:" + str(self.text))
