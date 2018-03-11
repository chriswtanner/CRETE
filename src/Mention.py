class Mention:
    def __init__(self, dirHalf, dir_num, doc_id, tokens, text, isPred, mentionType):
        self.dirHalf = dirHalf
        self.dir_num = dir_num
        self.doc_id = doc_id
        self.tokens = tokens
        self.text = text
        self.isPred = isPred
        self.relativeTokenIndices = []
        self.suffix = doc_id[doc_id.find("ecb"):]
        self.mentionType = mentionType
        self.UID = "" # a unique concatenation of its Tokens' UIDs
        
        # gets filled in by Corpus.addMention()
        self.MUID = -1
        self.REF = ""

        for t in self.tokens:
            self.UID += t.UID + ";"

    def setREF(self, REF):
        self.REF = REF

    def setMUID(self, MUID):
        self.MUID = MUID

    #def castAsHDDCRPMention(self)

    def __str__(self):
        return "MENTION: " + str(self.MUID) + " (dir " + str(self.dir_num) + "; doc: " + str(self.doc_id) + "): text: " + str(self.text) + " corpusIndices: " + str(self.corpusTokenIndices)

