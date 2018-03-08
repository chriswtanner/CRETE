class Mention:
    def __init__(self, dirNum, doc_id, m_id, tokens, corpusTokenIndices, text, isPred, mentionType):
        self.dirNum = dirNum
        self.doc_id = doc_id
        self.m_id = m_id
        self.tokens = tokens
        self.corpusTokenIndices = corpusTokenIndices
        self.text = text
        self.isPred = isPred
        self.relativeTokenIndices = []
        self.suffix = doc_id[doc_id.find("ecb"):]
        self.mentionType = mentionType
        self.UID = ""
        self.REF = "" # gets updated via Corpus.assignDMREF()
        for t in self.tokens:
            self.UID += t.UID + ";"

    def __str__(self):
        return "MENTION: " + str(self.m_id) + " (dir " + str(self.dirNum) + "; doc: " + str(self.doc_id) + "): text: " + str(self.text) + " corpusIndices: " + str(self.corpusTokenIndices)

