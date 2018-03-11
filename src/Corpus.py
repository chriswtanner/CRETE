
from collections import defaultdict
from DirHalf import DirHalf
class Corpus:
    def __init__(self):
        self.numCorpusTokens = 0
        self.corpusTokens = []
        self.corpusTokensToCorpusIndex = {}
        self.mentions = []
        self.curMUID = 0
        self.MUIDToMention = {}
        self.MUIDToREF = {}
        self.refToMUIDs = defaultdict(set)

        self.dirs = set()
        self.dirHalves = defaultdict(DirHalf) # same as what's contained across all dirs
        
        self.UIDToToken = {}

        # to easily parse the original sentence which contains each Mention
        self.globalSentenceNumToTokens = defaultdict(list)

    # adds a Token to the corpus
    def addToken(self, token):
        self.corpusTokens.append(token)
        self.corpusTokensToCorpusIndex[token] = self.numCorpusTokens
        self.numCorpusTokens = self.numCorpusTokens + 1

    # adds a Mention to the corpus
    def addMention(self, mention, REF):

        # updates the mention w/ REF and muid info
        mention.setMUID(self.curMUID)
        mention.setREF(REF)

        self.mentions.append(mention)
        self.MUIDToMention[self.curMUID] = mention
        self.MUIDToREF[self.curMUID] = REF
        self.refToMUIDs[REF].add(self.curMUID)
        self.dirHalves[mention.dirHalf].assignMUIDREF(self.curMUID, mention.doc_id, REF)
        self.curMUID += 1

    def assignGlobalSentenceNums(self):
        for t in self.corpusTokens:
            self.globalSentenceNumToTokens[int(t.globalSentenceNum)].append(t)

    def printStats(self):
        print("[ CORPUS STATS ]")
        print("\t# dirHalves:",str(len(self.dirHalves)))
        print("\t# docs:",len([doc for dh in self.dirHalves for doc in self.dirHalves[dh].docs]))
        print("\t# REFs:", len(self.refToMUIDs.keys()))
        print("\t# Mentions:", len(self.MUIDToREF.keys()))
        '''
        for REF in self.refToMUIDs:
            print("REF:",REF,"has # mentions:",len(self.refToMUIDs[REF]))
            for m in self.refToMUIDs[REF]:
                print(self.MUIDToMention[m])
        '''