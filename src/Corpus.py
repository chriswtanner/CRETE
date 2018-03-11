
from collections import defaultdict
from DirHalf import DirHalf
class Corpus:
    def __init__(self):

        # NOTE: about the naming convention, XUID is the int label
        # to address Mentions, whether they are ECB or HDDCRP
        # but when they are stored in data structures that are specifically/
        # exclusively intended for 1 or the other, we call them MUID or HMUID
        self.numCorpusTokens = 0
        self.corpusTokens = []
        self.corpusTokensToCorpusIndex = {}
        self.UIDToToken = {}
        self.curXUID = 0 # counter used for both ECB and HDDCRP, hence the X

        # only used for ECB Corpus mentions
        self.ecb_mentions = []
        self.MUIDToMention = {}
        self.MUIDToREF = {}
        self.refToMUIDs = defaultdict(set)

        # only used for HDDCRP mentions (we have no REF info)
        self.hddcrp_mentions = []
        self.HMUIDToMention = {}

        self.dirs = set()
        self.dirHalves = defaultdict(DirHalf) # same as what's contained across all dirs
        
        # to easily parse the original sentence which contains each Mention
        self.globalSentenceNumToTokens = defaultdict(list)

    # adds a Token to the corpus
    def addToken(self, token):
        self.corpusTokens.append(token)
        self.corpusTokensToCorpusIndex[token] = self.numCorpusTokens
        self.numCorpusTokens = self.numCorpusTokens + 1

    # adds a HDDCRP Mention to the corpus (no REF info)
    def addHDDCRPMention(self, mention):
        # updates the mention w/ MUID info
        mention.setXUID(self.curXUID)

        self.hddcrp_mentions.append(mention)
        self.HMUIDToMention[self.curXUID] = mention
        self.dirHalves[mention.dirHalf].assignHDDCRPMention(self.curXUID, mention.doc_id)

        self.curXUID += 1

    # adds a Mention to the corpus
    def addMention(self, mention, REF):
        # updates the mention w/ REF and MUID info
        mention.setXUID(self.curXUID)
        mention.setREF(REF)

        self.ecb_mentions.append(mention)
        self.MUIDToMention[self.curXUID] = mention
        self.MUIDToREF[self.curXUID] = REF
        self.refToMUIDs[REF].add(self.curXUID)
        self.dirHalves[mention.dirHalf].assignECBMention(self.curXUID, mention.doc_id, REF)
        self.curXUID += 1

    def assignGlobalSentenceNums(self):
        for t in self.corpusTokens:
            self.globalSentenceNumToTokens[int(t.globalSentenceNum)].append(t)

    def printStats(self):
        print("[ CORPUS STATS ]")
        print("\t# dirHalves:",str(len(self.dirHalves)))
        print("\t# docs:",len([doc for dh in self.dirHalves for doc in self.dirHalves[dh].docs]))
        print("\t# REFs:", len(self.refToMUIDs.keys()))
        print("\t# ECB Mentions:", len(self.ecb_mentions))
        '''
        for REF in self.refToMUIDs:
            print("REF:",REF,"has # mentions:",len(self.refToMUIDs[REF]))
            for m in self.refToMUIDs[REF]:
                print(self.MUIDToMention[m])
        '''
