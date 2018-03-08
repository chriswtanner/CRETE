
from collections import defaultdict
from DirHalf import DirHalf
class Corpus:
    def __init__(self):
        self.numCorpusTokens = 0
        self.corpusTokens = []
        self.corpusTokensToCorpusIndex = {}

        self.mentions = []
        self.dmToMention = {}  # (doc_id,m_id) -> Mention
        self.dmToREF = {}
        self.refToDMs = defaultdict(set)

        self.dirs = set()
        self.dirHalves = defaultdict(DirHalf) # same as what's contained across all dirs
        
        self.UIDToMentions = {}
        self.UIDToToken = {}
        self.typeToGlobalID = {}
        self.globalIDsToType = {}
        self.corpusTypeIDs = []

        # to easily parse the original sentence which contains each Mention
        self.globalSentenceNumToTokens = defaultdict(list)

    # adds a Token to the corpus
    def addToken(self, token):
        self.corpusTokens.append(token)
        self.corpusTokensToCorpusIndex[token] = self.numCorpusTokens
        self.numCorpusTokens = self.numCorpusTokens + 1

    # adds a Mention to the corpus
    def addMention(self, mention):
        dm = (mention.doc_id, mention.m_id)
        self.mentions.append(mention)
        self.dmToMention[dm] = mention

    def assignDMREF(self, dm, dirHalf, doc_id, ref_id):
        self.dmToREF[dm] = ref_id
        self.refToDMs[ref_id].append(dm)
        self.dmToMention[dm].REF = ref_id

        self.dirHalves[dirHalf].assignDMREF(dm, doc_id, ref_id)
