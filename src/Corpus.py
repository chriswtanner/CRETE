
from collections import defaultdict
from Dir import Dir
class Corpus:
    def __init__(self):
        self.numCorpusTokens = 0
        self.corpusTokens = []
        self.corpusTokensToCorpusIndex = {}

        self.mentions = []
        self.dmToMention = {}  # (doc_id,m_id) -> Mention
        self.dmToREF = {}
        self.refToDMs = defaultdict(list)

        self.dirs = set()
        self.dirHalves = set() # same as what's contained across all dirs
        
        self.typeToGlobalID = {}
        self.globalIDsToType = {}
        self.corpusTypeIDs = []

        # to easily parse the original sentence which contains each Mention
        self.globalSentenceNumToTokens = defaultdict(list)
