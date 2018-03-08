
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
        self.refToDMs = defaultdict(list)

        self.dirs = set()
        self.dirHalves = defaultdict(DirHalf) # same as what's contained across all dirs
        
        self.UIDToMentions = {}
        self.UIDToToken = {}
        self.typeToGlobalID = {}
        self.globalIDsToType = {}
        self.corpusTypeIDs = []

        # to easily parse the original sentence which contains each Mention
        self.globalSentenceNumToTokens = defaultdict(list)
