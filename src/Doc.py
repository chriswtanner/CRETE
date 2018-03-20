from collections import defaultdict
class Doc:
    def __init__(self, name):
        self.name = name
        self.highestSentenceNum = -1
        self.globalSentenceNums = set() # unique #s across entire corpus
        self.tokens = []
        self.REFToMUIDs = defaultdict(set)
        self.MUIDs = set()
        self.SUIDs = set()
        self.HMUIDs = set()

    def assignECBMention(self, MUID, REF):
        self.REFToMUIDs[REF].add(MUID)
        self.MUIDs.add(MUID)

    def assignStanMention(self, SUID):
        self.SUIDs.add(SUID)

    def assignHDDCRPMention(self, HMUID):
        self.HMUIDs.add(HMUID)
