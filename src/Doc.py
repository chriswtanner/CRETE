from collections import defaultdict
class Doc:
    def __init__(self, name):
        self.name = name
        self.highestSentenceNum = -1
        self.globalSentenceNums = set() # unique #s across entire corpus
        self.tokens = []
        self.REFToEUIDs = defaultdict(set)
        self.EUIDs = set()
        self.SUIDs = set()
        self.HMUIDs = set()

    def assignEMention(self, EUID, REF):
        print("*assignEmention:",EUID,REF)
        self.REFToEUIDs[REF].add(EUID)
        self.EUIDs.add(EUID)

    def assignStanMention(self, SUID):
        self.SUIDs.add(SUID)

    def assignHDDCRPMention(self, HMUID):
        self.HMUIDs.add(HMUID)