from collections import defaultdict
class Doc:
    def __init__(self, name):
        self.name = name
        self.highestSentenceNum = -1
        self.globalSentenceNums = set() # unique #s across entire corpus
        self.tokens = []
        self.REFToDMs = defaultdict(set)
        self.DMs = [] # NOTE: can this be changed to a Set()?
        self.UIDs = []
