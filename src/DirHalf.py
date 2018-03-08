from collections import defaultdict
from Doc import Doc
class DirHalf:
    def __init__(self): #, name):
        #self.name = name
        self.REFs = set()
        self.docs = defaultdict(lambda : Doc)
        self.REFToDMs = defaultdict(set) # should be supset of all its docs
        self.mentions = [] # NOTE: can this be changed to a Set()?
