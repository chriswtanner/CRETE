import pickle
from collections import defaultdict
class Feature:
    def __init__(self):
        self.singles = {} # maps a UID to a feature vector
        self.relational = defaultdict(list) # maps a pair of UIDs to relational scalar

    def addSingle(self, uid, vector):
        self.singles[uid] = vector
    
    def addRelational(self, uid1, uid2, val):
        li = sorted([uid1, uid2])
        self.relational[(li[0],li[1])].append(val)