from collections import defaultdict
from Doc import Doc
class DirHalf:
    def __init__(self):
        self.docs = defaultdict(lambda: Doc)
        self.REFToDMs = defaultdict(set) # should be supset of all its docs
        self.DMs = set() # NOTE: can this be changed to a Set()?

    def assignDMREF(self, dm, doc_id, ref_id):
        # assigns DirHalf vars
        self.REFToDMs[ref_id].add(dm)
        self.DMs.add(dm)

        # assigns Doc vars
        self.docs[doc_id].assignDMREF(dm, ref_id)
