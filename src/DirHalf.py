from collections import defaultdict
from Doc import Doc
class DirHalf:
    def __init__(self):
        self.docs = defaultdict(lambda: Doc)
        self.REFToMUIDs = defaultdict(set) # should be supset of all its docs
        self.MUIDs = set()
        #self.DMs = set() # NOTE: can this be changed to a Set()?

    def assignMUIDREF(self, MUID, doc_id, REF):
        # assigns DirHalf vars
        self.REFToMUIDs[REF].add(MUID)
        self.MUIDs.add(MUID)

        # assigns Doc vars
        self.docs[doc_id].assignMUIDREF(MUID, REF)
