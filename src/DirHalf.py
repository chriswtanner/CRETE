from collections import defaultdict
from Doc import Doc
class DirHalf:
    def __init__(self):
        self.docs = defaultdict(lambda: Doc)

        # ECB Mentions
        self.REFToMUIDs = defaultdict(set) # should be superset of all its docs
        self.MUIDs = set()

        self.SUIDs = set()  # Stan Mentions
        self.HMUIDs = set()  # HDDCRP Mentions

    # sets the MUID and REF info
    def assignECBMention(self, MUID, doc_id, REF):
        # assigns DirHalf vars
        self.REFToMUIDs[REF].add(MUID)
        self.MUIDs.add(MUID)

        # assigns Doc vars
        self.docs[doc_id].assignECBMention(MUID, REF)

    # sets the SUID info
    def assignStanMention(self, SUID, doc_id):
        self.SUIDs.add(SUID)  # assigns DirHalf vars
        self.docs[doc_id].assignStanMention(SUID)  # assigns Doc vars

    # sets the HMUID info
    def assignHDDCRPMention(self, HMUID, doc_id):
        self.HMUIDs.add(HMUID)  # assigns DirHalf vars
        self.docs[doc_id].assignHDDCRPMention(HMUID)  # assigns Doc vars
