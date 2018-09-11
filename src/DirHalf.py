from collections import defaultdict
from Doc import Doc
class DirHalf:
	def __init__(self):
		self.docs = defaultdict(lambda: Doc)

		# ECB Mentions
		self.REFToEUIDs = defaultdict(set) # should be superset of all its docs
		self.EUIDs = set()  # ECB parsed Mentions
		self.SUIDs = set()  # Stan Mentions
		self.HMUIDs = set()  # HDDCRP Mentions

	# sets the MUID and REF info
	def assignECBMention(self, EUID, doc_id, REF):
		# assigns DirHalf vars
		self.REFToEUIDs[REF].add(EUID)
		self.EUIDs.add(EUID)

		# assigns Doc vars
		self.docs[doc_id].assignEMention(EUID, REF)

	# sets the SUID info
	def assignStanMention(self, SUID, doc_id):
		self.SUIDs.add(SUID)  # assigns DirHalf vars
		self.docs[doc_id].assignStanMention(SUID)  # assigns Doc vars

	# sets the HMUID info
	def assignHDDCRPMention(self, HMUID, doc_id):
		self.HMUIDs.add(HMUID)  # assigns DirHalf vars
		self.docs[doc_id].assignHDDCRPMention(HMUID)  # assigns Doc vars

	def __str__(self):
		return "[dirHalf] # ECB Mentions:" + str(len(self.EUIDs)) + "; # SUIDs:" + str(len(self.SUIDs)) + "; # HMUIDs:" + str(len(self.HMUIDs))
