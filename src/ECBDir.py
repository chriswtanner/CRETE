from collections import defaultdict
from Doc import Doc
class ECBDir:
	def __init__(self):
		self.docs = defaultdict(lambda: Doc)

		# ECB Mentions
		self.REFToEUIDs = defaultdict(set) # should be superset of all its docs
		self.EUIDs = set()  # ECB parsed Mentions
		self.SUIDs = set()  # Stan Mentions
		self.HUIDs = set()  # HDDCRP Mentions

	# sets the EUID and REF info
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
	def assignHDDCRPMention(self, HUID, doc_id):
		self.HUIDs.add(HUID)  # assigns DirHalf vars
		self.docs[doc_id].assignHDDCRPMention(HUID)  # assigns Doc vars

	def __str__(self):
		return "[ECBDir] # ECB Mentions:" + str(len(self.EUIDs)) + "; # SUIDs:" + str(len(self.SUIDs)) + "; # HUIDs:" + str(len(self.HUIDs))
