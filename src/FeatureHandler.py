import operator
import sys
from Mention import Mention
from StanToken import StanToken
from collections import defaultdict
class FeatureHandler:
	def __init__(self, args, helper, mentions):
		self.args = args
		self.helper = helper
		self.corpus = helper.corpus
		self.mentions = mentions
	
	def createWordNetFeatures(self):
		muidPairs = self.getMUIDPairs()
		print(len(muidPairs))
	def getMUIDPairs(self):
		ret = set()
		for dirhalf in self.corpus.dirHalves:
			for m1 in self.corpus.dirHalves[dirhalf].MUIDs:
				for m2 in self.corpus.dirHalves[dirhalf].MUIDs:
					if m2 <= m1:
						continue
					ret.add((m1,m2))
			for m1 in self.corpus.dirHalves[dirhalf].HMUIDs:
				for m2 in self.corpus.dirHalves[dirhalf].HMUIDs:
					if m2 <= m1:
						continue
					ret.add((m1, m2))
			for m1 in self.corpus.dirHalves[dirhalf].SUIDs:
				for m2 in self.corpus.dirHalves[dirhalf].SUIDs:
					if m2 <= m1:
						continue
					ret.add((m1, m2))
		return ret