import operator
import sys
import pickle
from nltk.corpus import wordnet as wn
from Mention import Mention
from StanToken import StanToken
from collections import defaultdict
class FeatureHandler:
	def __init__(self, args, helper, mentions):
		self.args = args
		self.helper = helper
		self.corpus = helper.corpus
		self.mentions = mentions
	
	def saveBoWFeatures(self, fileOut):
		# i can use corpusTokensToCorpusIndex and corpusTokens to 
		# access each token by # in teh sliding window i care about
	def saveWordNetFeatures(self, fileOut):
		synSynToScore = {}
		muidPairs = self.getMUIDPairs()
		print("calculating wordnet features for",len(muidPairs),"unique pairs")
		i = 0
		uiduidToFeature = {} # pickled map of the score
		for m1, m2 in muidPairs:
			textTokens1 = self.corpus.XUIDToMention[m1].text
			textTokens2 = self.corpus.XUIDToMention[m2].text
			bestScore = -1
			for t1 in textTokens1:
				syn1 = wn.synsets(t1)
				if len(syn1) == 0:
					continue
				syn1 = syn1[0]
				for t2 in textTokens2:
					syn2 = wn.synsets(t2)
					if len(syn2) == 0:
						continue
					syn2 = syn2[0]
					curScore = -1
					if (syn1, syn2) in synSynToScore:
						curScore = synSynToScore[(syn1, syn2)]
					elif (syn2, syn1) in synSynToScore:
						curScore = synSynToScore[(syn2, syn1)]
					else: # calculate it
						curScore = wn.wup_similarity(syn1, syn2)
						synSynToScore[(syn1, syn2)] = curScore # don't want to store tons.  look-up is cheap
						if curScore != None and curScore > bestScore:
							bestScore = curScore
			uid1 = self.corpus.XUIDToMention[m1].UID
			uid2 = self.corpus.XUIDToMention[m2].UID
			li = sorted([uid1, uid2]) # sorts them lexicographically
			uiduidToFeature[(li[0], li[1])] = bestScore
			i += 1
			if i % 1000 == 0:
				print("\tprocessed", i, "of", len(muidPairs), "(%2.2f)" % float(100.0*i/len(muidPairs)), end="\r")
		
		pickle_out = open(fileOut, 'wb')
		pickle.dump(uiduidToFeature, pickle_out)
		print("")

	# create all pairs, for ecb, hddcrp, stan
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
