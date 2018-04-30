import operator
import sys
import pickle
from math import sqrt
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
		self.bowWindow = 3 # number of tokens on each side to look at
		self.gloveEmb = {} # to be filled in via loadGloveEmbeddings()
		self.badPOS = ["‘’", "``", "POS", "$", "''"]
	############################
	###   HELPER FUNCTIONS   ###
	############################
	def loadGloveEmbeddings(self):
		f = open(self.args.embeddingsFile, 'r', encoding="utf-8")
		for line in f:
			tokens = line.rstrip().split(" ")
			word = tokens[0]
			emb = [float(x) for x in tokens[1:]]
			self.gloveEmb[word] = emb
		f.close()

	# removes the leading and trailing quotes, if they exist
	def removeQuotes(self, token):
		if len(token) > 0:
			if token == "''" or token == "\"":
				return "\""
			elif token == "'" or token == "'s":
				return token
			else:  # there's more substance to it, not a lone quote
				if token[0] == "'" or token[0] == "\"":
					token = token[1:]
				if len(token) > 0:
					if token[-1] == "'" or token[-1] == "\"":
						token = token[0:-1]
				return token
		else:
			print("* found a blank")
			exit(1)
			return ""

	def getBestStanToken(self, stanTokens, token=None):
		longestToken = ""
		bestStanToken = None
		for stanToken in stanTokens:
			if stanToken.pos in self.badPOS:
				# only use the badPOS if no others have been set
				if bestStanToken == None:
					bestStanToken = stanToken
			else:  # save the longest, nonBad POS tag
				if len(stanToken.text) > len(longestToken):
					longestToken = stanToken.text
					bestStanToken = stanToken
		if len(stanTokens) > 1 and token != None:
			print("token:", str(token.text), "=>", str(bestStanToken))
		if bestStanToken == None:
			print("* ERROR: our bestStanToken is empty!")
			exit(1)
		return bestStanToken
	################################################

	# singleFileOut = each mention's BoW vector:
	#    uidToVector (pickled), which contains:
	#         uid -> 1800: [[a][b][c][d][e][f]], where each letter is a 300-length word emb
	# relFileOut = every pair of mentions' cosine sim.
	#    uiduidToFeature -> 2 ([dot-product, cosine sim.])
	def saveBoWFeatures(self, singleFileOut, relFileOut):
		if len(self.gloveEmb) == 0: # don't want to wastefully load again
			self.loadGloveEmbeddings()
		muidPairs = self.getMUIDPairs()
		muids = set()
		for (m1,m2) in muidPairs:
			muids.add(m1)
			muids.add(m2)

		uidToVector = {} # will pickle
		uiduidToFeature = {} # will pickle
		# gets a vector for each mention
		for muid in muids:
			t_startIndex = 99999999
			t_endIndex = -1
			doc_id = self.corpus.XUIDToMention[muid].doc_id
			for t in self.corpus.XUIDToMention[muid].tokens:
				ind = self.corpus.corpusTokensToCorpusIndex[t]
				if ind < t_startIndex:
					t_startIndex = ind
				if ind > t_endIndex:
					t_endIndex = ind
			# the N tokens before and after, only 0'ing if it's part
			# of a diff document
			tmpTokens = []
			for i in range(self.bowWindow):
				ind = t_startIndex - self.bowWindow + i
				cur_t = self.corpus.corpusTokens[ind]
				found = False
				if ind >= 0 and cur_t.doc_id == doc_id and cur_t.text.rstrip() != "":
					#print("a:", cur_t.text, cur_t.doc_id, cur_t.sentenceNum,"cur:",self.corpus.corpusTokens[t_startIndex - 3], ",", self.corpus.corpusTokens[t_startIndex - 2], ",", self.corpus.corpusTokens[t_startIndex - 1])
					cleanedText = self.removeQuotes(cur_t.text)
					if cleanedText in self.gloveEmb:
						tmpTokens.append(self.gloveEmb[cleanedText])
						found = True
					elif len(cur_t.stanTokens) > 0:
						#print("b:", self.getBestStanToken(cur_t.stanTokens).text.lower())
						cleanedStan = self.removeQuotes(self.getBestStanToken(cur_t.stanTokens).text.lower())
						if cleanedStan in self.gloveEmb:
							tmpTokens.append(self.gloveEmb[cleanedStan])
							found = True
					if not found:
						print("WARNING: we don't have prevToken:", cleanedText, "or", cur_t.stanTokens)
						print("token:", cur_t, "stans:", cur_t.stanTokens)
						randEmb = []
						for i in range(300):
							randEmb.append(1)
						tmpTokens.append([0] * 300)
				else:
					tmpTokens.append([0] * 300)

			# N tokens after
			for i in range(self.bowWindow):
				ind = t_endIndex + 1 + i
				cur_t = self.corpus.corpusTokens[ind]
				found = False
				if ind < self.corpus.numCorpusTokens - 1 and cur_t.doc_id == doc_id:
					#print("c:", cur_t.text)
					cleanedText = self.removeQuotes(cur_t.text)
					if cleanedText in self.gloveEmb:
						tmpTokens.append(self.gloveEmb[cleanedText])
						found = True
					elif len(cur_t.stanTokens) > 0:
						#print("d:", self.getBestStanToken(cur_t.stanTokens).text.lower())
						cleanedStan = self.removeQuotes(self.getBestStanToken(cur_t.stanTokens).text.lower())
						if cleanedStan in self.gloveEmb:
							tmpTokens.append(self.gloveEmb[cleanedStan])
							found = True
					if not found:
						print("WARNING: we don't have nextToken:", cleanedText, "or", cur_t.stanTokens)
						print("token:", cur_t, "stans:", cur_t.stanTokens)
						tmpTokens.append([0] * 300)
				else:
					tmpTokens.append([0] * 300)
			uidToVector[self.corpus.XUIDToMention[muid].UID] = tmpTokens
		proc = 0
		for muid1, muid2 in muidPairs:
			uid1, uid2 = sorted([self.corpus.XUIDToMention[muid1].UID, self.corpus.XUIDToMention[muid2].UID])
			v1 = uidToVector[uid1]
			v2 = uidToVector[uid2]
			flatv1 = [item for sublist in v1 for item in sublist]
			flatv2 = [item for sublist in v2 for item in sublist]
			# dot product
			dp = 0
			denom1 = 0
			denom2 = 0
			for i in range(len(v1)):
				dp += flatv1[i] * flatv2[i]
				denom1 += flatv1[i]*flatv1[i]
				denom2 += flatv2[i]*flatv2[i]

			# cosine sim
			denom1 = sqrt(denom1)
			denom2 = sqrt(denom2)
			cs = -1
			if denom1 != 0 and denom2 != 0:
				cs = float(dp / (denom1 * denom2))

			uiduidToFeature[(uid1,uid2)] = [dp, cs]
			if proc % 1000 == 0:
				print("\tprocessed", proc, "of", len(muidPairs), "(%2.2f)" %
				      float(100.0*proc/len(muidPairs)), end="\r")
			proc += 1
		pickle_out = open(singleFileOut, 'wb')
		pickle.dump(uidToVector, pickle_out)
		pickle_out = open(relFileOut, 'wb')
		pickle.dump(uiduidToFeature, pickle_out)
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
