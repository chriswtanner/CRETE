import operator
import sys
import pickle
from math import sqrt
from nltk.corpus import wordnet as wn
from Mention import Mention
from StanToken import StanToken
from Feature import Feature
from collections import defaultdict

# NOTE: the save* functions operate on *all* mentions in self.corpus
# regardless if ecb, hddcrp, stan.
# the load() functions should ensure we only care about the right mentions
class FeatureHandler:
	def __init__(self, args, helper): #, trainXUIDs, devXUIDs, testXUIDs):
		self.args = args
		self.helper = helper
		self.corpus = helper.corpus
		'''
		self.trainXUIDs = trainXUIDs
		self.devXUIDs = devXUIDs
		self.testXUIDs = testXUIDs
		'''
		self.saveRelationalFeatures = False # NOTE: ensure this is what you want
		self.bowWindow = 3 # number of tokens on each side to look at
		self.gloveEmb = {} # to be filled in via loadGloveEmbeddings()
		self.charEmb = {} # to be filled in via loadCharEmbeddings()
		self.posEmb = {} # to be filled in via loadPOSEmbeddings()
		self.badPOS = ["‘’", "``", "POS", "$", "''"]
	############################
	###   HELPER FUNCTIONS   ###
	############################
	# create all pairs, for ecb, hddcrp, stan
	# this is correct; don't want to merely create *every* XUID pairs
	# because that would create, e.g., HDDCRP <-> STAN pairs
	def getAllXUIDPairs(self):
		ret = set()

		# on a per-dir basis
		for d in sorted(self.corpus.dirs):
			for euid1 in self.corpus.dirs[d].EUIDs:
				if euid1 in self.corpus.dirs[d].HUIDs or euid1 in self.corpus.dirs[d].SUIDs:
					print("DUPE1")
					exit(1)
				for euid2 in self.corpus.dirs[d].EUIDs:
					if euid2 <= euid1:
						continue
					ret.add((euid1, euid2))
			for huid1 in self.corpus.dirs[d].HUIDs:
				if huid1 in self.corpus.dirs[d].SUIDs:
					print("DUPE2")
					exit(1)
				for huid2 in self.corpus.dirs[d].HUIDs:
					if huid2 <= huid1:
						continue
					ret.add((huid1, huid2))
			for suid1 in self.corpus.dirs[d].SUIDs:
				for suid2 in self.corpus.dirs[d].SUIDs:
					if suid2 <= suid1:
						continue
					ret.add((suid1, suid2))
		# a per-dirhalf basis
		'''
		for dirhalf in self.corpus.dirHalves:
			for euid1 in self.corpus.dirHalves[dirhalf].EUIDs:
				if euid1 in self.corpus.dirHalves[dirhalf].HUIDs or euid1 in self.corpus.dirHalves[dirhalf].SUIDs:
					print("DUPE1")
					exit(1)
				for euid2 in self.corpus.dirHalves[dirhalf].EUIDs:
					if euid2 <= euid1:
						continue
					ret.add((euid1, euid2))
			for huid1 in self.corpus.dirHalves[dirhalf].HUIDs:
				if huid1 in self.corpus.dirHalves[dirhalf].SUIDs:
					print("DUPE2")
					exit(1)
				for huid2 in self.corpus.dirHalves[dirhalf].HUIDs:
					if huid2 <= huid1:
						continue
					ret.add((huid1, huid2))
			for huid1 in self.corpus.dirHalves[dirhalf].SUIDs:
				for huid2 in self.corpus.dirHalves[dirhalf].SUIDs:
					if huid2 <= huid1:
						continue
					ret.add((huid1, huid2))
		'''
		return ret

	# return dot product and cosine sim
	def getDPCS(self, v1, v2):
		# dot product
		dp = 0
		denom1 = 0
		denom2 = 0
		for i in range(len(v1)):
			dp += v1[i] * v2[i]
			denom1 += v1[i]*v1[i]
			denom2 += v2[i]*v2[i]

		# cosine sim
		denom1 = sqrt(denom1)
		denom2 = sqrt(denom2)
		cs = -1
		if denom1 != 0 and denom2 != 0:
			cs = float(dp / (denom1 * denom2))
		return (dp, cs)

	def loadPOSEmbeddings(self):
		f = open(self.args.posEmbeddingsFile, 'r', encoding="utf-8")
		for line in f:
			tokens = line.rstrip().split(" ")
			pos = tokens[0]
			emb = [float(x) for x in tokens[1:]]
			self.posEmb[pos] = emb
		f.close()

	def loadGloveEmbeddings(self):
		f = open(self.args.embeddingsFile, 'r', encoding="utf-8")
		for line in f:
			tokens = line.rstrip().split(" ")
			word = tokens[0]
			emb = [float(x) for x in tokens[1:]]
			self.gloveEmb[word] = emb
		f.close()

	def loadCharEmbeddings(self):
		f = open(self.args.charEmbeddingsFile, 'r', encoding="utf-8")
		for line in f:
			tokens = line.rstrip().split(" ")
			char = tokens[0]
			emb = [float(x) for x in tokens[1:]]
			self.charEmb[char] = emb
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

######### END OF HELPER FUNCTIONS ###########
#############################################

	def saveWordFeatures(self, fileOut):
		feature = Feature()
		if len(self.gloveEmb) == 0: # don't want to wastefully load again
			self.loadGloveEmbeddings()
		xuidPairs = self.getAllXUIDPairs()
		xuids = set()
		for (xuid1, xuid2) in xuidPairs:
			xuids.add(xuid1)
			xuids.add(xuid2)
		for xuid in xuids:
			sumEmb = [0] * 300
			for t in self.corpus.XUIDToMention[xuid].tokens:
				word = self.getBestStanToken(t.stanTokens).text.lower()
				if word not in self.gloveEmb:
					print("* ERROR: no word emb for", word)
					continue
				curEmb = self.gloveEmb[word]
				sumEmb = [x + y for x,y in zip(sumEmb, curEmb)]
			feature.setSingle(self.corpus.XUIDToMention[xuid].UID, sumEmb)
		
		# go through all pairs to compute relational data
		if self.saveRelationalFeatures:
			proc = 0
			completed = set()
			for xuid1, xuid2 in xuidPairs:
				uid1, uid2 = sorted([self.corpus.XUIDToMention[xuid1].UID, self.corpus.XUIDToMention[xuid2].UID])
				if (uid1,uid2) in completed or (uid2,uid1) in completed:
					continue
				completed.add((uid1,uid2))
				flatv1 = feature.singles[uid1]
				flatv2 = feature.singles[uid2]
				(dp, cs) = self.getDPCS(flatv1, flatv2)
				feature.addRelational(uid1, uid2, dp)
				feature.addRelational(uid1, uid2, cs)
				if proc % 1000 == 0:
					print("\tprocessed", proc, "of", len(xuidPairs), "(%2.2f)" %
						float(100.0*proc/len(xuidPairs)), end="\r")
				proc += 1

		pickle_out = open(fileOut, 'wb')
		pickle.dump(feature, pickle_out)

	def saveLemmaFeatures(self, fileOut):
		feature = Feature()
		if len(self.gloveEmb) == 0:  # don't want to wastefully load again
			self.loadGloveEmbeddings()
		xuidPairs = self.getAllXUIDPairs()
		xuids = set()
		for (xuid1, xuid2) in xuidPairs:
			xuids.add(xuid1)
			xuids.add(xuid2)
		for xuid in xuids:
			sumEmb = [0] * 300
			for t in self.corpus.XUIDToMention[xuid].tokens:
				lemma = self.getBestStanToken(t.stanTokens).lemma.lower()
				if lemma not in self.gloveEmb:
					print("* ERROR: no emb for",lemma)
					continue
				curEmb = self.gloveEmb[lemma]
				sumEmb = [x + y for x, y in zip(sumEmb, curEmb)]
			feature.setSingle(self.corpus.XUIDToMention[xuid].UID, sumEmb)

		if self.saveRelationalFeatures:
			# go through all pairs to compute relational data
			proc = 0
			completed = set()
			for xuid1, xuid2 in xuidPairs:
				uid1, uid2 = sorted([self.corpus.XUIDToMention[xuid1].UID,
								self.corpus.XUIDToMention[xuid2].UID])
				if (uid1, uid2) in completed or (uid2, uid1) in completed:
					continue
				completed.add((uid1, uid2))
				flatv1 = feature.singles[uid1]
				flatv2 = feature.singles[uid2]

				(dp, cs) = self.getDPCS(flatv1, flatv2)
				feature.addRelational(uid1, uid2, dp)
				feature.addRelational(uid1, uid2, cs)
				if proc % 1000 == 0:
					print("\tprocessed", proc, "of", len(xuidPairs), "(%2.2f)" %
						float(100.0*proc/len(xuidPairs)), end="\r")
				proc += 1
		pickle_out = open(fileOut, 'wb')
		pickle.dump(feature, pickle_out)

	def saveCharFeatures(self, fileOut):
		feature = Feature()
		if len(self.charEmb) == 0:
			self.loadCharEmbeddings()
		xuidPairs = self.getAllXUIDPairs()
		xuids = set()
		for (xuid1, xuid2) in xuidPairs:
			xuids.add(xuid1)
			xuids.add(xuid2)
		for xuid in xuids:
			charEmb = []
			numCharsFound = 0
			for t in self.corpus.XUIDToMention[xuid].tokens:
				lemma = self.getBestStanToken(t.stanTokens).lemma.lower()
				for char in lemma:
					if char == "ô":
						char = "o"
					if char in self.charEmb:
						if numCharsFound == 20:
							break
						else:
							charEmb += self.charEmb[char]
							numCharsFound += 1
					else:
						print("* WARNING: we don't have char:", str(char))
						#exit(1)
			while len(charEmb) < 400: # 20 chars * 20 dim
				charEmb.append(0.0)
			feature.setSingle(self.corpus.XUIDToMention[xuid].UID, charEmb)

		# go through all pairs to compute relational data
		if self.saveRelationalFeatures:
			proc = 0
			completed = set()
			for xuid1, xuid2 in xuidPairs:
				uid1, uid2 = sorted([self.corpus.XUIDToMention[xuid1].UID,
								self.corpus.XUIDToMention[xuid2].UID])
				if (uid1, uid2) in completed or (uid2, uid1) in completed:
					continue
				completed.add((uid1, uid2))
				flatv1 = feature.singles[uid1]
				flatv2 = feature.singles[uid2]
				(dp, cs) = self.getDPCS(flatv1, flatv2)
				feature.addRelational(uid1, uid2, dp)
				feature.addRelational(uid1, uid2, cs)
				if proc % 1000 == 0:
					print("\tprocessed", proc, "of", len(xuidPairs), "(%2.2f)" %
						float(100.0*proc/len(xuidPairs)), end="\r")
				proc += 1
		pickle_out = open(fileOut, 'wb')
		pickle.dump(feature, pickle_out)

	def savePOSFeatures(self, fileOut):
		feature = Feature()

		posLength = 50
		if len(self.posEmb) == 0:
			self.loadPOSEmbeddings()
		xuidPairs = self.getAllXUIDPairs()
		xuids = set()
		for (xuid1, xuid2) in xuidPairs:
			xuids.add(xuid1)
			xuids.add(xuid2)
		for xuid in xuids:
			sumEmb = [0]*posLength
			for t in self.corpus.XUIDToMention[xuid].tokens:
				pos = ""
				posOfLongestToken = ""
				longestToken = ""
				for stanToken in t.stanTokens:
					if stanToken.pos in self.badPOS:
						# only use the badPOS if no others have been set
						if pos == "":
							pos = stanToken.pos
					else: # save the longest, nonBad POS tag
						if len(stanToken.text) > len(longestToken):
							longestToken = stanToken.text
							posOfLongestToken = stanToken.pos 

				if posOfLongestToken != "":
					pos = posOfLongestToken
				if pos == "":
					print("* ERROR: our POS empty!")
					exit(1)

				curEmb = self.posEmb[pos]
				sumEmb = [x + y for x,y in zip(sumEmb, curEmb)]
			feature.setSingle(self.corpus.XUIDToMention[xuid].UID, sumEmb)

		# go through all pairs to compute relational data
		if self.saveRelationalFeatures:
			completed = set()
			proc = 0
			for xuid1, xuid2 in xuidPairs:
				uid1, uid2 = sorted([self.corpus.XUIDToMention[xuid1].UID,
								self.corpus.XUIDToMention[xuid2].UID])
				if (uid1, uid2) in completed or (uid2, uid1) in completed:
					continue
				completed.add((uid1, uid2))
				flatv1 = feature.singles[uid1]
				flatv2 = feature.singles[uid2]

				(dp, cs) = self.getDPCS(flatv1, flatv2)
				feature.addRelational(uid1, uid2, dp)
				feature.addRelational(uid1, uid2, cs)
				if proc % 1000 == 0:
					print("\tprocessed", proc, "of", len(xuidPairs), "(%2.2f)" %
						float(100.0*proc/len(xuidPairs)), end="\r")
				proc += 1
		pickle_out = open(fileOut, 'wb')
		pickle.dump(feature, pickle_out)

	def saveDependencyFeatures(self, fileOut):
		feature = Feature()
		if len(self.gloveEmb) == 0:
			self.loadGloveEmbeddings()
		xuidPairs = self.getAllXUIDPairs()
		xuids = set()
		for (xuid1, xuid2) in xuidPairs:
			xuids.add(xuid1)
			xuids.add(xuid2)
		for xuid in xuids:
			sumParentEmb = [0]*300
			sumChildrenEmb = [0]*300
			numParentFound = 0
			tmpParentLemmas = []
			numChildrenFound = 0
			tmpChildrenLemmas = []
			for t in self.corpus.XUIDToMention[xuid].tokens:
				bestStanToken = self.getBestStanToken(t.stanTokens)
				
				if len(bestStanToken.parentLinks) == 0:
					print("* token has no dependency parent!")
					exit(1)
				for stanParentLink in bestStanToken.parentLinks:
					parentLemma = self.removeQuotes(stanParentLink.parent.lemma.lower())
					curEmb = [0]*300
					
					# TMP: just to see which texts we are missing
					tmpParentLemmas.append(parentLemma)

					if parentLemma == "ROOT":
						curEmb = [1]*300
					elif parentLemma in self.gloveEmb:
						curEmb = self.gloveEmb[parentLemma]
						numParentFound += 1
					sumParentEmb = [x + y for x,y in zip(sumParentEmb, curEmb)]
				
				# makes embedding for the dependency children's lemmas
				#if len(bestStanToken.childLinks) == 0:
				#	print("* token has no dependency children!")
				for stanChildLink in bestStanToken.childLinks:
					childLemma = self.removeQuotes(stanChildLink.child.lemma.lower())
					curEmb = [0]*300
					
					# TMP: just to see which texts we are missing
					tmpChildrenLemmas.append(childLemma)

					if childLemma == "ROOT":
						curEmb = [1]*300
					elif childLemma in self.gloveEmb:
						curEmb = self.gloveEmb[childLemma]
						numChildrenFound += 1					
					sumChildrenEmb = [x + y for x,y in zip(sumChildrenEmb, curEmb)]
			parentEmb = sumParentEmb  # makes parent emb
			childrenEmb = sumChildrenEmb  # makes chid emb
			feature.setSingle(self.corpus.XUIDToMention[xuid].UID, parentEmb + childrenEmb)
		
		# go through all pairs to compute relational data
		if self.saveRelationalFeatures:
			proc = 0
			completed = set()
			for xuid1, xuid2 in xuidPairs:
				uid1, uid2 = sorted([self.corpus.XUIDToMention[xuid1].UID, self.corpus.XUIDToMention[xuid2].UID])
				if (uid1, uid2) in completed or (uid2, uid1) in completed:
					continue
				completed.add((uid1, uid2))
				flatv1 = feature.singles[uid1]
				flatv2 = feature.singles[uid2]

				(dp, cs) = self.getDPCS(flatv1, flatv2)
				feature.addRelational(uid1, uid2, dp)
				feature.addRelational(uid1, uid2, cs)
				if proc % 1000 == 0:
					print("\tprocessed", proc, "of", len(xuidPairs), "(%2.2f)" %
										float(100.0*proc/len(xuidPairs)), end="\r")
				proc += 1
		pickle_out = open(fileOut, 'wb')
		pickle.dump(feature, pickle_out)

	# entire feature is relational
	def saveWordNetFeatures(self, fileOut):
		feature = Feature()

		synSynToScore = {}
		xuidPairs = self.getAllXUIDPairs()
		print("calculating wordnet features for", len(xuidPairs), "unique pairs")
		i = 0
		completed = set()
		for xuid1, xuid2 in xuidPairs:
			uid1 = self.corpus.XUIDToMention[xuid1].UID
			uid2 = self.corpus.XUIDToMention[xuid2].UID
			if (uid1, uid2) in completed or (uid2, uid1) in completed:
				continue
			completed.add((uid1, uid2))
			textTokens1 = self.corpus.XUIDToMention[xuid1].text
			textTokens2 = self.corpus.XUIDToMention[xuid2].text
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
					else:  # calculate it
						curScore = wn.wup_similarity(syn1, syn2)
						# don't want to store tons.  look-up is cheap
						synSynToScore[(syn1, syn2)] = curScore
						if curScore != None and curScore > bestScore:
							bestScore = curScore

			feature.addRelational(uid1, uid2, bestScore)
			i += 1
			if i % 1000 == 0:
				print("\tprocessed", i, "of", len(xuidPairs), "(%2.2f)" %
					  float(100.0*i/len(xuidPairs)), end="\r")

		pickle_out = open(fileOut, 'wb')
		pickle.dump(feature, pickle_out)
		print("")


	# singleFileOut = each mention's BoW vector:
	#    uidToVector (pickled), which contains:
	#         uid -> 1800: [[a][b][c][d][e][f]], where each letter is a 300-length word emb
	# relFileOut = every pair of mentions' cosine sim.
	#    uiduidToFeature -> 2 ([dot-product, cosine sim.])
	def saveBoWFeatures(self, fileOut):
		feature = Feature()
		if len(self.gloveEmb) == 0: # don't want to wastefully load again
			self.loadGloveEmbeddings()
		xuidPairs = self.getAllXUIDPairs()
		xuids = set()
		for (xuid1,xuid2) in xuidPairs:
			xuids.add(xuid1)
			xuids.add(xuid2)

		#uidToVector = {} # will pickle
		#uiduidToFeature = {} # will pickle
		# gets a vector for each mention
		for xuid in xuids:
			t_startIndex = 99999999
			t_endIndex = -1
			doc_id = self.corpus.XUIDToMention[xuid].doc_id
			for t in self.corpus.XUIDToMention[xuid].tokens:
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
			#uidToVector[self.corpus.XUIDToMention[xuid].UID] = tmpTokens
			flatvector = [item for sublist in tmpTokens for item in sublist]
			#print(flatvector)
			feature.setSingle(self.corpus.XUIDToMention[xuid].UID, flatvector)
		
		if self.saveRelationalFeatures:
			proc = 0
			completed = set()
			for xuid1, xuid2 in xuidPairs:
				uid1, uid2 = sorted([self.corpus.XUIDToMention[xuid1].UID, self.corpus.XUIDToMention[xuid2].UID])
				if (uid1, uid2) in completed or (uid2, uid1) in completed:
					continue
				completed.add((uid1, uid2))
				flatv1 = feature.singles[uid1]
				flatv2 = feature.singles[uid2]

				(dp, cs) = self.getDPCS(flatv1, flatv2)
				feature.addRelational(uid1, uid2, dp)
				feature.addRelational(uid1, uid2, cs)
				if proc % 1000 == 0:
					print("\tprocessed", proc, "of", len(xuidPairs), "(%2.2f)" %
						float(100.0*proc/len(xuidPairs)), end="\r")
				proc += 1

		pickle_out = open(fileOut, 'wb')
		pickle.dump(feature, pickle_out)
