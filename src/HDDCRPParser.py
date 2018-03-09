import sys
from Token import *
from Mention import *
from collections import defaultdict

class HDDCRPParser:
	def __init__(self, args):
		self.args = args
		
	# parses the hddcrp *semeval.txt file (which is in CoNLL-ready format)
	def parseCorpus(self, inputFile):
		self.htokens = {}
		self.UIDToToken = {}
		self.hmentions = []
		#self.docToHMentions = defaultdict(list)
		#self.docToUIDs = defaultdict(list)
		self.HMUIDToHMention = {}
		#self.dirToDocs = defaultdict(set)

		REFToStartTuple = defaultdict(list)
		tokenIndex = 0
		sentenceNum = 0
		HMUID = 0

		#self.docREFToHMUIDs = defaultdict(set)
		#self.docSentences = defaultdict(lambda: defaultdict(list))

		f = open(inputFile, "r")
		for line in f:
			line = line.rstrip()
			tokens = line.split("\t")
			if line.startswith("#") and "document" in line:
				sentenceNum = 0
			elif line == "":
				sentenceNum += 1
			elif len(tokens) == 5:
				doc, _, tokenNum, text, ref_ = tokens
				dir_num = doc[0:doc.find("_")]

				# the construction sets a member variable "uid" = doc_id, sentence_id, token_num
				curToken = HToken(doc, sentenceNum, tokenNum, text.lower())
				self.htokens[tokenIndex] = curToken
				self.UIDToToken[curToken.UID] = curToken
				self.docToUIDs[doc].append(curToken.UID)

				# TMP: only used for analyzeResults() in CCNN (to see the original sentences)
				self.docSentences[doc][sentenceNum].append(text.lower())

				refs = []
				if ref_.find("|") == -1:
					refs.append(ref_)
				else:  # we at most have 1 |
					refs.append(ref_[0:ref_.find("|")])
					refs.append(ref_[ref_.find("|")+1:])
					#print("***** FOUND 2:",str(line))

				isFirst = True
				for ref in refs:
					if ref[0] == "(" and ref[-1] != ")":  # i.e. (ref_id
						ref_id = int(ref[1:])

						# only store them if it's truly the un-finished start of a Mention,
						# which will later be closed.  otherwise, we don't need to store it, as
						# it'll be a () on the same line
						REFToStartTuple[ref_id].append((tokenIndex, isFirst))

					# represents we are ending a mention
					elif ref[-1] == ")":  # i.e., ref_id) or (ref_id)
						ref_id = -1
						tokens = []
						MUID = ""
						endTuple = (tokenIndex, isFirst)
						startTuple = ()
						# we set ref_if, tokens, UID
						if ref[0] != "(":  # ref_id)
							ref_id = int(ref[:-1])
							startTuple = REFToStartTuple[ref_id].pop()

							# add all tokens, including current one
							for i in range(startTuple[0], tokenIndex+1):
								tokens.append(self.htokens[i])
								MUID += self.htokens[i].UID + ";"

						else:  # (ref_id)
							ref_id = int(ref[1:-1])
							startTuple = (tokenIndex, isFirst)
							tokens.append(curToken)
							MUID = curToken.UID + ";"

						curMention = HMention(doc, ref_id, tokens, MUID,
						                      HMUID, startTuple, endTuple)
						self.docToHMentions[doc].append(curMention)
						self.hmentions.append(curMention)
						self.MUIDToHMentions[MUID] = curMention
						self.HMUIDToHMention[HMUID] = curMention
						self.docREFsToHMUIDs[(doc, ref_id)].add(HMUID)
						self.dirToDocs[dir_num].add(doc)
						HMUID += 1

					isFirst = False
				# end of current token line
				tokenIndex += 1  # this always increases whenever we see a token

			else:
				print("ERROR: curLine:", str(line))
				exit(1)
		f.close()
		hms = set()
		for doc_id in self.docToHMentions.keys():
			for hm in self.docToHMentions[doc_id]:
				hms.add(hm)
		print("\t# hms by end of parsing, based on a per doc basis:", str(len(hms)))
