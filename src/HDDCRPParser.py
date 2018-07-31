import sys

from collections import defaultdict
class HDDCRPParser:
	def __init__(self, args):
		self.args = args

	# parses the hddcrp *semeval.txt file (which is in CoNLL-ready format)
	def parseCorpus(self, inputFile):
		
		self.hmentions = [] # STORE ALL PARSED HMENTIONS, even if we don't
		# add it to corpus (i.e., ones not in ECB's ValidSentences)
		
		REFToStartTuple = defaultdict(list)
		tokenIndex = 0
		sentenceNum = 0
		tokenIndexToHUID = {}

		# TMP -- keeping track of singletons w.r.t. cross-doc
		REFToDocs = defaultdict(set)
		REFToDirHalves = defaultdict(set)
		f = open(inputFile, "r")
		for line in f:
			line = line.rstrip()
			tokens = line.split("\t")
			if line.startswith("#") and "document" in line:
				sentenceNum = 0
			elif line == "":
				sentenceNum += 1
			elif len(tokens) == 5:
				doc_id, _, hTokenNum, text, ref_ = tokens
				HUID = str(doc_id) + ";" + str(sentenceNum) + \
                                    ";" + str(hTokenNum) + \
                                    ";" + str(text.lower())
				tokenIndexToHUID[tokenIndex] = HUID

				dir_num = int(doc_id.split("_")[0])
				extension = doc_id[doc_id.find("ecb"):]
				dirHalf = str(dir_num) + extension


				refs = []
				if ref_.find("|") == -1:
					refs.append(ref_)
				else: # we at most have 1 |
					refs.append(ref_[0:ref_.find("|")])
					refs.append(ref_[ref_.find("|")+1:])

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
						HUIDs = []
						startTuple = ()

						# we set ref_if, tokens, UID
						if ref[0] != "(":  # ref_id)
							ref_id = int(ref[:-1])
							startTuple = REFToStartTuple[ref_id].pop()

							# add all tokens, including current one
							for i in range(startTuple[0], tokenIndex+1):
								HUIDs.append(tokenIndexToHUID[i])

						else: # (ref_id)
							ref_id = int(ref[1:-1])
							startTuple = (tokenIndex, isFirst)
							HUIDs.append(HUID)

						#  TMP -- keeping track of singletons w.r.t. cross-doc
						REFToDocs[ref_id].add(doc_id)
						REFToDirHalves[ref_id].add(dirHalf)
						self.hmentions.append(HUIDs)

					isFirst = False
				# end of current token line
				tokenIndex += 1  # this always increases whenever we see a token
			else:
				print("ERROR: curLine:", str(line))
				exit(1)
		f.close()

		numSingletons = 0
		for ref in REFToDocs:
			if len(REFToDocs[ref]) == 1:
				numSingletons += 1
			#if len(REFToDirHalves[ref]) > 1:
			#	print("* WHOA. 2 dirhalves for REF:",ref,REFToDirHalves[ref])
		print("HDDCRP Gold file had #REFs:", len(REFToDocs),
		      "(", numSingletons, " were singletons)")

		'''
		# TMP -- writing new non-singleton file
		g_wd = open("gold.ns.wd.txt", 'w')
		g_cd = open("gold.ns.cd.txt", 'w')
		dirHalfToWriter = defaultdict(list)

		f = open(inputFile, "r")
		header = True
		cd_line_to_make = ""
		for line in f:
			line = line.rstrip()
			tokens = line.split("\t")

			if line.startswith("#") and "document" in line:
				sentenceNum = 0
				g_wd.write(line + "\n")

			elif line == "":
				sentenceNum += 1
				g_wd.write(line + "\n")
			elif len(tokens) == 5:
				doc_id, _, hTokenNum, text, ref_ = tokens
				HUID = str(doc_id) + ";" + str(sentenceNum) + \
                                    ";" + str(hTokenNum) + \
                                    ";" + str(text.lower())
				tokenIndexToHUID[tokenIndex] = HUID

				dir_num = int(doc_id.split("_")[0])
				extension = doc_id[doc_id.find("ecb"):]
				dirHalf = str(dir_num) + extension

				suffix = "_ecb"
				if "plus" in extension:
					suffix = "_ecbplus"
				g_wd.write(doc_id + "\t" + str(_) + "\t" + hTokenNum + "\t" + text + "\t")
				cd_line_to_make = (str(dir_num) + suffix + "\t" + str(_) + "\t" + hTokenNum + "\t" + text + "\t")

				refs = []
				if ref_.find("|") == -1:
					refs.append(ref_)
				else: # we at most have 1 |
					refs.append(ref_[0:ref_.find("|")])
					refs.append(ref_[ref_.find("|")+1:])
				if len(refs) > 1:
					print("**** WE HAVE > 1 ref!!!", ref_)

				isFirst = True
				for ref in refs:
					if ref[0] == "(" and ref[-1] != ")":  # i.e. (ref_id
						ref_id = int(ref[1:])
						if len(REFToDocs[ref_id]) > 1:  # it's cross-doc, so let's add it like normal
							g_wd.write("(" + str(ref_id))
							cd_line_to_make += "(" + str(ref_id)
						else: # it's not good, so let's not add an opening (
							g_wd.write("-")
							cd_line_to_make += "-"
						# only store them if it's truly the un-finished start of a Mention,
						# which will later be closed.  otherwise, we don't need to store it, as
						# it'll be a () on the same line
						REFToStartTuple[ref_id].append((tokenIndex, isFirst))

					# represents we are ending a mention
					elif ref[-1] == ")":  # i.e., ref_id) or (ref_id)
						ref_id = -1
						HUIDs = []
						startTuple = ()

						# we set ref_id, tokens, UID
						if ref[0] != "(":  # ref_id)
							ref_id = int(ref[:-1])
							startTuple = REFToStartTuple[ref_id].pop()

							if len(REFToDocs[ref_id]) > 1:  # it's cross-doc, so let's add it like normal
								g_wd.write(str(ref_id) + ")")
								cd_line_to_make += str(ref_id) + ")"
							else:  # it's not good, so let's not add an opening (
								g_wd.write("-")
								cd_line_to_make += "-"

							# add all tokens, including current one
							for i in range(startTuple[0], tokenIndex+1):
								HUIDs.append(tokenIndexToHUID[i])

						else:  # (ref_id)
							ref_id = int(ref[1:-1])
							startTuple = (tokenIndex, isFirst)
							HUIDs.append(HUID)

							if len(REFToDocs[ref_id]) > 1:  # it's cross-doc, so let's add it like normal
								g_wd.write("(" + str(ref_id) + ")")
								cd_line_to_make += "(" + str(ref_id) + ")"
							else:  # it's not good, so let's not add an opening (
								g_wd.write("-")
								cd_line_to_make += "-"
						#  TMP -- keeping track of singletons w.r.t. cross-doc
						REFToDocs[ref_id].add(doc_id)
						self.hmentions.append(HUIDs)
					elif ref == "-":
						g_wd.write("-")
						cd_line_to_make += "-"
					isFirst = False
				# end of current token line
				tokenIndex += 1  # this always increases whenever we see a token
				g_wd.write("\n")
				dirHalfToWriter[dirHalf].append(cd_line_to_make)
			else:
				print("ERROR: curLine:", str(line))
				exit(1)
			header=False

		g_wd.close()
		for dirHalf in dirHalfToWriter:
			g_cd.write("#begin document (" + dirHalf + "); part 000\n")
			for line in dirHalfToWriter[dirHalf]:
				g_cd.write(line + "\n")
			g_cd.write("#end document\n\n")
		g_cd.close()
		'''
		f.close()
		#print("hddcrp parsed:",len(self.hmentions))
		return self.hmentions