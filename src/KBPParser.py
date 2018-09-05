import os
import re
import fnmatch
from array import array
from collections import defaultdict
from KBPToken import KBPToken
from KBPDoc import KBPDoc
from KBPMention import KBPMention
from KBPCorpus import KBPCorpus
class KBPParser:
	def __init__(self, args, corpusPath):
		print("KBP Parser()")
		self.corpusPath = corpusPath
		self.trainingFilePairs = [] # list of tuples: (source, hopper)
		self.testingFilePairs = []  # list of tuples: (source, hopper)
		
		self.replacements = {}
		#self.replacements["vote-buying"] = "vote buying"
		#self.replacements["ex-husband"] = "ex husband"
		self.replacements["\x92"] = ""
		self.replacements["\x93"] = "'"
		self.replacements["\x94"] = "'"
		self.replacements["....."] = "."
		self.replacements["...."] = "."

		self.permissible = set() # keeps track of tokens which are okay, should align -- grabbed, mention annotations
		self.permissible.add(("election-day", "election"))
		self.permissible.add(("vote-buying", "buying"))
		self.permissible.add(("ex-husband", "ex husband"))
		self.permissible.add(("pre-election", "election"))
		self.permissible.add(("taxpayer-backed", "taxpayer"))
		self.permissible.add(("picked.", "picked"))
		self.permissible.add(("picked.'", "picked"))
		self.permissible.add(("rulers/war", "war"))
		self.permissible.add(("re-extradition", "extradition"))

	def tokenizeLine(self, line):
		ret = [] # returns a list of tuples (word, offset w.r.t to current line)
		char_index = 0
		cur_token = ""
		for char in line:
			if char == " ":
				# optionally finish token
				if len(cur_token) > 0:
					ret.append((cur_token, char_index - len(cur_token)))
					cur_token = ""
			else:
				cur_token += line[char_index]
			char_index += 1
		if len(cur_token) > 0:
			ret.append((cur_token, char_index - len(cur_token)))
		#print("ret:", ret)
		return ret

	def parseCorpus(self):
		'''
		0a94a90f8451a8bb7b64ad15120db374.txt == source
		0a94a90f8451a8bb7b64ad15120db374.event_hoppers.xml == hopper
		NYT_ENG_20130622.0061.txt
		NYT_ENG_20130622.0061.event_hoppers.xml
		1b386c986f9d06fd0a0dda70c3b8ade9.event_hoppers.xml
		'''

		corpus = KBPCorpus()
		for root, _, filenames in os.walk(self.corpusPath):
			for filename in fnmatch.filter(filenames, '*.txt'):
				source_file = os.path.join(root, filename)
				doc_id = source_file[source_file.rfind("/") + 1:]
				base = doc_id[:doc_id.rfind(".txt")]
				event_file = root[:source_file.rfind("/source")] + "/hopper/" + base + ".event_hoppers.xml"
				if "TRAIN" in source_file:
					self.trainingFilePairs.append((source_file, event_file))
				else:
					self.testingFilePairs.append((source_file, event_file))
				
				corpus.eventToSourceFile[event_file] = source_file
				corpus.sourceToEventFile[source_file] = event_file

		# reads each source file
		#<hopper id=\"(.+)\">
		ahref_regex = r"(.*)(<a href.+>)(.+)(</a>)(.*)"
		hopper_regex = r"<hopper id=\"(.+)\">"
		trigger_regex = r"<trigger source.+offset=\"(\d+)\" length=\"(\d+)\">(.+)<.+>"
		training_mentions = set()
		testing_mentions = set()
		UREF = 0 # hopper_id's repeat across documents, so we need to map them to unique REFs
		DOCHOPPERToUREF = {}
		globalTokenNum = 0
		self.numDocsProcessed = 0
		for source_file, event_file in sorted(self.trainingFilePairs + self.testingFilePairs):
			print("source_file:",source_file)
			charStartingIndexToToken = {}
			textToCharStartingIndex = defaultdict(set)
			#if "04bfe2831596d665b1585d8bf7bedd47.txt" not in source_file:
			#	continue
			# read the source file, while saving every token
			with open(source_file, 'r', encoding="utf-8") as f:
				charOffset = 1
				adjustment = 0 # how much we appear to be off; it's self-correcting
				cur_doc = KBPDoc(event_file, source_file)
				cur_tokens = []
				for line in f:

					it_a = tuple(re.finditer(ahref_regex, line))
					for match in it_a:
						line = match.group(1) + "\n" + match.group(2) + "\n" + match.group(3) + "\n" + match.group(4) + "\n" + match.group(5) + "\n"
						line = line.strip()
						
					for r in self.replacements:
						line = line.replace(r, self.replacements[r])
						#line = line.replace("/", " ")
					print("line:", line, "\t\tlen:",len(line))
					'''
					if line == "":
						#print("* blank line")
						if len(cur_tokens) > 0:
							cur_doc.addLineOfTokens(cur_tokens)
							#print("*** BLANK LINE, ADDING:", cur_tokens)
							cur_tokens = []
						continue
					'''
					# it's a < > line, so we can skip it, but should update our offsets
					if line.startswith("<") or line == "" or line == "\n":
						#charOffset += len(line)
						#print("line starts w/ <")
						if len(cur_tokens) > 0:
							cur_doc.addLineOfTokens(cur_tokens)
							#print("*** < LINE, ADDING:", cur_tokens)
							cur_tokens = []
					else: # save the Tokens on the line
						tokens = self.tokenizeLine(line)
						#print("tokens:", tokens)
						for token, relative_offset in tokens:
						#for i, token in enumerate(line.split(" ")):
							
							#print("making a new token:", token, "len:",len(token))
							newToken = KBPToken(event_file, source_file, \
							charOffset + relative_offset, charOffset+relative_offset+len(token), token, globalTokenNum)
							cur_tokens.append(newToken)
							charStartingIndexToToken[charOffset + relative_offset] = newToken

							# TODO: was token not newToken.cleaned_text
							textToCharStartingIndex[newToken.cleaned_text].add(charOffset + relative_offset)
							#print("made a token:",newToken)
							# updates corpus w/ the new Token
							corpus.addToken(newToken, globalTokenNum)

							#charOffset += len(token)
							#if i > 0:
							#	charOffset += 1 # if it's not the first token, then we were preceeded by a space
							globalTokenNum += 1
					charOffset += len(line)
			if len(cur_tokens) > 0:
				cur_doc.addLineOfTokens(cur_tokens)
				#print("*** END OF DOC, ADDING:", cur_tokens)
				cur_tokens = []

			print("globalTokenNum:", globalTokenNum)
			for i, line in enumerate(cur_doc.linesOfTokens):
				for t in line:
					print(i,t)
			
			corpus.addDoc(cur_doc)

			# read event file
			with open(event_file, 'r', encoding="utf-8") as f:

				cur_DOCHOPPER = ()
				for line in f:
					line = line.strip().lower()

					it_h = tuple(re.finditer(hopper_regex, line))
					for match in it_h:
						cur_DOCHOPPER = (event_file, match.group(1))
						continue

					it = tuple(re.finditer(trigger_regex, line))
					for match in it:
						offset = int(match.group(1))
						length = int(match.group(2))
						text = match.group(3)

						# finds the tokens involved in the given Mention
						(tokens, adjustment) = self.findTokens(corpus, textToCharStartingIndex, charStartingIndexToToken, offset, adjustment, text)
						
						# gets the REF for it (we could have just stored the curREF,
						# but if it repeats in the same doc, we're screwed)
						cur_REF = -1
						if cur_DOCHOPPER in DOCHOPPERToUREF:
							cur_REF = DOCHOPPERToUREF[cur_DOCHOPPER]
						else:
							cur_REF = UREF
							DOCHOPPERToUREF[cur_DOCHOPPER] = UREF
							UREF += 1

						'''
						then, as i loop through our source documents, i can see if 
						everythign aligns w/ its corresponding event_file_mentions
						(make sure i have a data structure from event_File -> mentions)
						m = KBPMention(event_file, source_file, offset, length, text, cur_REF)
						corpus.
						'''
			self.numDocsProcessed += 1
		#print("DOCHOPPERToUREF:", DOCHOPPERToUREF)

	def levenshtein(self, seq1, seq2, max_dist=-1):
		if seq1 == seq2:
			return 0
		len1, len2 = len(seq1), len(seq2)
		if max_dist >= 0 and abs(len1 - len2) > max_dist:
			return -1
		if len1 == 0:
			return len2
		if len2 == 0:
			return len1
		if len1 < len2:
			len1, len2 = len2, len1
			seq1, seq2 = seq2, seq1
		
		column = array('L', range(len2 + 1))
		
		for x in range(1, len1 + 1):
			column[0] = x
			last = x - 1
			for y in range(1, len2 + 1):
				old = column[y]
				cost = int(seq1[x - 1] != seq2[y - 1])
				column[y] = min(column[y] + 1, column[y - 1] + 1, last + cost)
				last = old
			if max_dist >= 0 and min(column) > max_dist:
				return -1
		
		if max_dist >= 0 and column[len2] > max_dist:
			# stay consistent, even if we have the exact distance
			return -1
		return column[len2]

	def findTokens(self, corpus, textToCharStartingIndex, charStartingIndexToToken, offset, adjustment, text):
		print("looking for:", text, str(offset + adjustment), "(offset:", offset, " adj:", adjustment)
		tokens = []
		text = text.strip().lower()

		# find the 2 closest tokens, and pick the one that looks most similar
		minDists = [9999, 9999]
		stack = [-1, -1]
		for char_index in charStartingIndexToToken:
			diff = abs((offset + adjustment) - char_index)
			# at least equals the lowest
			if diff <= minDists[0]:
				minDists[1] = minDists[0]
				minDists[0] = diff
				stack[1] = stack[0]
				stack[0] = char_index
			elif diff <= minDists[1]: # at least equals the 2nd best
				minDists[1] = diff
				stack[1] = char_index

		print("\tclosest ones:", stack,
		      charStartingIndexToToken[stack[0]].text, charStartingIndexToToken[stack[1]].text)

		chosenIndex = stack[0]

		# TODO: wasn't cleaned_text.  was just .text
		token1text = charStartingIndexToToken[stack[0]].cleaned_text.strip().lower()
		token2text = charStartingIndexToToken[stack[1]].cleaned_text.strip().lower()
		sed1 = self.levenshtein(token1text, text)
		sed2 = self.levenshtein(token2text, text)
		print("\ttext.find(token1text):", text.find(token1text))
		print("\ttoken1text.find(text):", token1text.find(text))
		if sed1 <= sed2 or text.find(token1text) == 0 or token1text.find(text) == 0:
			tokens.append(charStartingIndexToToken[stack[0]])
		else:
			tokens.append(charStartingIndexToToken[stack[1]])
			chosenIndex = stack[1]


		needsToValidate = True
		while needsToValidate:
			firstToken = tokens[0]
			print("\tfound:", firstToken)
			tokenIndex = corpus.corpusTokensToCorpusIndex[firstToken]
			curGrabbedText = firstToken.cleaned_text
			print("starting, curGrabbedText:", curGrabbedText, "; text:", text)

			while len(curGrabbedText) < len(text):
				tokenIndex += 1
				nextToken = corpus.corpusTokens[tokenIndex]
				if "-" in text:
					curGrabbedText += "-" + str(nextToken.cleaned_text)
				else:
					curGrabbedText += " " + str(nextToken.cleaned_text)

				tokens.append(nextToken)
			print("\tending, curGrabbedText:", curGrabbedText, "; text:", text)
			#print("*** we had multi tokens:", curGrabbedText, "vs", text)
			curGrabbedText = curGrabbedText.lower()

			if curGrabbedText.lower() != text.lower():
				print("** WARNING: texts don't match:", curGrabbedText, "vs", text, "numDocsProcessed:", self.numDocsProcessed)
				if (curGrabbedText, text) not in self.permissible:
					print("\t** ERROR, not in permissible")

					# ATTEMPTS TO JUST FIND THE TOKEN BY TEXT
					closestOffset = -1
					minDist = 9999
					if text in textToCharStartingIndex:
						for char_index in textToCharStartingIndex[text]:
							diff = abs((offset + adjustment) - char_index)
							if diff < minDist:
								minDist = diff
								closestOffset = char_index
					'''
					else:
						print("* error< we dont have the word, oddly")
						print("textToCharStartingIndex:", textToCharStartingIndex)
						exit(1)
					'''
					if closestOffset != -1:  # means we found the exact string somewhere
						print("\ttier2 -- we found the exact text, but at location:", closestOffset)
						tokens = []
						tokens.append(charStartingIndexToToken[closestOffset])
						chosenIndex = closestOffset
					#exit(1)
				else:
					print("** WE FOUND one in permissible")
					needsToValidate = False
			else:
				needsToValidate = False
		new_adj = chosenIndex - offset
		print("matched, so our adjustment is:" )
		return (tokens, new_adj)
		'''

		# check if a token exists at the exact location
		if offset in charStartingIndexToToken:
			print("\ttier1 -- returning token at exact location:", charStartingIndexToToken[offset])
			return charStartingIndexToToken[offset]

		# try matching by text, searching for nearby locations
		closestOffset = -1
		minDist = 9999
		if text in textToCharStartingIndex:
			for char_index in textToCharStartingIndex[text]:
				diff = abs(offset - char_index)
				if diff < minDist:
					minDist = diff
					closestOffset = char_index
		if closestOffset != -1: # means we found the exact string somewhere
			print("\ttier2 -- we found the exact text, but at location:", closestOffset)
			return charStartingIndexToToken[closestOffset]

		# try matching tokens that are similar, searching for nearby locations
		# find the closest key:
		closestKey = ""
		minSED = 9999
		for k in textToCharStartingIndex:
			cur_sed = self.levenshtein(k, text)
			if cur_sed < minSED:
				minSED = cur_sed
				closestKey = k
		print("\ttier3 -- the closest key seems to be:", closestKey, "with a SED of:", minSED)
		closestOffset = -1
		minDist = 9999
		for char_index in textToCharStartingIndex[closestKey]:
			diff = abs(offset - char_index)
			if diff < minDist:
				minDist = diff
				closestOffset = char_index
		if closestOffset != -1:  # means we found the nearby match, somewhere
			print("\ttier3 -- the closest location of", closestKey, "is:", closestOffset)
			return charStartingIndexToToken[closestOffset]
		else:
			print("* ERROR: we should never reach here")

		
		if offset in charStartingIndexToToken:
			print("we have offset,",offset)
			return charStartingIndexToToken[offset]
		else:

			candidates = []
			closestOffset = -1
			minDist = 9999
			for k in charStartingIndexToToken:
				diff = abs(offset - k)
				#print("\t\ttrying k, diff:", k, diff)
				if diff < minDist:
					minDist = diff
					closestOffset = k
					candidates = []
					candidates.append(closestOffset)
				elif diff == minDist: # tie
					candidates.append(k)
			if len(candidates) == 2:
				print("we have 2 candidates")
				
				for offset in candidates:
					if charStartingIndexToToken[offset].text == text:
						print("and we're returning", charStartingIndexToToken[offset])
						return charStartingIndexToToken[offset]
			
			
			print("\t\twe think the closest offset is:", closestOffset,
				  "which is:", charStartingIndexToToken[closestOffset])
			return charStartingIndexToToken[closestOffset]
		'''

