import os
import re
import fnmatch
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
		self.replacements["vote-buying"] = "vote buying"
		self.replacements["ex-husband"] = "ex husband"
		self.replacements["\x92"] = ""
		self.replacements["\x93"] = "'"
		self.replacements["\x94"] = "'"

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
		hopper_regex = r"<hopper id=\"(.+)\">"
		trigger_regex = r"<trigger source.+offset=\"(\d+)\" length=\"(\d+)\">(.+)<.+>"
		training_mentions = set()
		testing_mentions = set()
		UREF = 0 # hopper_id's repeat across documents, so we need to map them to unique REFs
		DOCHOPPERToUREF = {}
		globalTokenNum = 0
		for source_file, event_file in sorted(self.trainingFilePairs + self.testingFilePairs):
			print("source_file:",source_file)
			#if "04bfe2831596d665b1585d8bf7bedd47.txt" not in source_file:
			#	continue
			# read the source file, while saving every token
			with open(source_file, 'r', encoding="utf-8") as f:
				charOffset = 1
				cur_doc = KBPDoc(event_file, source_file)
				cur_tokens = []
				for line in f:

					for r in self.replacements:
						line = line.replace(r, self.replacements[r])
						#line = line.replace("/", " ")
					#print("line:", line, "\t\tlen:",len(line))
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
						for token, relative_offset in tokens:
						#for i, token in enumerate(line.split(" ")):
							
							#print("making a new token:", token, "len:",len(token))
							newToken = KBPToken(event_file, source_file, \
							charOffset + relative_offset, charOffset+relative_offset+len(token), token, globalTokenNum)
							cur_tokens.append(newToken)
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

			'''
			print("globalTokenNum:", globalTokenNum)
			for i, line in enumerate(cur_doc.linesOfTokens):
				for t in line:
					print(i,t)
			'''
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
						tokens = []

						firstToken = self.findToken(corpus, offset, text)
						#print("looking for:", offset, "found:",firstToken)
						tokenIndex = corpus.corpusTokensToCorpusIndex[firstToken]
						curGrabbedText = firstToken.cleaned_text
						tokens.append(firstToken)
						#print("starting, curGrabbedText:", curGrabbedText, "; text:", text)
						while len(curGrabbedText) < len(text):
							tokenIndex += 1
							nextToken = corpus.corpusTokens[tokenIndex]
							if "-" in text:
								curGrabbedText += "-" + str(nextToken.cleaned_text)
							else:
								curGrabbedText += " " + str(nextToken.cleaned_text)

							tokens.append(nextToken)
						#print("*** we had multi tokens:", curGrabbedText, "vs", text)
						curGrabbedText = curGrabbedText.lower()
						if curGrabbedText != text:
							print("** ERROR: texts don't match:", curGrabbedText, "vs", text)
							exit(1)
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

		#print("DOCHOPPERToUREF:", DOCHOPPERToUREF)

	def findToken(self, corpus, offset, text):
		if offset in corpus.charStartingIndexToToken:
			return corpus.charStartingIndexToToken[offset]
		else:
			closestOffset = -1
			minDist = 9999
			for k in corpus.charStartingIndexToToken:
				diff = abs(offset - k)
				if diff < minDist:
					minDist = diff
					closestOffset = k
			return corpus.charStartingIndexToToken[closestOffset]

