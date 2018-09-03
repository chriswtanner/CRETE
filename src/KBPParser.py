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

			if "APW_ENG_20090605.0323.txt" not in source_file:
				continue
			# read the source file, while saving every token
			filebase = source_file[source_file.rfind("/") + 1:]
			fout = open("../data/kbp_stanford_input/" + filebase, 'w')
			with open(source_file, 'r', encoding="utf-8") as f:
				charOffset = 1
				cur_doc = KBPDoc(event_file, source_file)
				cur_tokens = []
				for line in f:
					#print("line:", line)
					if line == "":
						#print("* blank line")
						if len(cur_tokens) > 0:
							cur_doc.addLineOfTokens(cur_tokens)
							#print("*** BLANK LINE, ADDING:", cur_tokens)
							cur_tokens = []
						continue

					# it's a < > line, so we can skip it, but should update our offsets
					if line.startswith("<"):
						charOffset += len(line)
						#print("line starts w/ <")
						if len(cur_tokens) > 0:
							cur_doc.addLineOfTokens(cur_tokens)
							#print("*** < LINE, ADDING:", cur_tokens)
							cur_tokens = []
					else: # save the Tokens on the line
						for i, token in enumerate(line.split(" ")):
							newToken = KBPToken(event_file, source_file, \
							charOffset, charOffset+len(token), token, globalTokenNum)
							cur_tokens.append(newToken)
							#print("made a token:",newToken)
							# updates corpus w/ the new Token
							corpus.corpusTokens.append(newToken)
							corpus.corpusTokensToCorpusIndex[newToken] = globalTokenNum
							charOffset += len(token)
							if i > 0:
								charOffset += 1 # if it's not the first token, then we were preceeded by a space
							globalTokenNum += 1
			if len(cur_tokens) > 0:
				cur_doc.addLineOfTokens(cur_tokens)
				#print("*** END OF DOC, ADDING:", cur_tokens)
				cur_tokens = []
			fout.close()
			print("filebase:", filebase)
			print("globalTokenNum:", globalTokenNum)
			for i, line in enumerate(cur_doc.linesOfTokens):
				for t in line:
					print(i,t)

			corpus.addDoc(cur_doc)
			exit(1)
			# read event file
			'''
			filebase = event_file[event_file.rfind("/") + 1:]
			fout = open("../data/kbp_stanford_input/" + filebase, 'w')
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

						# gets the REF for it (we could have just stored the curREF,
						# but if it repeats in the same doc, we're screwed)
						cur_REF = -1
						if cur_DOCHOPPER in DOCHOPPERToUREF:
							cur_REF = DOCHOPPERToUREF[cur_DOCHOPPER]
						else:
							cur_REF = UREF
							DOCHOPPERToUREF[cur_DOCHOPPER] = UREF
							UREF += 1


						then, as i loop through our source documents, i can see if 
						everythign aligns w/ its corresponding event_file_mentions
						(make sure i have a data structure from event_File -> mentions)
						m = KBPMention(event_file, source_file, offset, length, text, cur_REF)
						corpus.
			fout.close()
			'''

		print("DOCHOPPERToUREF:", DOCHOPPERToUREF)
