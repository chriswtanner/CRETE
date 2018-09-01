import os
import re
import fnmatch
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

		# reads each source file
		#<hopper id=\"(.+)\">
			# reads <markables> 1st time
		regex = r"^<.*>$"
		for source_file, event_file in self.trainingFilePairs + self.testingFilePairs:
			
			# go through all event files and this regex should work
			# ^<.+>(.+)<.+>

			# read source file
			#f = open(source_file, 'r')
			filebase = source_file[source_file.rfind("/") + 1:]
			fout = open("../data/kbp_stanford_input/" + filebase, 'w')
			lastLineDateline = False
			with open(source_file, 'r', encoding="utf-8") as f:
				for line in f:
					line = line.rstrip()
					if line == "<DATELINE>":
						lastLineDateline = True
					else:
						lastLineDateline = False
					it = tuple(re.finditer(regex, line))
					if len(line) > 0 and len(it) == 0 and lastLineDateline == False:
						fout.write(line + "\n")
			fout.close()
