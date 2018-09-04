class KBPToken:
	def __init__(self, event_file, source_file, charStartingIndex, charEndingIndex, text, globalTokenNum):
		self.trimEndChars = ["'", "\"", ".", ","]
		self.event_file = event_file
		self.source_file = source_file
		self.charStartingIndex = charStartingIndex
		self.charEndingIndex = charEndingIndex
		self.text = text.strip() # the actual text found in the corpus
		self.cleaned_text = self.cleanText(self.text) # the format that will be used for predicting mentions and displaying
		self.globalTokenNum = globalTokenNum

	def cleanText(self, text):
		if len(text) > 0:
			if text == "''" or text == "\"":
				return "\""
			elif text == "'" or text == "'s":
				return text
			else:  # there's more substance to it, not a lone quote
				if text[0] == "'" or text[0] == "\"":
					text = text[1:]
				if len(text) > 0:
					if text[-1] in self.trimEndChars:
						text = text[0:-1]
				if text[0:3] == "ex-":
					text = "ex"

				return text
		else:
			print("* found a blank")
			exit(1)
			return ""

	def __str__(self):
		return "text " + str(self.globalTokenNum) + ": " + \
			str(self.text) + " (" + str(self.charStartingIndex) + " - " + str(self.charEndingIndex) + ") => " + self.cleaned_text
