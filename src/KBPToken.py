class KBPToken:
	def __init__(self, event_file, source_file, charStartingIndex, charEndingIndex, text, globalTokenNum):
		self.event_file = event_file
		self.source_file = source_file
		self.charStartingIndex = charStartingIndex
		self.charEndingIndex = charEndingIndex
		self.text = text.strip()
		self.globalTokenNum = globalTokenNum

	def __str__(self):
		return "TOKEN " + str(self.globalTokenNum) + ": " + \
			str(self.text) + " (" + str(self.charStartingIndex) + " - " + str(self.charEndingIndex) + ")"
