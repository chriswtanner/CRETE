class KBPMention:
	def __init__(event_file, source_file, offset, length, text, REF):
		self.event_file = event_file
		self.source_file = source_file
		self.offset = offset
		self.length = length
		self.text = text
		self.REF = REF
