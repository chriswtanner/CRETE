class KBPDoc:
    def __init__(self, event_file, source_file):
        self.event_file = event_file
        self.source_file = source_file
        self.linesOfTokens = [] # not necessarily sentences; this only
        # exists so that we can print a nicely formatted file for stanford
    
    def addLineOfTokens(self, tokens):
        self.linesOfTokens.append(tokens)
    