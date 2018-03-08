import sys
import re
import os
import fnmatch
from collections import defaultdict
from Corpus import Corpus
from Token import Token
from Mention import Mention
class ECBParser:
    def __init__(self, args):
        self.args = args

        # sets global vars
        self.replacements = {}
        self.replacementsList = []
        # for quicker indexing, since we'll do it over every token
        self.replacementsSet = set()
        self.endPunctuation = set()
        self.endPunctuation.update(".", "!", "?")
        self.validMentions = set()

        # invokes functions
        self.loadReplacements(args.replacementsFile)
        corpus = self.parseCorpus(args.corpusPath, args.verbose)

    def parseCorpus(self, corpusPath, isVerbose):
        print("* parsing ECB corpus...", end='')

        # globally sets params
        corpus = Corpus()

        files = []
        for root, dirnames, filenames in os.walk(corpusPath):
            for filename in fnmatch.filter(filenames, '*.xml'):
                files.append(os.path.join(root, filename))

        globalSentenceNum = 0

    # loads replacement file
    def loadReplacements(self, replacementsFile):
        f = open(replacementsFile, 'r', encoding="utf-8")
        for line in f:
            tokens = line.rstrip().split(" ")
            self.replacements[tokens[0]] = tokens[1]
            self.replacementsList.append(tokens[0])
            self.replacementsSet.add(tokens[0])
        f.close()
