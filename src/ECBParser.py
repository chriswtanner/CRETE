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

        self.docToGlobalSentenceNums = defaultdict(set)
        self.docToTokens = defaultdict(list)
        self.docToREFs = defaultdict(list)

        # dirHalf's (for cross-doc work)
        self.dirHalfREFToDMs = defaultdict(lambda: defaultdict(set))
        self.dirHalfToHMs = defaultdict(list)

        # key: (doc_id,ref_id) -> [(doc_id1,m_id1), ... (doc_id3,m_id3)]
        self.docREFsToDMs = defaultdict(list)
        self.docToDMs = defaultdict(list)
        self.docToUIDs = defaultdict(list)
        self.UIDToMentions = {}
        self.UIDToToken = {}

        # same tokens as corpusTokens, just made into lists according
        # to each doc.  (1 doc = 1 list of tokens); used for printing corpus to .txt file
        self.docTokens = []

        self.typeToGlobalID = {}
        self.globalIDsToType = {}
        self.corpusTypeIDs = []

        self.docToHighestSentenceNum = defaultdict(
            int)  # TMP -- just for creating
        # an aligned goldTruth from HDDCRP
        self.globalSentenceNumToTokens = defaultdict(
            list)  # added so that we can
        # easily parse the original sentence which contains each Mention

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
