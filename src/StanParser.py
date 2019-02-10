import fnmatch, os
from StanToken import StanToken
from StanLink import StanLink
from collections import defaultdict
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
class StanParser:
    def __init__(self, args, corpus):
        print("init stan parser")
        self.args = args
        self.corpus = corpus

        self.dependency_parses = {"basic-dependencies", "collapsed-dependencies", \
            "collapsed-ccprocessed-dependencies", "enhanced-dependencies", \
            "enhanced-plus-plus-dependencies"}

        self.replacements = {} # for O(1) look-up and the mapping
        self.replacementsList = [] # must maintain order
        self.relationshipTypes = set()

        self.docToSentenceTokens = {}

        self.loadReplacements(args.replacementsFile)
        self.parseDir(args.stanOutputDir)

    def parseDir(self, stanOutputDir):
        print("parse dir:", stanOutputDir)
        files = []
        for root, _, filenames in os.walk(stanOutputDir):
            for filename in fnmatch.filter(filenames, '*.xml'):
                files.append(os.path.join(root, filename))
        for f in files:
            doc_id = str(f[f.rfind("/")+1:])
            if doc_id in self.corpus.doc_idToDocs:
                # format: [sentenceNum] -> {[tokenNum] -> StanToken}
                self.docToSentenceTokens[doc_id] = self.parseFile(f)
                #exit(1)
    # (1) reads stanford's output, saves it
    # (2) aligns it w/ our sentence tokens
    def parseFile(self, inputFile):
        print("* parsing file:", inputFile)
        sentenceTokens = defaultdict(lambda: defaultdict(int))
        tree = ET.ElementTree(file=inputFile)
        root = tree.getroot()

        document = root[0]
        
        doc_id, sentences, _ = document
        '''
        print("doc:", inputFile)
        for elem in corefs:
            print("el:",elem)
            for section in elem:
                print("sec:",section)
                for s2 in section:
                    print("s2:",s2)
        '''         
            

        self.relationshipTypes = set()
        for elem in sentences:  # tree.iter(tag='sentence'):
            sentenceNum = int(elem.attrib["id"])
            for section in elem:
                # process every token for the given sentence
                if section.tag == "tokens":
                    # constructs a ROOT StanToken, which represents the NULL ROOT of the DependencyParse
                    rootToken = StanToken(True, sentenceNum, 0, "ROOT", "ROOT", -1, -1, "-", "-")
                    sentenceTokens[sentenceNum][0] = rootToken
                    for token in section:
                        tokenNum = int(token.attrib["id"])
                        word = ""
                        lemma = ""
                        startIndex = -1
                        endIndex = -1
                        pos = ""
                        ner = ""
                        for item in token:
                            if item.tag == "word":
                                word = item.text
                                if word == "''":
                                    word = "\""
                            elif item.tag == "lemma":
                                lemma = item.text
                            elif item.tag == "CharacterOffsetBegin":
                                startIndex = item.text
                            elif item.tag == "CharacterOffsetEnd":
                                startIndex = item.text
                            elif item.tag == "POS":
                                pos = item.text
                            elif item.tag == "NER":
                                ner = item.text

                        for badToken in self.replacementsList:
                            if badToken == word:
                                word = self.replacements[badToken]
                            if badToken in word:
                                word = word.replace(badToken, self.replacements[badToken])

                        # constructs and saves the StanToken
                        stanToken = StanToken(False, sentenceNum, tokenNum, word, lemma, startIndex, endIndex, pos, ner)
                        if word.lower() != stanToken.text.lower():
                            print(word, " ->", stanToken)
                        sentenceTokens[sentenceNum][tokenNum] = stanToken

                elif section.tag == "dependencies" and section.attrib["type"] in self.dependency_parses:
                    dep_parse_type = section.attrib["type"]

                    # iterates over all dependencies for the given sentence
                    for dep in section:

                        parent, child = dep
                        relationship = dep.attrib["type"]
                        self.relationshipTypes.add(relationship)
                        parentToken = sentenceTokens[sentenceNum][int(parent.attrib["idx"])]
                        childToken = sentenceTokens[sentenceNum][int(child.attrib["idx"])]

                        # ensures correctness from Stanford
                        if parentToken.text != parent.text:
                            for badToken in self.replacementsList: # self.replacementsSet:
                                if badToken in parent.text:
                                    parent.text = parent.text.replace(
                                        badToken, self.replacements[badToken])

                        if childToken.text != child.text:
                            for badToken in self.replacementsList: # self.replacementsSet:
                                if badToken in child.text:
                                    child.text = child.text.replace(badToken, self.replacements[badToken])

                        if parentToken.text != parent.text or childToken.text != child.text:
                            print("STAN's DEPENDENCY TEXT MISMATCHES WITH STAN'S TOKENS")
                            print("1", str(parentToken.text))
                            print("2", str(parent.text))
                            print("3", str(childToken.text))
                            print("4", str(child.text))
                            exit(1)

                        # creates stanford link
                        curLink = StanLink(
                            parentToken, childToken, relationship, dep_parse_type)
                        #print("making new stanlink:", curLink, "bw:", parentToken, "and", childToken)
                        parentToken.addChild(dep_parse_type, curLink)
                        childToken.addParent(dep_parse_type, curLink)
        return sentenceTokens

    # replaces the ill-formed characters/words
    def loadReplacements(self, replacementsFile):
        f = open(replacementsFile, 'r', encoding="utf-8")
        for line in f:
            tokens = line.rstrip().split(" ")
            self.replacements[tokens[0]] = tokens[1]
            self.replacementsList.append(tokens[0])
        f.close()
