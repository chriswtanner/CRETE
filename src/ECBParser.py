import sys
import re
import os
import fnmatch
from collections import defaultdict
from Corpus import Corpus
from Doc import Doc
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
        lastToken_id = -1
        for f in files:

            doc_id = f[f.rfind("/") + 1:]
            dir_num = int(doc_id.split("_")[0])
            extension = doc_id[doc_id.find("ecb"):]
            dirHalf = str(dir_num) + extension

            curDoc = Doc(doc_id)
            corpus.dirHalves[dirHalf].docs[doc_id] = curDoc
            tmpDocTokens = []
            tmpDocTokenIDsToTokens = {}

            # opens the xml file and makes needed replacements
            with open(f, 'r', encoding="utf-8") as myfile:
                fileContents = myfile.read().replace('\n', ' ')

                for badToken in self.replacementsList:  # self.replacementsSet:
                    fileContents = fileContents.replace(badToken, self.replacements[badToken])

            # reads <tokens>
            it = tuple(re.finditer(
                r"<token t\_id=\"(\d+)\" sentence=\"(\d+)\" number=\"(\d+)\".*?>(.*?)</(.*?)>", fileContents))
            lastSentenceNum = -1

            # numbers every token in each given sentence, starting at 1 (each sentence starts at 1)
            tokenNum = 0
            firstToken = True
            lastTokenText = ""
            for match in it:
                t_id = match.group(1)
                sentenceNum = int(match.group(2))
                hTokenNum = int(match.group(3))  # only used for matching w/ HDDCRP's files
                tokenText = match.group(4).lower().rstrip()
                # removes tokens that end in : (e.g., newspaper:) but leaves the atomic ":" alone
                if len(tokenText) > 1 and tokenText[-1] == ":":
                    tokenText = tokenText[:-1]
                if tokenText == "''":
                    tokenText = "\""
                elif tokenText == "''bagman\"":
                    tokenText = "\"bagman\""
                    print("* replaced bagman1")
                elif tokenText == "''bagman":
                    tokenText = "\"bagman"
                    print("* replaced bagman2")
    
                if sentenceNum > curDoc.highestSentenceNum:
                    curDoc.highestSentenceNum = sentenceNum
                
                if sentenceNum > 0 or "plus" not in doc_id:
                    hSentenceNum = sentenceNum
                    if "plus" in doc_id:
                        hSentenceNum = sentenceNum - 1

                    # we are starting a new sentence
                    if sentenceNum != lastSentenceNum:
                        # we are possibly ending the prev sentence
                        if not firstToken:
                            # if sentence ended with an atomic ":", let's change it to a "."
                            if lastTokenText == ":":
                                lastToken = tmpDocTokenIDsToTokens[lastToken_id]
                                lastToken.text = "."
                                tmpDocTokenIDsToTokens[lastToken_id] = lastToken
                            elif lastTokenText not in self.endPunctuation:
                                endToken = Token("-1", lastSentenceNum, globalSentenceNum, tokenNum, doc_id, hSentenceNum, hTokenNum, ".")
                                tmpDocTokens.append(endToken)

                            globalSentenceNum = globalSentenceNum + 1

                        tokenNum = 0
                    # adds token
                    curToken = Token(t_id, sentenceNum, globalSentenceNum, tokenNum, doc_id, hSentenceNum, hTokenNum, tokenText)
                    corpus.UIDToToken[curToken.UID] = curToken
                    curDoc.UIDs.append(curToken.UID)
                    tmpDocTokenIDsToTokens[t_id] = curToken

                    firstToken = False
                    tmpDocTokens.append(curToken)
                    tokenNum = tokenNum + 1
                    curDoc.globalSentenceNums.add(globalSentenceNum)
                lastSentenceNum = sentenceNum
                lastTokenText = tokenText
                lastToken_id = t_id

                NOTE: this is where i left off.  my RSI hurts
	# loads replacement file
    def loadReplacements(self, replacementsFile):
        f = open(replacementsFile, 'r', encoding="utf-8")
        for line in f:
            tokens = line.rstrip().split(" ")
            self.replacements[tokens[0]] = tokens[1]
            self.replacementsList.append(tokens[0])
            self.replacementsSet.add(tokens[0])
        f.close()
