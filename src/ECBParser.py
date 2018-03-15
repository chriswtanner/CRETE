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

        # filled in via loadReplacements()
        self.replacements = {}
        self.replacementsList = []
        self.replacementsSet = set() # for quicker indexing

        self.endPunctuation = set()
        self.endPunctuation.update(".", "!", "?")
        
        # invokes functions
        self.loadReplacements(args.replacementsFile)

    def parseCorpus(self, docToVerifiedSentences):

        print("* parsing ECB corpus...")
        numMentionsIgnored = 0
        corpus = Corpus()
        files = []
        for root, dirnames, filenames in os.walk(self.args.corpusPath):
            for filename in fnmatch.filter(filenames, '*.xml'):
                files.append(os.path.join(root, filename))

        globalSentenceNum = 0
        lastToken_id = -1
        for f in files:

            lm_idToMention = {} # only used to tmp store the mentions
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
                    #corpus.UIDToToken[curToken.UID] = curToken
                    #curDoc.UIDs.append(curToken.UID)
                    tmpDocTokenIDsToTokens[t_id] = curToken

                    firstToken = False
                    tmpDocTokens.append(curToken)
                    tokenNum = tokenNum + 1
                    curDoc.globalSentenceNums.add(globalSentenceNum)
                lastSentenceNum = sentenceNum
                lastTokenText = tokenText
                lastToken_id = t_id

            # if sentence ended with an atomic ":", let's change it to a "."
            if lastTokenText == ":":
                lastToken = tmpDocTokenIDsToTokens[lastToken_id]
                lastToken.text = "."
                tmpDocTokenIDsToTokens[lastToken_id] = lastToken
            elif lastTokenText not in self.endPunctuation:
                endToken = Token("-1", lastSentenceNum, globalSentenceNum, tokenNum, doc_id, -1, -1, ".")
                tmpDocTokens.append(endToken)

            globalSentenceNum = globalSentenceNum + 1

            # reads <markables> 1st time
            regex = r"<([\w]+) m_id=\"(\d+)?\".*?>(.*?)?</.*?>"
            markables = fileContents[fileContents.find("<Markables>")+11:fileContents.find("</Markables>")]
            it = tuple(re.finditer(regex, markables))
            for match in it:
                # gets the token IDs
                regex2 = r"<token_anchor t_id=\"(\d+)\".*?/>"
                it2 = tuple(re.finditer(regex2, match.group(3)))
                tmpCurrentMentionSpanIDs = []
                hasAllTokens = True
                for match2 in it2:
                    tokenID = match2.group(1)
                    tmpCurrentMentionSpanIDs.append(int(tokenID))
                    if tokenID not in tmpDocTokenIDsToTokens.keys():
                        hasAllTokens = False

            for t in tmpDocTokens:
                corpus.addToken(t)
                curDoc.tokens.append(t)
                corpus.UIDToToken[t.UID] = t
                
            # reads <markables> 2nd time
            regex = r"<([\w]+) m_id=\"(\d+)?\".*?>(.*?)?</.*?>"
            markables = fileContents[fileContents.find("<Markables>")+11:fileContents.find("</Markables>")]
            it = tuple(re.finditer(regex, markables))
            for match in it:
                isPred = False
                entityType = match.group(1)
                if "ACTION" in entityType:
                    isPred = True
                m_id = int(match.group(2))

                # gets the token IDs
                regex2 = r"<token_anchor t_id=\"(\d+)\".*?/>"
                it2 = tuple(re.finditer(regex2, match.group(3)))
                tmpTokens = []  # can remove after testing if our corpus matches HDDCRP's
                text = []
                hasAllTokens = True
                for match2 in it2:
                    tokenID = match2.group(1)
                    if tokenID in tmpDocTokenIDsToTokens.keys():
                        cur_token = tmpDocTokenIDsToTokens[tokenID]
                        tmpTokens.append(cur_token)
                        text.append(cur_token.text)
                    else:
                        hasAllTokens = False

                # we should only have incomplete Mentions for our hand-curated, sample corpus,
                # for we do not want to have all mentions, so we curtail the sentences of tokens
                if hasAllTokens:
                    curMention = Mention(dirHalf, dir_num, doc_id, tmpTokens, text, isPred, entityType)
                    lm_idToMention[m_id] = curMention
                    #corpus.addMention(curMention, "123")
            # reads <relations>
            relations = fileContents[fileContents.find("<Relations>"):fileContents.find("</Relations>")]
            regex = r"<CROSS_DOC_COREF.*?note=\"(.+?)\".*?>(.*?)?</.*?>"
            it = tuple(re.finditer(regex, relations))
            for match in it:
                REF = match.group(1)
                regex2 = r"<source m_id=\"(\d+)\".*?/>"
                it2 = tuple(re.finditer(regex2, match.group(2)))

                # only keep track of REFs for which we have found Mentions
                for match2 in it2:
                    m_id = int(match2.group(1))
                    if m_id not in lm_idToMention:
                        print("*** MISSING MENTION! EXITING")
                        exit(1)
                    else: #elif lm_idToMention[m_id].isPred:
                        foundMention = lm_idToMention[m_id]
                        
                        token0 = foundMention.tokens[0]
                        if token0.sentenceNum in docToVerifiedSentences[doc_id]:
                            corpus.addMention(foundMention, REF)
                        else:
                            numMentionsIgnored += 1
            corpus.addDocPointer(doc_id, curDoc)
        corpus.assignGlobalSentenceNums()
        print("numMentionsIgnored:", numMentionsIgnored)
        return corpus

	# loads replacement file
    def loadReplacements(self, replacementsFile):
        f = open(replacementsFile, 'r', encoding="utf-8")
        for line in f:
            tokens = line.rstrip().split(" ")
            self.replacements[tokens[0]] = tokens[1]
            self.replacementsList.append(tokens[0])
            self.replacementsSet.add(tokens[0])
        f.close()
