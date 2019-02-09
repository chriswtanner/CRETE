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
    def __init__(self, args, helper):
        self.write_stanford_input = True
        self.onlyEvents = False
        self.printCorpusTokens = False
        self.args = args
        self.helper = helper

        # filled in via loadReplacements()
        self.replacements = {}
        self.replacementsList = []
        self.replacementsSet = set() # for quicker indexing

        self.endPunctuation = set()
        self.endPunctuation.update(".", "!", "?")
        
        
        # invokes functions
        self.loadReplacements(args.replacementsFile)

    def parseCorpus(self, docToVerifiedSentences):

        # maps the long, original REF names to a small, more readable REF ID
        REFToUREF = {}
        UREF = 0

        print("* parsing ECB corpus:", self.args.corpusPath)
        numMentionsIgnored = 0
        corpus = Corpus()
        files = []

        filteredTrainingDirs = self.helper.trainingDirs[0:self.args.devDir]
        print("filteredTrainingDirs:", filteredTrainingDirs)
        for root, _, filenames in os.walk(self.args.corpusPath):
            for filename in fnmatch.filter(filenames, '*.xml'):
                f = os.path.join(root, filename)
                doc_id = f[f.rfind("/") + 1:]
                dir_num = int(doc_id.split("_")[0])
                if dir_num in self.helper.trainingDirs and dir_num not in filteredTrainingDirs:
                    continue
                files.append(os.path.join(root, filename))

        globalSentenceNum = 0
        lastToken_id = -1
        intraCount = 0
        
        # used for keeping track of how many mentions were pronouns
        had_pronoun = 0
        not_had_pronoun = 0
        for f in sorted(files):
            lm_idToMention = {} # only used to tmp store the mentions
            removed_m_ids = set() # keeps track of the mentions that had pronouns and we removed (if we care to remove them)
            doc_id = f[f.rfind("/") + 1:]
            dir_num = int(doc_id.split("_")[0])
            extension = doc_id[doc_id.find("ecb"):]
            dirHalf = str(dir_num) + extension

            curDoc = Doc(doc_id)
            corpus.ECBDirs[dir_num].docs[doc_id] = curDoc
            corpus.dirHalves[dirHalf].docs[doc_id] = curDoc
            tmpDocTokens = []
            tmpDocTokenIDsToTokens = {}

            # opens the xml file and makes needed replacements
            input_file = open(f, 'r', encoding="utf-8")
            #with open(f, 'r', encoding="utf-8") as myfile:
            fileContents = input_file.read().replace('\n', ' ')            
            for badToken in self.replacementsList:  # self.replacementsSet:
                fileContents = fileContents.replace(badToken, self.replacements[badToken])

            # reads <tokens>
            it = tuple(re.finditer(r"<token t\_id=\"(\d+)\" sentence=\"(\d+)\" number=\"(\d+)\".*?>(.*?)</(.*?)>", fileContents))
            lastSentenceNum = -1

            if self.write_stanford_input:
                tmp_line_to_stanford_input = defaultdict(list)

            # numbers every token in each given sentence, starting at 1 (each sentence starts at 1)
            tokenNum = 0
            firstToken = True
            lastTokenText = ""
            for match in it:
                t_id = match.group(1)
                sentenceNum = int(match.group(2))
                hTokenNum = int(match.group(3))  # only used for matching w/ HDDCRP's files
                
                #tokenText = match.group(4).rstrip() # should be used if i'll write out hte corpus for Stan
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
                    
                    # writes Stanford_input
                    if self.write_stanford_input:
                        tmp_line_to_stanford_input[int(sentenceNum)].append(match.group(4).rstrip())

                    hSentenceNum = sentenceNum
                    if "plus" in doc_id:
                        hSentenceNum = sentenceNum - 1

                    # TMP
                    '''
                    if sentenceNum not in tmpSentenceNums:
                        tmpSentenceNums.append(sentenceNum)
                    '''

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

            if self.write_stanford_input:
                tmpFOUT = open("../data/stanford_in/"+doc_id, "w")
                for sent_num in sorted(tmp_line_to_stanford_input.keys()):
                    tmpFOUT.write(" ".join(tmp_line_to_stanford_input[sent_num]) + "\n")
                tmpFOUT.close()

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

                #if doc_id == "31_3ecbplus.xml":
                #    print("t:",t)
                
            # reads <markables> 2nd time
            regex = r"<([\w]+) m_id=\"(\d+)?\".*?>(.*?)?</.*?>"
            markables = fileContents[fileContents.find("<Markables>")+11:fileContents.find("</Markables>")]
            it = tuple(re.finditer(regex, markables))
            for match in it:
                isPred = False
                mentionType = match.group(1)
                if "ACTION" in mentionType:
                    isPred = True
                m_id = int(match.group(2))

                # gets the token IDs
                regex2 = r"<token_anchor t_id=\"(\d+)\".*?/>"
                it2 = tuple(re.finditer(regex2, match.group(3)))
                tmpTokens = []
                text = []
                hasAllTokens = True

                has_pronoun = False
                for match2 in it2:
                    tokenID = match2.group(1)
                    if tokenID in tmpDocTokenIDsToTokens.keys():
                        cur_token = tmpDocTokenIDsToTokens[tokenID]
                        tmpTokens.append(cur_token)
                        text.append(cur_token.text)

                    else:
                        hasAllTokens = False

                # only process Mentions if they adhere to our preferences of using pronouns or not
                # determines if it has a pronoun or not (and if we care)
                if len(text) == 1:
                    if text[0] in self.helper.pronouns:
                        has_pronoun = True

                if has_pronoun:
                    had_pronoun += 1
                else:
                    not_had_pronoun += 1

                # possibly add the mention 
                use_pronoun = False
                if isPred:
                    use_pronoun = self.helper.event_pronouns
                else:
                    use_pronoun = self.helper.entity_pronouns
                
                use_mention = True
                if not use_pronoun and has_pronoun:
                    use_mention = False
                    #print("* not constructing mention:", text)
                    removed_m_ids.add(m_id)

                # we should only have incomplete Mentions for our hand-curated, sample corpus,
                # for we do not want to have all mentions, so we curtail the sentences of tokens
                if hasAllTokens and use_mention:
                    curMention = Mention(dirHalf, dir_num, doc_id, tmpTokens, text, isPred, mentionType)
                    lm_idToMention[m_id] = curMention
                    #tmpSentenceNumToMentions[tmpTokens[0].sentenceNum].append(curMention)
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
                        
                        if  m_id not in removed_m_ids:
                            print("*** MISSING MENTION! EXITING 1")
                            exit(1)
                    else: #elif lm_idToMention[m_id].isPred:
                        foundMention = lm_idToMention[m_id]
                        if self.onlyEvents and not foundMention.isPred:
                            continue
                        token0 = foundMention.tokens[0]

                        if self.args.onlyValidSentences and token0.sentenceNum not in docToVerifiedSentences[doc_id]:
                            numMentionsIgnored += 1
                            continue
                        else:
                            corpus.addMention(foundMention, REF)

            if self.args.addIntraDocs:
                regex = r"<INTRA_DOC_COREF.*?>(.*?)?</.*?>"
                it = tuple(re.finditer(regex, relations))
                for match in it:
                    regex2 = r"<source m_id=\"(\d+)\".*?/>"
                    it2 = tuple(re.finditer(regex2, match.group(1)))
                    # only keep track of REFs for which we have found Mentions
                    for match2 in it2:
                        m_id = int(match2.group(1))
                        if m_id not in lm_idToMention:
                            print("*** MISSING MENTION! EXITING 2")
                            exit(1)
                        else:
                            foundMention = lm_idToMention[m_id]
                            if self.onlyEvents and not foundMention.isPred:
                                continue
                            token0 = foundMention.tokens[0]

                            if self.args.onlyValidSentences and token0.sentenceNum not in docToVerifiedSentences[doc_id]:
                                numMentionsIgnored += 1
                                continue
                            else:
                                corpus.addMention(foundMention, "INTRA"+str(intraCount))
                                intraCount += 1
            corpus.addDocPointer(doc_id, curDoc)

            # optionally displays annotations (Mentions clearly designated w/ unique REF IDs#)
            if self.printCorpusTokens:
                print("\n------------------\ndoc:",doc_id,"\n------------------")
                sent_num = -1
                oline = ""
                lastMentions = set()
                for t in curDoc.tokens:
                    if t.sentenceNum != sent_num and sent_num != -1:
                        sent_num = t.sentenceNum
                        print(oline)
                        oline = ""
                    added = False
                    removed = False
                    urefToAdd = -1
                    entOrEventToAdd = ""
                    for m in t.mentions:
                        if m not in lastMentions:
                            if m.REF in REFToUREF.keys():
                                urefToAdd = REFToUREF[m.REF]
                            else:
                                urefToAdd = UREF
                                REFToUREF[m.REF] = UREF
                                UREF += 1
                            if m.isPred:
                                entOrEventToAdd = "v"
                            else:
                                entOrEventToAdd = "ent"
                            added = True
                    
                    if len(lastMentions) > 0:
                        for m in lastMentions:
                            if m not in t.mentions:
                                removed = True
                    if removed:
                        oline += "] "
                    if added:
                        if len(oline) > 0 and oline[-1] != " ":
                            oline += " "
                        oline += str(entOrEventToAdd) + str(urefToAdd) + "["
                    if len(oline) > 0 and oline[-1] != " " and oline[-1] != "[":
                        oline += " "
                    oline += str(t.text)
                    lastMentions = t.mentions
                print(oline)
        corpus.assignGlobalSentenceNums()
        print("numMentionsIgnored:", numMentionsIgnored)
        print("# ECB mentions created:", len(corpus.ecb_mentions))
        print("# ECB+ tokens:", len(corpus.corpusTokens))
        print("# mentions that had_pronoun:", had_pronoun)
        print("# mentions that did not had_pronoun:", not_had_pronoun)
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
