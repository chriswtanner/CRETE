import pickle
import operator
from Mention import Mention
from collections import defaultdict
class ECBHelper:
    def __init__(self, args):
        self.args = args
        self.corpus = None # should be passed-in

        # data splits
        self.trainingDirs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 19, 20, 21, 22]
        self.devDirs = [23, 24, 25]
        self.testingDirs = [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]

        print("trainingDirs:", str(self.trainingDirs))
        print("devDirs:", str(self.devDirs))
        print("testingDirs:", str(self.testingDirs))

        # filled in via createHDDCRPMentions() (maps text UID
        # lines from CoNLL file to the created HDDCRP Mention)
        self.UIDToHMUID = defaultdict(list)

        # filled in via createStanMentions() (maps text UID
        # lines from CoNLL file to the created Stan Mention)
        # where StanMention is a regular Mention w/ regular Tokens
        # (which contain lists of StanTokens)
        self.UIDToSUID = defaultdict(list)
        self.docToVerifiedSentences = self.loadVerifiedSentences(args.verifiedSentencesFile)

    def addECBCorpus(self, corpus):
        self.corpus = corpus

    # ECB+ only makes guarantees that the following sentences are correctly annotated
    def loadVerifiedSentences(self, sentFile):
        docToVerifiedSentences = defaultdict(set)
        f = open(sentFile, 'r')
        f.readline()
        for line in f:
            tokens = line.rstrip().split(",")
            doc_id = tokens[0] + "_" + tokens[1] + ".xml"
            sentNum = tokens[2]
            docToVerifiedSentences[doc_id].add(int(sentNum))
        f.close()
        return docToVerifiedSentences

    # pickles the StanTokens
    def saveStanTokens(self):
        UIDToStanTokens = {}  # UID -> StanToken[]
        for t in self.corpus.corpusTokens:
            UIDToStanTokens[t.UID] = t.stanTokens
        print("* writing out", len(UIDToStanTokens), "UIDs' StanTokens")
        pickle_out = open(self.args.stanTokensFile, 'wb')
        pickle.dump(UIDToStanTokens, pickle_out)

    # reads in the pickled StanTokens
    def loadStanTokens(self):
        pickle_in = open(self.args.stanTokensFile, 'rb')
        UIDToStanTokens = pickle.load(pickle_in)
        for uid in UIDToStanTokens:
            self.corpus.UIDToToken[uid].stanTokens = UIDToStanTokens[uid]
        print("* loaded", len(UIDToStanTokens), "UIDs' StanTokens")
        print("corpus has #UIDS:", str(len(self.corpus.UIDToToken)))

    def addStanfordAnnotations(self, stanfordParser):
        stanDocSet = set()
        for doc_id in stanfordParser.docToSentenceTokens.keys():
            stanDocSet.add(doc_id)
        # adds stan links on a per doc basis
        for doc_id in stanDocSet:
            stanTokens = []  # builds list of stanford tokens
            for sent_num in sorted(stanfordParser.docToSentenceTokens[doc_id].keys()):
                for token_num in stanfordParser.docToSentenceTokens[doc_id][sent_num]:
                    sToken = stanfordParser.docToSentenceTokens[doc_id][sent_num][token_num]
                    if sToken.isRoot == False:
                        stanTokens.append(sToken)

            # for readability, make a new var
            ourTokens = self.corpus.doc_idToDocs[doc_id].tokens
            j = 0
            i = 0
            while i < len(ourTokens):
                if j >= len(stanTokens):
                    if i == len(ourTokens) - 1 and stanTokens[-1].text == "...":
                        tmp = [stanTokens[-1]]
                        ourTokens[i].addStanTokens(tmp)
                        break
                    else:
                        print("ran out of stan tokens")
                        exit(1)

                stanToken = stanTokens[j]
                ourToken = ourTokens[i]

                curStanTokens = [stanToken]
                curOurTokens = [ourToken]

                stan = stanToken.text
                ours = ourToken.text

                # pre-processing fixes since the replacements file can't handle spaces
                if stan == "''":
                    stan = "\""
                elif stan == "2 1/2":
                    stan = "2 1/2"
                elif stan == "3 1/2":
                    stan = "3 1/2"
                elif stan == "877 268 9324":
                    stan = "8772689324"
                elif stan == "0845 125 2222":
                    stan = "08451252222"
                elif stan == "0800 555 111":
                    stan = "0800555111"
                elif stan == "0800 555111":
                    stan = "0800555111"
                elif stan == "0845 125 222":
                    stan = "0845125222"

                # get the words to equal lengths first
                while len(ours) != len(stan):
                    while len(ours) > len(stan):
                        #print("\tstan length is shorter:", str(ours)," vs:",str(stan)," stanlength:",str(len(stan)))
                        if j+1 < len(stanTokens):

                            if stanTokens[j+1].text == "''":
                                stanTokens[j+1].text = "\""
                                print("TRYING TO FIX THE UPCOMING STAN TOKEN!")
                            stan += stanTokens[j+1].text

                            curStanTokens.append(stanTokens[j+1])
                            if stan == "71/2":
                                stan = "7 ½"
                            elif stan == "31/2":
                                stan = "3½"
                            j += 1
                        else:
                            print("\tran out of stanTokens")
                            exit(1)

                    while len(ours) < len(stan):
                        if i+1 < len(ourTokens):
                            ours += ourTokens[i+1].text
                            curOurTokens.append(ourTokens[i+1])
                            if ours == "31/2":
                                ours = "3 1/2"
                            elif ours == "21/2":
                                ours = "2 1/2"
                            elif ours == "31/2-inch":
                                ours = "3 1/2-inch"
                            elif ours == "3 1/2":
                                ours = "3 1/2"
                            i += 1
                        else:
                            print("\tran out of ourTokens")
                            exit(1)

                if ours != stan:
                    print("\tMISMATCH: [", str(ours), "] [", str(stan), "]")
                    exit(1)
                else:  # texts are identical, so let's set the stanTokens
                    for t in curOurTokens:
                        t.addStanTokens(curStanTokens)

                j += 1
                i += 1

            # ensures every Token in the doc has been assigned at least 1 StanToken
            for t in self.corpus.doc_idToDocs[doc_id].tokens:
                if len(t.stanTokens) == 0:
                    print("Token:", str(t), " never linked w/ a stanToken!")
                    exit(1)
        print("we've successfully added stanford links to every single token within our", str(len(self.corpus.doc_idToDocs)), "docs")

    def createStanMentions(self):
        print("in createStanMentions()")
        last_ner = ""
        toBeMentions = [] # list of StanToken Lists
        curTokens = []
        tokenToNER = {} # only used for creating a 'type' field in Mention
        for each_token in self.corpus.corpusTokens:
            if each_token.sentenceNum not in self.docToVerifiedSentences[each_token.doc_id]:
                continue
            
            cur_ner = ""
            for st in each_token.stanTokens:
                if st.ner != "O":
                    cur_ner = st.ner
            
            if cur_ner != "":
                tokenToNER[each_token] = cur_ner

            if last_ner == "" and cur_ner != "":  # start of a new StanMention
                curTokens.append(each_token)
            else:
                if cur_ner != last_ner: # if we end curMention
                    toBeMentions.append(curTokens)
                    curTokens = []  # new clean slate
                if cur_ner != "": # either keep it going or start a new one
                    curTokens.append(each_token)
            last_ner = cur_ner
        if len(curTokens) > 0: # had some left, let's make a mention
            toBeMentions.append(curTokens)
        
        for m in toBeMentions:
            doc_id = m[0].doc_id 
            dir_num = int(doc_id.split("_")[0])
            extension = doc_id[doc_id.find("ecb"):]
            dirHalf = str(dir_num) + extension
            text = []
            SUIDs = []
            for t in m:
                sentenceNum = m[0].sentenceNum
                hTokenNum = t.hTokenNum
                SUID = str(doc_id) + ";" + str(sentenceNum) + ";" + str(hTokenNum)
                SUIDs.append(SUID)
                if self.corpus.UIDToToken[SUID] != t:
                    print("ERROR: Token mismatch")
                text.append(t.text)

            curMention = Mention(dirHalf, dir_num, doc_id, m, text, False, tokenToNER[m[0]])
            self.corpus.addStanMention(curMention)
            for SUID in SUIDs:
                self.UIDToSUID[SUID].append(curMention.XUID)

        '''
        tokenToMention = {}
        stanStats = defaultdict(lambda: defaultdict(int))
        for m in self.corpus.ecb_mentions:
            for t in m.tokens:
                tokenToMention[t] = m

        numStanNER = 0
        numAligned = 0
        numNotAligned = 0
        numZeros = 0
        TN = 0
        for each_token in self.corpus.corpusTokens:
            if each_token.sentenceNum not in self.docToVerifiedSentences[each_token.doc_id]:
                continue
            dir_num = int(each_token.doc_id.split("_")[0])
            if dir_num not in self.testingDirs:
                continue
        #numMentions = 0
        #for m in self.corpus.ecb_mentions:
        #    if m.dir_num not in self.testingDirs:
        #        continue
        #    numMentions += 1
            #for each_token in m.tokens:
            mentionsFound = set()
            for st in each_token.stanTokens:
                if st.ner != "O":
                    if each_token in tokenToMention:
                        mentionsFound.add(tokenToMention[each_token])
                        numAligned += 1
                    else:
                        mentionsFound.add("n/a")
                        numNotAligned += 1
                    numStanNER += 1
                else:
                    if each_token not in tokenToMention:
                        TN += 1
                    numZeros += 1
            if len(mentionsFound) == 1:  # all or nothing
                _ = mentionsFound.pop()
                if _ == "n/a":
                    stanStats["n/a"]["n/a"] += 1
                else:
                    stanStats["all"][_.mentionType] += 1
            else:
                for m in mentionsFound:
                    if m != "n/a":
                        stanStats["partial"][m.mentionType] += 1
        

        for d in stanStats:
            sorted_x = sorted(stanStats[d].items(), key=operator.itemgetter(1), reverse=True)
            for (k, v) in sorted_x:
                print(d, str(k), v)
        print("numStanNER", numStanNER)
        print("numAligned", numAligned)
        print("numNotAligned", numNotAligned)
        print("numZeros", numZeros)
        print("TN:", TN)
        #print("numMentions", numMentions)
        '''
    def createHDDCRPMentions(self, hddcrp_mentions):
        for i in range(len(hddcrp_mentions)):
            HUIDs = hddcrp_mentions[i]
            tokens = []
            text = []
            if len(HUIDs) == 0:
                print("ERROR: empty HDDCRP Mention")
                exit(1)

            for HUID in HUIDs:
                HUID_minus_text = HUID[0:HUID.rfind(";")]

                doc_id = HUID.split(";")[0]
                dir_num = int(doc_id.split("_")[0])
                extension = doc_id[doc_id.find("ecb"):]
                dirHalf = str(dir_num) + extension

                token = self.corpus.UIDToToken[HUID_minus_text]
                tokens.append(token)
                text.append(token.text)
                if HUID.split(";")[-1] != token.text:
                    print("WARNING: TEXT MISMATCH: HUID:", HUID, "ECB token:", token)
            
            if tokens[0].sentenceNum in self.docToVerifiedSentences[tokens[0].doc_id]:
                curMention = Mention(dirHalf, dir_num, doc_id, tokens, text, True, "unknown")
                self.corpus.addHDDCRPMention(curMention)
                for HUID in HUIDs: # UID == HUID always, because we checked above (in the warning check)
                    self.UIDToHMUID[HUID].append(curMention.XUID)

    def printCorpusStats(self):
        mentionStats = defaultdict(lambda: defaultdict(int))
        for m in self.corpus.ecb_mentions:
            if m.dir_num in self.trainingDirs:
                mentionStats["train"][m.mentionType] += 1
            elif m.dir_num in self.devDirs:
                mentionStats["dev"][m.mentionType] += 1
            elif m.dir_num in self.testingDirs:
                mentionStats["test"][m.mentionType] += 1
            else:
                print("* ERROR: wrong dir")
                exit(1)
        
        print("[ CORPUS STATS ]")
        print("\t# dirHalves:", str(len(self.corpus.dirHalves)))
        print("\t# docs:", len(self.corpus.doc_idToDocs))
        print("\t# REFs:", len(self.corpus.refToMUIDs.keys()))
        numS = 0
        numC = 0
        for ref in self.corpus.refToMUIDs:
            if len(self.corpus.refToMUIDs[ref]) == 1:
                numS +=1 
            else:
                numC += 1
        print("\t\t# singletons:",numS,"# nons",numC)
        print("\t# ECB Mentions (total):", len(self.corpus.ecb_mentions))
        
        tokenToMention = {}
        hddcrpStats = defaultdict(lambda: defaultdict(int))
        for m in self.corpus.ecb_mentions:
            for t in m.tokens:
                tokenToMention[t] = m
        
        for hm in self.corpus.hddcrp_mentions:
            mentionsFound = set()
            for t in hm.tokens:
                if t in tokenToMention:
                    mentionsFound.add(tokenToMention[t])
                else:
                    mentionsFound.add("n/a")
            if len(mentionsFound) == 1: # all or nothing
                _ = mentionsFound.pop()
                
                if _ == "n/a":
                    hddcrpStats["n/a"]["n/a"] += 1
                else:
                    hddcrpStats["all"][_.mentionType] += 1
            else:
                for m in mentionsFound:
                    if m != "n/a":
                        hddcrpStats["partial"][m.mentionType] += 1
        '''
        for d in hddcrpStats:
            sorted_x = sorted(hddcrpStats[d].items(), key=operator.itemgetter(1), reverse=True)
            for (k, v) in sorted_x:
                print(d, str(k), v)

        print("\t# HDDCRP Mentions (total):", len(self.corpus.hddcrp_mentions))
        
        for d in mentionStats:
            sorted_x = sorted(mentionStats[d].items(), key=operator.itemgetter(1), reverse=True)
            for (k,v) in sorted_x:
                print(d,str(k),v)
        
        print("\t# ECB Mentions (train): NON-ACTION:", mentionStats[("train", 0)], "ACTION:", mentionStats[("train",1)])
        print("\t# ECB Mentions (dev): NON-ACTION:", mentionStats[("dev", 0)], "ACTION:", mentionStats[("dev", 1)])
        print("\t# ECB Mentions (test): NON-ACTION:", mentionStats[("test", 0)], "events:", mentionStats[("test", 1)])
        '''
        print("\t# ECB Tokens:", len(self.corpus.corpusTokens))
