import pickle
from Mention import Mention
from collections import defaultdict
class ECBHelper:
    def __init__(self, args, corpus):
        self.args = args
        self.corpus = corpus

        # data splits
        self.trainingDirs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 19, 20, 21, 22]
        self.devDirs = [23, 24, 25]
        self.testingDirs = [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]

        print("trainingDirs:", str(self.trainingDirs))
        print("devDirs:", str(self.devDirs))
        print("testingDirs:", str(self.testingDirs))

        # filled in via createHDDCRPMentions() (maps text UID
        # lines from CoNLL file to the created HDDCRP Mention)
        self.HUIDToHMUID = {}

    def saveStanTokens(self):
        UIDToStanTokens = {}  # UID -> StanToken[]
        for t in self.corpus.corpusTokens:
            UIDToStanTokens[t.UID] = t.stanTokens
        print("* writing out", len(UIDToStanTokens), "UIDs' StanTokens")
        pickle_out = open(self.args.stanTokensFile, 'wb')
        pickle.dump(UIDToStanTokens, pickle_out)

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
            #for sent_num in sorted(self.corpus.docToGlobalSentenceNums[doc_id]):
            #    for token in self.corpus.globalSentenceNumToTokens[sent_num]:
            #        ourTokens.append(token)

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
                            #print("\tstan is now:", str(stan))
                        else:
                            print("\tran out of stanTokens")
                            exit(1)

                    while len(ours) < len(stan):
                        #print("\tour length is shorter:",str(ours),"vs:",str(stan),"stanlength:",str(len(stan)))
                        if i+1 < len(ourTokens):
                            ours += ourTokens[i+1].text
                            curOurTokens.append(ourTokens[i+1])
                            if ours == "31/2":
                                ours = "3 1/2"
                            elif ours == "21/2":
                                #print("converted to: 2 1/2")
                                ours = "2 1/2"
                            elif ours == "31/2-inch":
                                ours = "3 1/2-inch"
                            elif ours == "3 1/2":
                                ours = "3 1/2"
                            i += 1
                            #print("\tours is now:", str(ours))
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
            curMention = Mention(dirHalf, dir_num, doc_id, tokens, text, True, "unknown")
            self.corpus.addHDDCRPMention(curMention)
            self.HUIDToHMUID[HUID] = curMention.XUID

    def printCorpusStats(self):
        trainEnts = 0
        trainEvents = 0
        devEnts = 0
        devEvents = 0
        testEnts = 0
        testEvents = 0
        mentionStats = defaultdict(int)
        for m in self.corpus.ecb_mentions:
            if m.dir_num in self.trainingDirs:
                mentionStats[("train",m.isPred)] += 1
            elif m.dir_num in self.devDirs:
                mentionStats[("dev", m.isPred)] += 1
            elif m.dir_num in self.testingDirs:
                mentionStats[("test", m.isPred)] += 1
            else:
                print("* ERROR: wrong dir")
                exit(1)
        
        print("[ CORPUS STATS ]")
        print("\t# dirHalves:", str(len(self.corpus.dirHalves)))
        print("\t# docs:", len(self.corpus.doc_idToDocs))
        print("\t# REFs:", len(self.corpus.refToMUIDs.keys()))
        print("\t# ECB Mentions (total):", len(self.corpus.ecb_mentions))
        print("\t# ECB Mentions (train): entities:", mentionStats[("train", 0)], "events:", mentionStats[("train",1)])
        print("\t# ECB Mentions (dev): entities:", mentionStats[("dev", 0)], "events:", mentionStats[("dev",1)])
        print("\t# ECB Mentions (test): entities:", mentionStats[("test", 0)], "events:", mentionStats[("test", 1)])
        print("\t# ECB Tokens:", len(self.corpus.corpusTokens))
