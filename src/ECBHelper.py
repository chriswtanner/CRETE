from Mention import Mention
class ECBHelper:
    def __init__(self, args, ecb_corpus):
        self.args = args
        self.ecb_corpus = ecb_corpus

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

                token = self.ecb_corpus.UIDToToken[HUID_minus_text]
                tokens.append(token)
                text.append(token.text)
                if HUID.split(";")[-1] != token.text:
                    print("WARNING: TEXT MISMATCH: HUID:", HUID, "ECB token:", token)
            curMention = Mention(dirHalf, dir_num, doc_id, tokens, text, True, "unknown")
            self.ecb_corpus.addHDDCRPMention(curMention)
            self.HUIDToHMUID[HUID] = curMention.XUID
