
from collections import defaultdict
from DirHalf import DirHalf
class Corpus:
    def __init__(self):

        # NOTE: about the naming convention, XUID is the int label
        # to address Mentions, whether they are ECB or HDDCRP
        # but when they are stored in data structures that are specifically/
        # exclusively intended for 1 or the other, we call them MUID or HMUID
        self.numCorpusTokens = 0
        self.corpusTokens = []
        self.corpusTokensToCorpusIndex = {}
        self.UIDToToken = {}
        self.curXUID = 0 # counter used for both ECB and HDDCRP, hence the X

        self.XUIDToMention = {} # used for *all* mentions, regardless if ECB, HDDCRP, Stan

        # only used for ECB Corpus mentions
        self.ecb_mentions = []
        self.MUIDToMention = {}
        self.MUIDToREF = {}
        self.refToMUIDs = defaultdict(set)
        self.docSentToMentions = defaultdict(list)

        # only used for HDDCRP mentions (we have no REF info)
        self.hddcrp_mentions = []
        self.HMUIDToMention = {}
        self.docSentToHMentions = defaultdict(list)

        # only used for Stan mentions (we have no REF info)
        self.stan_mentions = []
        self.SUIDToMention = {}
        self.docSentToSMentions = defaultdict(list)

        self.dirs = set()
        self.dirHalves = defaultdict(DirHalf) # same as what's contained across all dirs

        # Docs are stored within dirHalves,
        # but also accessible from here, by its name
        self.doc_idToDocs = {}

        # to easily parse the original sentence which contains each Mention
        self.globalSentenceNumToTokens = defaultdict(list)

    # ensures we've created ECB/HDDCRP/Stan Mentions all from teh same sentences
    def checkMentions(self):
        allKeys = set()
        allKeys.update(self.docSentToMentions.keys())
        allKeys.update(self.docSentToHMentions.keys())
        allKeys.update(self.docSentToSMentions.keys())
        print("# keys:",len(allKeys))
        docToSent = defaultdict(set)
        for doc, _ in allKeys:
            docToSent[doc].add(_)
        for doc in docToSent.keys():
            #print(doc)
            for sent in docToSent[doc]:
                necb = len(self.docSentToMentions[(doc, sent)])
                nh = len(self.docSentToHMentions[(doc, sent)])
                ns = len(self.docSentToSMentions[(doc, sent)])
                if necb == 0 and ns > 0:
                    print("doc:",doc, "sent:",sent)
                    print(necb,",",nh,",",ns)
    # adds a Token to the corpus
    def addToken(self, token):
        self.corpusTokens.append(token)
        self.corpusTokensToCorpusIndex[token] = self.numCorpusTokens
        self.numCorpusTokens = self.numCorpusTokens + 1

    # adds a Stan Mention to the corpus (no REF info)
    def addStanMention(self, mention):
        (doc_id, sentenceNum) = self.getDocAndSentence(mention)
        mention.setXUID(self.curXUID)  # updates the mention w/ MUID info
        self.stan_mentions.append(mention)
        self.SUIDToMention[self.curXUID] = mention
        self.dirHalves[mention.dirHalf].assignStanMention(self.curXUID, mention.doc_id)
        self.docSentToSMentions[(doc_id, sentenceNum)].append(mention)
        self.XUIDToMention[self.curXUID] = mention
        self.curXUID += 1

    # adds a HDDCRP Mention to the corpus (no REF info)
    def addHDDCRPMention(self, mention):
        (doc_id, sentenceNum) = self.getDocAndSentence(mention)
        mention.setXUID(self.curXUID)  # updates the mention w/ MUID info
        self.hddcrp_mentions.append(mention)
        self.HMUIDToMention[self.curXUID] = mention
        self.dirHalves[mention.dirHalf].assignHDDCRPMention(self.curXUID, mention.doc_id)
        self.docSentToHMentions[(doc_id, sentenceNum)].append(mention)
        self.XUIDToMention[self.curXUID] = mention
        self.curXUID += 1

    # adds a Mention to the corpus
    def addMention(self, mention, REF):
        (doc_id, sentenceNum) = self.getDocAndSentence(mention)

        # updates the mention w/ REF and MUID info
        mention.setXUID(self.curXUID)
        mention.setREF(REF)

        self.ecb_mentions.append(mention)
        self.MUIDToMention[self.curXUID] = mention
        self.MUIDToREF[self.curXUID] = REF
        self.refToMUIDs[REF].add(self.curXUID)
        self.dirHalves[mention.dirHalf].assignECBMention(self.curXUID, mention.doc_id, REF)
        self.docSentToMentions[(doc_id, sentenceNum)].append(mention)
        self.XUIDToMention[self.curXUID] = mention
        self.curXUID += 1

    def assignGlobalSentenceNums(self):
        for t in self.corpusTokens:
            self.globalSentenceNumToTokens[int(t.globalSentenceNum)].append(t)

    # adds access to the Docs via their names
    def addDocPointer(self, doc_id, curDoc):
        self.doc_idToDocs[doc_id] = curDoc

    def getDocAndSentence(self, mention):
        doc_id = mention.doc_id
        sentenceNum = mention.tokens[0].sentenceNum
        return (doc_id, sentenceNum)
