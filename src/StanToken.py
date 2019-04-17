import pickle
from collections import defaultdict
class StanToken:
    def __init__(self, isRoot, sentenceNum, tokenNum, text, lemma, startIndex, endIndex, pos, ner):
        self.isRoot = isRoot
        self.sentenceNum = sentenceNum
        self.tokenNum = tokenNum
        self.text = text
        self.lemma = lemma
        self.startIndex = startIndex
        self.endIndex = endIndex
        self.pos = pos
        self.ner = ner

        # StanLinks
        self.parentLinks = defaultdict(list)
        self.childLinks = defaultdict(list)

    def addParent(self, dep_parse_type, parentLink):
        self.parentLinks[dep_parse_type].append(parentLink)

    def addChild(self, dep_parse_type, childLink):
        self.childLinks[dep_parse_type].append(childLink)

    def __str__(self):
        return str("stan:" + self.text)
        '''
        parents = ""
        for pl in self.parentLinks:
            parents += "\n\t   ---" + str(pl.relationship) + "-->" + str(pl.parent.text)
        children = ""
        for cl in self.childLinks:
            children += "\n\t   ---" + str(cl.relationship) + "-->" + str(cl.child.text)

        if len(self.parentLinks) > 0:
            parent = str(self.parentLinks[0].parent)
        return(self.text)
        '''
        #return("STAN TEXT: [" + str(self.text) + "]" + "; LEMMA:" + str(self.lemma) + "; POS:" + str(self.pos) + "; NER:" + str(self.ner)) # + "\n\tparents:" + parents + "\n\tchildren:" + children)
