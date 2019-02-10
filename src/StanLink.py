import pickle
class StanLink:
    def __init__(self, parent, child, relationship, dep_parse_type):
        self.parent = parent
        self.child = child
        self.relationship = relationship
        self.dep_parse_type = dep_parse_type
        
    def __str__(self):
        #return(str(self.parent))
        return(str(self.parent) + " --(" + str(self.relationship) + ")--> " + str(self.child))
