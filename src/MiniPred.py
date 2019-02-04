# stores event gold, event pred, entity gold for a given xuid-pair
class MiniPred:
    def __init__(self, xuid_pair, event_gold, ent_gold):
        self.xuid_pair = xuid_pair
        self.event_gold = event_gold
        self.ent_gold = ent_gold
        self.event_pred = None
    def set_event_pred(self, event_pred):
        self.event_pred = event_pred

    def __str__(self):
        return "XUIDs:" + str(self.xuid_pair) + " event_gold:" + str(self.event_gold) \
            + " event_pred:" + str(self.event_pred) + " ent_gold:" + str(self.ent_gold)