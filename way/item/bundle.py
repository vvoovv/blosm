from .item import Item


class Bundle(Item):
    
    ID = 1  # Must not start with zero to get unambiguous connector indices!
    
    def __init__(self):
        super().__init__()
        self.id = Bundle.ID
        Bundle.ID += 1

        self._pred = None
        self._succ = None

        self.streetsHead = []
        self.streetsTail = []

    @property
    def pred(self):
        return self._pred
    
    @property
    def succ(self):
        return self._succ
    
    @pred.setter
    def pred(self,val):
        self._pred = val
    
    @succ.setter
    def succ(self):
        return self._succ