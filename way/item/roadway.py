from .item import Item


class Roadway(Item):
    
    ID = 1  # Must not start with zero to get unambiguous connector indices!
    
    def __init__(self,src,dst):
        super().__init__()
        self.id = Roadway.ID
        Roadway.ID += 1

        self._src = src
        self._dst = dst

    @property
    def src(self):
        return self._src
    
    @property
    def dst(self):
        return self._dst
