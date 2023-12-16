class Roadway(object):
    ID = 1  # Must not start with zero to get unambiguous connector indices!
    def __init__(self,src,dst):
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
