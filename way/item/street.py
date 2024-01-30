

class Street:
    ID = 1  # Must not start with zero to get unambiguous connector indices!
    
    def __init__(self, src, dst):
        self.id = Street.ID
        Street.ID += 1

        self._src = src
        self._dst = dst
        
        self.start = None
        self.end = None

    @property
    def src(self):
        return self._src
    
    @property
    def dst(self):
        return self._dst
    
    def getMainCategory(self):
        return self.start.getMainCategory
    
    def setStyle(self, style):
        self.style = style