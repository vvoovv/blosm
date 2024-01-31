from .item import Item


class Street(Item):
    
    ID = 1  # Must not start with zero to get unambiguous connector indices!
    
    def __init__(self, src, dst):
        super().__init__()
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
        return self.start.getMainCategory()
    
    def setStyle(self, style):
        self.style = style
        self.setStyleBlockFromTop(style)
        
        # set a style block for the items of the street (sections, crosswalks, etc)
        self.start.setStyleBlockFromTop(style)