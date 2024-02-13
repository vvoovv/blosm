from .item import Item


class Street(Item):
    
    ID = 1  # Must not start with zero to get unambiguous connector indices!
    
    def __init__(self, src, dst):
        super().__init__()
        self.id = Street.ID
        Street.ID += 1

        self._src = src
        self._dst = dst
        
        self.head = None
        self.tail = None

        self.pred = None
        self.succ = None

    @property
    def src(self):
        return self._src
    
    @property
    def dst(self):
        return self._dst
    
    def getCategory(self):
        return self.head.getCategory()
    
    def setStyle(self, style):
        self.style = style
        
        self.setStyleBlockFromTop(style)
        
        if self.head is self.tail:
            self.head.setStyleBlockFromTop(style)
        else:
            item = self.start
            while item is not self.tail:
                item.setStyleBlockFromTop(style)
                item = item.succ
            # set the style to <self.end>
            item.setStyleBlockFromTop(style)
    
    def getName(self):
        return self.head.getName()