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
    
    def append(self, item):
        if self.head is None:
            self.head = item
            self.tail = self.head
            return
         
        self.tail.succ = item
        self.tail.succ.pred = self.tail
        self.tail = self.tail.succ

    def iterItems(self):
        ptr = self.head
        while ptr is not None:
            yield ptr
            ptr = ptr.succ
            
    def getCategory(self):
        return self.head.getCategory()
    
    def setStyleForItems(self, style):
        if self.head is self.tail:
            self.head.setStyleBlockFromTop(style)
        else:
            item = self.head
            while item is not self.tail:
                item.setStyleBlockFromTop(style)
                item = item.succ
            # set the style to <self.end>
            item.setStyleBlockFromTop(style)
    
    def getName(self):
        return self.head.getName()