from . import Item


class Footprint(Item):
    
    def init(self, part):
        self.part = part
        
    @classmethod
    def getItem(cls, itemFactory, part):
        item = itemFactory.getItem(cls)
        item.init(part)
        return item