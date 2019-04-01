from . import Item


class Footprint(Item):
    
    def init(self, part):
        self.part = part
        self.calculatedStyle.clear()
        
    @classmethod
    def getItem(cls, itemFactory, part):
        item = itemFactory.getItem(cls)
        item.init(part)
        return item
    
    def attr(self, attr):
        return self.part.tags.get(attr)