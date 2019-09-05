from . import Item


class Balcony(Item):
    
    # default values
    width = 2.4
    marginLeft = 1.
    marginRight = 1.
    
    @classmethod
    def getItem(cls, itemFactory, parent, styleBlock):
        item = itemFactory.getItem(cls)
        item.init()
        item.parent = parent
        item.footprint = parent.footprint
        item.building = parent.building
        item.styleBlock = styleBlock
        return item
    
    def getWidth(self):
        return Balcony.marginLeft + (self.getStyleBlockAttr("width") or Balcony.width) + Balcony.marginRight