from . import Item


class Door(Item):
    
    # default values
    width = 1.2
    height = 2.
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
        return Door.marginLeft + (self.getStyleBlockAttr("width") or Door.width) + Door.marginRight