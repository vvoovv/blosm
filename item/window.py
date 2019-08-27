from . import Item


class Window(Item):
    
    # default values
    width = 1.2
    height = 1.8
    marginLeft = 1.
    marginRight = 1.
    
    @classmethod
    def getItem(cls, itemFactory, parent, styleBlock):
        item = itemFactory.getItem(cls)
        item.init()
        item.parent = parent
        item.building = parent.building
        item.styleBlock = styleBlock
        return item
    
    def getWidth(self):
        return Window.marginLeft + (self.getStyleBlockAttr("width") or Window.width) + Window.marginRight