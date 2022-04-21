from . import Item


class Door(Item):
    
    # default values
    width = 1.2
    height = 2.
    marginLeft = 1.
    marginRight = 1.

    def getWidth(self):
        return Door.marginLeft + (self.getStyleBlockAttr("width") or Door.width) + Door.marginRight