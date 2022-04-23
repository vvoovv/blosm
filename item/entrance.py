from . import Item


class Entrance(Item):
    
    # default values
    width = 1.2
    height = 2.
    marginLeft = 1.
    marginRight = 1.

    def getWidth(self):
        return Entrance.marginLeft + (self.getStyleBlockAttr("width") or Entrance.width) + Entrance.marginRight