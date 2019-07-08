from . import Item


class Facade(Item):
    """
    Represents a building facade.
    It's typically composed of one or more faces (in the most cases rectangular ones)
    """
    
    def __init__(self):
        pass
    
    def init(self):
        self._markupWidth = None
        self.faces = []
        self.normal = None
        self.type = ("front", "back", "side")
        self.neighborL = None
        self.neighborR = None
        self.neighborT = None
        self.neighborB = None

    @classmethod
    def getItem(cls, itemFactory, parent, indices, width, heightLeft, heightRightOffset):
        item = itemFactory.getItem(cls)
        item.parent = parent
        item.indices = indices
        item.width = width
        # assign uv-coordinates
        item.uvs = ( (0., 0.), (width, heightRightOffset), (width, heightLeft), (0., heightLeft) )
        item.init()
        return item
    
    def checkWidth(self, styleBlock):
        """
        Check if the facade has a markup definition and
        the total width if the markup elements does not exceed the width of the facade
        """
        markup = styleBlock.markup
        if not markup:
            return True
        return self.getMarkupWidth() <= self.width
    
    def getMarkupWidth(self):
        if not self._markupWidth:
            self._markupWidth = self.width/2.
        return self._markupWidth
    
    @property
    def front(self):
        return True
    
    @property
    def back(self):
        return True