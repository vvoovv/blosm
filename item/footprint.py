from . import Item
from util.polygon import Polygon


class Footprint(Item):
    
    def __init__(self):
        super().__init__()
        self.polygon = Polygon()
    
    def init(self):
        super().init()
        # reset <self.polygon>
        self.polygon.allVerts = None
    
    @classmethod
    def getItem(cls, itemFactory, element, styleBlock=None):
        # <styleBlock> is the style block within the markup definition,
        # if the footprint is generated through the markup definition
        item = itemFactory.getItem(cls)
        item.styleBlock = styleBlock
        item.init()
        item.element = element
        return item
    
    def attr(self, attr):
        return self.element.tags.get(attr)
    
    def calculateFootprint(self):
        """
        Calculate footprint out of its <self.calculatedStyle>
        """
        return