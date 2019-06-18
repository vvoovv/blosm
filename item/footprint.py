from . import Item
from util.polygon import Polygon


class Footprint(Item):
    
    def __init__(self):
        super().__init__()
        self.polygon = Polygon()
        self.projections = []
        self.minProjIndex = 0
        self.maxProjIndex = 0
        self.polygonWidth = 0.
        self.lastLevelOffset = 0.
        # for example, church parts may not have levels
        self.hasLevels = True
    
    def init(self):
        super().init()
        self.lastLevelOffset = 0.
        # reset <self.polygon>
        self.polygon.allVerts = None
        self.projections.clear()
    
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