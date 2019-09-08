from . import Item
from util.polygon import Polygon
from grammar import perBuilding
from action.volume.level_heights import LevelHeights


_facadeClassName = "Facade"


class Footprint(Item):
    
    def __init__(self):
        super().__init__()
        self.building = None
        self.polygon = Polygon()
        self.projections = []
        self.minProjIndex = 0
        self.maxProjIndex = 0
        self.polygonWidth = 0.
        self.lastLevelOffset = 0.
        # for example, church parts may not have levels
        self.hasLevels = True
        # A pointer to the Python list that contains style blocks for the facades generated out of the footprint;
        # see the code in the method <self.calculateStyling(..)> for the details
        self.facadeStyle = None
        self.facades = []
        self.levelHeights = LevelHeights(self)
    
    def init(self):
        super().init()
        self.building = None
        self.lastLevelOffset = 0.
        # reset <self.polygon>
        self.polygon.allVerts.clear()
        self.projections.clear()
        self.facadeStyle = None
        self.facades.clear()
        self.levelHeights.init()
    
    @classmethod
    def getItem(cls, itemFactory, element, building, styleBlock=None):
        # <styleBlock> is the style block within the markup definition,
        # if the footprint is generated through the markup definition
        item = itemFactory.getItem(cls)
        item.init()
        item.styleBlock = styleBlock
        item.element = element
        item.building = building
        return item
    
    def attr(self, attr):
        return self.element.tags.get(attr)
    
    def calculateFootprint(self):
        """
        Calculate footprint out of its <self.calculatedStyle>
        """
        return

    def calculateStyling(self, style):
        """
        Calculates a specific style for the item out of the set of style definitions <styleDefs>
        
        Args:
            style (grammar.Grammar): a set of style definitions
        """
        super().calculateStyling(style)
        
        # Find <Facade> style blocks in <markup> (actually in <self.styleBlock.styleBlocks>),
        # also try to find them at the very top of the style definitions
        self.facadeStyle = self.styleBlock.styleBlocks.get(
            _facadeClassName,
            style.styleBlocks.get(_facadeClassName)
        )
    
    def getStyleBlockCache(self, scope):
        return self.building._styleBlockCache if scope is perBuilding else self._styleBlockCache