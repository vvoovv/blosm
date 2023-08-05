from building import BldgPolygon
from . import Item
from defs.base.level_heights import LevelHeights


_facadeClassName = "Facade"


class Footprint(Item):
    
    def __init__(self, bldgPart, building, styleBlock=None):
        # <styleBlock> is the style block within the markup definition,
        # if the footprint is generated through the markup definition
        super().__init__(None, None, None, styleBlock)
        self.bldgPart = bldgPart
        if bldgPart:
            self.polygon = bldgPart.polygon
            self.element = bldgPart.element
        self.building = building
        # all style blocks that define the style for the building
        self.buildingStyle = None
        self.projections = None
        self.minProjIndex = 0
        self.maxProjIndex = 0
        self.polygonWidth = 0.
        self.lastLevelOffset = 0.
        # for example, church parts may not have levels
        self.hasLevels = True
        self.numRoofLevels = 0
        self.minLevel = 0
        # A pointer to the Python list that contains style blocks for the facades generated out of the footprint;
        # see the code in the method <self.calculateStyling(..)> for the details
        self.facadeStyle = None
        self.facades = []
        self.levelHeights = LevelHeights(self)
        # <self.rectangularWalls> defines if ALL walls in the volume extruded from the footprint are rectangles
        self.rectangularWalls = False
    
    def attr(self, attr):
        return self.element.tags.get(attr)
    
    def getStyleBlockAttrDeep(self, attr):
        return self.getStyleBlockAttr(attr)
    
    def calculateFootprint(self):
        """
        Calculate footprint out of its <self.calculatedStyle>
        """
        return

    def calculateStyling(self):
        """
        Calculates a specific style for the item out of the set of style definitions <self.buildingStyle>
        Lookups the style for the item at the very top of style definitions.
        It may perform other styling calculations
        """
        
        className = self.__class__.__name__
        buildingStyle = self.buildingStyle
        # Some items (Footprint, Facade, Roofside, Ridge, Roof) can be defined right at the top
        # of the style definition. We treat that case below in the code
        if className in buildingStyle.styleBlocks:
            for _styleBlock in buildingStyle.styleBlocks[className]:
                # Temporarily set <self.styleBlock> to <_styleBlock> to use attributes
                # from <_styleBlock> in the condition evaluation
                self.styleBlock = _styleBlock
                if self.evaluateCondition(_styleBlock):
                    # the rest of the style blocks is ignored, so break the "for" cycle
                    break
                else:
                    # cleanup
                    self.styleBlock = None
            else:
                # no style block
                return
        
        # Find <Facade> style blocks in <markup> (actually in <self.styleBlock.styleBlocks>),
        # also try to find them at the very top of the style definitions
        self.facadeStyle = self.styleBlock.styleBlocks.get(
            _facadeClassName,
            self.buildingStyle.styleBlocks.get(_facadeClassName)
        )
    
    def createPolygon(self, lineString, manager):
        """
        Create and set <self.polygon>. Used only for some exotic cases.
        """
        self.polygon = BldgPolygon(lineString, manager, self.building)