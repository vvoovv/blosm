from . import Item
from util.polygon import Polygon
from action.volume.level_heights import LevelHeights


_facadeClassName = "Facade"


class Footprint(Item):
    
    def __init__(self, entranceAttr):
        """
        Args:
            entranceAttr (str): A datta attribute for the entrance to look up in the polygon vertices.
                Typically, it's "entrance" from OSM, that designates an etrance to the building.
        """
        super().__init__()
        self.entranceAttr = entranceAttr
        self.building = None
        # all style blocks that define the style for the building
        self.buildingStyle = None
        self.polygon = Polygon()
        self.projections = []
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
        
        if entranceAttr:
            # indices of original <self.polygon> vertices containg <entranceAttr>
            self.entranceVertexIndices = []
            # Element index of <self.entranceAttrVertexMapping> is the index of
            # a original polygon vertex
            # Element is None if the related vertex doesn't contain <entranceAttr> OR
            # the element index if the vertex is preserved after <self.polygon.removeStraightAngle(..)> OR
            # index of the nearest remaing vertex if the original vertex was skippped
            # after <self.polygon.removeStraightAngle(..)>
            self.entranceAttrVertexMapping = []
    
    def init(self):
        super().init()
        self.building = None
        self.buildingStyle = None
        self.numRoofLevels = 0
        self.minLevel = 0
        self.lastLevelOffset = 0.
        self.projections.clear()
        self.facadeStyle = None
        self.facades.clear()
        self.levelHeights.init()

    def clone(self):
        item = self.__class__(self.entranceAttr)
        # set item factory to be used inside <item.calculateMarkupDivision(..s)>
        item.itemFactory = self.itemFactory
        return item
    
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
    
    def processFacades(self, data):
        # Definitions for the terms used below:
        # Original polygon: <self.polygon> before <self.removeStraightAngle(..)>
        
        # First, we check if facades contains entrances
        facades = self.facades
        polygon = self.polygon
        indices = polygon.indices
        nodes = tuple( self.element.getNodes(data) )
        numNodes = len(nodes)
        
        facadesWithEntrance = []
        prevFacade = facades[-1]
        _vertIndex = indices[prevFacade.edgeIndex]
        for facade in facades:
            if not facade.outer:
                # The special case for the inner facades.
                # For now we consider that they are side facades
                facade.side = True
                continue
            # vertex index in the original polygon
            vertIndex = indices[facade.edgeIndex]
            # If <_vertIndex> and <vertIndex> are adjacent in the original polygon,
            # <difference> is equal to 1, if some vertices were skipped by
            # <self.removeStraightAngle(..)>, <difference> is more than 1.
            if polygon.reversed:
                difference = numNodes + _vertIndex - vertIndex\
                    if _vertIndex < vertIndex else\
                    _vertIndex - vertIndex
            else:
                difference = numNodes - _vertIndex + vertIndex\
                    if vertIndex < _vertIndex else\
                    vertIndex - _vertIndex
            while difference != 1:
                # getting the next vertex in the original polygon
                _vertIndex = (_vertIndex-1 if _vertIndex else numNodes-1)\
                    if polygon.reversed else\
                    (_vertIndex+1) % numNodes
                if nodes[_vertIndex].tags and nodes[_vertIndex].tags.get(self.entranceAttr):
                    if prevFacade.numEntrances:
                        prevFacade.numEntrances += 1
                    else:
                        prevFacade.numEntrances = 1
                        facadesWithEntrance.append(prevFacade)
                        # consider for now that a facade with a door is a front facade
                        prevFacade.front = True
                difference -= 1
            if nodes[vertIndex].tags and nodes[vertIndex].tags.get(self.entranceAttr):
                facade.numEntrances = 1
                facadesWithEntrance.append(facade)
                # consider for now that a facade with a door is a front facade
                facade.front = True
            prevFacade = facade
            _vertIndex = vertIndex
        
        if not facadesWithEntrance:
            # Consider that the facade corresponding to the longest edge of <self.polygon>
            # has an entrance
            maxEdgeIndex = polygon.maxEdgeIndex
            if facades[0].edgeIndex:
                # treat the case if facade count in <facades> from an index greater than zero
                maxEdgeIndex = polygon.n - facades[0].edgeIndex + maxEdgeIndex\
                    if maxEdgeIndex < facades[0].edgeIndex  else\
                    maxEdgeIndex-facades[0].edgeIndex
            
            facades[maxEdgeIndex].numEntrances = 1
            # consider for now that a facade with a door is a front facade
            facades[maxEdgeIndex].front = True
            facadesWithEntrance.append(facades[maxEdgeIndex])
        # Next, find back facades. We consider that a back facade has a normal opposite to
        # the main front facade. If there are more than two front facades, we take the longest
        # of them as the main front facade.
        backNormal = -(
            facadesWithEntrance[0].normal\
            if len(facadesWithEntrance) == 1 else\
            max(facadesWithEntrance, key = lambda facade: facade.width).normal
        )
        
        # Next, mark the remaining facades as back or side
        for facade in facades:
            if facade.outer and not facade.front:
                if facade.normal.dot(backNormal) > 0.98:
                    facade.back = True
                else:
                    facade.side = True