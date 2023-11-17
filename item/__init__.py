

class Item:
    
    def __init__(self, parent, footprint, facade, styleBlock):
        self.valid = True
        # for example, a parent for a facade is a footprint
        self.parent = parent
        # a direct access to the footprint
        self.footprint = footprint
        self.building = parent.building if parent else None
        self.facade = facade
        # div and level are containers
        self.isContainer = False
        
        # is the item located in the left corner of <facade>?
        self.cornerL = False
        # is the item located in the right corner of <facade>?
        self.cornerR = False
        
        # A style block (an instance of grammar.Item) that defines the style for the item
        # within a markup definition.
        # Typically a style block is defined in the markup definition, however it can be also defined
        # at the very top if the style definition for the item Footprint, Facade, RoofSide, Ridge, Roof
        self.styleBlock = styleBlock
        self.width = 0.
        self.height = 0.
        self.widthType = WidthType.flexible
        # the following variable is used to cache a material id (e.g a string name) 
        self.materialId = None
        # Python dictionary to cache attributes from <self.styleBlock> that are derived
        # from <grammar.value.Value>
        self._cache = {}
        self.assetInfo = None
    
    def evaluateCondition(self, styleBlock):
        return not styleBlock.condition or styleBlock.condition(self)

    def getStyleBlockAttr(self, attr):
        attrs = self.styleBlock.attrs
        if not attr in attrs:
            return
        value, isComplexValue = attrs.get(attr)
        if isComplexValue:
            if attr in self._cache:
                return self._cache[attr]
            value = value.getValue(self)
            # There is no need to perform a possibly complex calculations of the complex value once again,
            # so we simply store the resulting value in the cache
            self._cache[attr] = value
        return value

    def getStyleBlockAttrDeep(self, attr):
        if self.styleBlock and attr in self.styleBlock.attrs:
            return self.getStyleBlockAttr(attr)
        elif attr in self._cache:
            return self._cache[attr]
        else:
            # try to get the attribute from <self.parent>
            value = self.parent.getStyleBlockAttrDeep(attr)
            self._cache[attr] = value
            return value
    
    def getCache(self, scope):
        return self.building.renderInfo._cache if scope is perBuilding else self._cache
    
    def getItemRenderer(self, itemRenderers):
        """
        Get a renderer for the item contained in the markup.
        """
        return itemRenderers[self.__class__.__name__]
    
    def clone(self):
        item = self.__class__()
        # set item factory to be used inside <item.calculateMarkupDivision(..s)>
        item.itemFactory = self.itemFactory
        return item

    def getCladdingMaterial(self):
        return self.getStyleBlockAttrDeep("claddingMaterial")
    
    def getCladdingColor(self):
        return self.getStyleBlockAttrDeep("claddingColor")

    def getWidth(self, globalRenderer):
        return globalRenderer.itemRenderers[self.__class__.__name__].getTileWidthM(self)
    
    def getBuildingPart(self):
        return self.buildingPart

    def getClass(self):
        return self.getStyleBlockAttrDeep("cl")
    
    def getWidthType(self):
        return self.getStyleBlockAttr("widthType")
    
    def postStyleSet(self):
        if self.styleBlock.buildingCount:
            for counter in self.styleBlock.buildingCount:
                if counter in self.building.renderInfo._cache:
                    self.building.renderInfo._cache[counter] += 1
                else:
                    self.building.renderInfo._cache[counter] = 1


from grammar.building import perBuilding
from grammar.width_type import WidthType