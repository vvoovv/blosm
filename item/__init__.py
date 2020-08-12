from grammar import perBuilding


class Item:
    
    def __init__(self):
        self.valid = True
        # for example, a parent for a facade is a footprint
        self.parent = None
        # a direct access to the footprint
        self.footprint = None
        # A style block (an instance of grammar.Item) that defines the style for the item
        # within a markup definition.
        # Typically a style block is defined in the markup definition, however it can be also defined
        # at the very top if the style definition for the item Footprint, Facade, RoofSide, Ridge, Roof
        self.styleBlock = None
        self.width = None
        self.relativeWidth = None
        self.hasFlexWidth = False
        # the following variable is used to cache a material id (e.g a string name) 
        self.materialId = None
        # Python dictionary to cache attributes from <self.styleBlock> that are derived
        # from <grammar.value.Value>
        self._styleBlockCache = {}
    
    def init(self):
        self.valid = True
        self.parent = None
        self.footprint = None
        self.styleBlock = None
        self.width = None
        self.relativeWidth = None
        self.hasFlexWidth = False
        self.materialId = None
        self._styleBlockCache.clear()
    
    def evaluateCondition(self, styleBlock):
        return not styleBlock.condition or styleBlock.condition(self)

    def getStyleBlockAttr(self, attr):
        attrs = self.styleBlock.attrs
        if not attr in attrs:
            return
        value, isComplexValue = attrs.get(attr)
        if isComplexValue:
            value = value.getValue(self)
        return value

    def getStyleBlockAttrDeep(self, attr):
        if self.styleBlock and attr in self.styleBlock.attrs:
            value = self.getStyleBlockAttr(attr)
        elif attr in self._styleBlockCache:
            value = self._styleBlockCache[attr]
        else:
            # try to get the attribute from <self.parent>
            value = self.parent.getStyleBlockAttrDeep(attr)
            self._styleBlockCache[attr] = value
        return value
    
    def getStyleBlockCache(self, scope):
        return self.building._cache if scope is perBuilding else self._styleBlockCache
    
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
    
    def getMargin(self):
        return 0.

    def getCladdingMaterial(self):
        return self.getStyleBlockAttrDeep("claddingMaterial")
    
    def getCladdingColor(self):
        return self.getStyleBlockAttrDeep("claddingColor")