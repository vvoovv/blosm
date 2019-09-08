

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
        self._styleBlockCache.clear()
    
    def evaluateCondition(self, styleBlock):
        return not styleBlock.condition or styleBlock.condition(self)

    def getStyleBlockAttr(self, attr):
        attrs = self.styleBlock.attrs
        if not attr in attrs:
            return
        value, scope, isComplexValue = attrs.get(attr)
        if isComplexValue:
            styleBlockCache = self.getStyleBlockCache(scope)
            if attr in styleBlockCache:
                value = styleBlockCache[attr]
            else:
                value = value.value
                value.setData(self)
                value = value.value
                if value is None:
                    return
                # keep the entry for <attr> in the cache
                styleBlockCache[attr] = value
        return value
    
    def getStyleBlockCache(self, scope):
        return self._styleBlockCache
    
    def calculateStyling(self, style):
        """
        Lookups the style for the item at the very top of style definitions.
        It may perform other styling calculations
        
        Args:
            style (grammar.Grammar): a set of style definitions
        """
        
        className = self.__class__.__name__
        # Some items (Footprint, Facade, Roofside, Ridge, Roof) can be defined right at the top
        # of the style definition. We treat that case below in the code
        if className in style.styleBlocks:
            for _styleBlock in style.styleBlocks[className]:
                if self.evaluateCondition(_styleBlock):
                    self.styleBlock = _styleBlock
                    # the rest of the style blocks is ignored, so break the "for" cycle
                    break
            else:
                # no style block
                return
    
    def clone(self):
        item = self.__class__()
        # set item factory to be used inside <item.calculateMarkupDivision(..s)>
        item.itemFactory = self.itemFactory
        return item