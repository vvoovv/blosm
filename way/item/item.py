

class Item:
    
    def __init__(self):
        # A cache to store evaluated values of attributes and maybe other stuff
        self._cache = {}

        self.street = None

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
    
    def setStyleBlockFromTop(self, streetStyle):
        """
        Calculates a specific style for the item out of the set of style definitions <streetStyle>
        Lookups the style for the item at the very top of style definitions.
        """
        
        className = self.__class__.__name__
        if className in streetStyle.styleBlocks:
            for _styleBlock in streetStyle.styleBlocks[className]:
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
    
    def getClass(self):
        return self.getStyleBlockAttr("cl")