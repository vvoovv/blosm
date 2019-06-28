from grammar.value import Value
from grammar.library import library


class Item:
    
    def __init__(self):
        # For example, a parent for a facade is a footprint
        self.parent = None
        # Markup definitin for the item. It's a Python list of style blocks
        self.markup = None
        # A style block (an instance of grammar.Item) that defines the style for the item
        # within a markup definition.
        # Typically a style block is defined in the markup definition, however it can be also defined
        # at the very top if the style definition for the item Footprint, Facade, RoofSide, Ridge, Roof
        self.styleBlock = None
        self.calculatedStyle = {}
    
    def init(self):
        self.parent = None
        self.calculatedStyle.clear()
        self.markup = None
        self.styleBlock = None

    def calculateStyle(self, style):
        """
        Calculates a specific style for the item out of the set of style definitions <styleDefs>
        
        Args:
            style (grammar.Grammar): a set of style definitions
        """
        className = self.__class__.__name__
        calculatedStyle = self.calculatedStyle
        
        styleBlock = self.styleBlock
        if styleBlock:
            # If <self.styleBlock> is not None, it means that the item is defined in the markup.
            # That's the case for the most of item. However, Footprint, Facade, RoofSide, Ridge, Roof can
            # defined right at the top of the style definition
            
            # evaluate the condition if it's available
            if styleBlock.condition and not styleBlock.condition(self):
                # the condition isn't satisfied
                # empty style
                return
            
            self._calculateStyle(styleBlock)
        else:
            # Some items (Footprint, Facade, Roofside, Ridge, Roof) can be defined right at the top
            # of the style definition. We treat that case below in the code
            if className in style.styleBlocks:
                for _styleBlock in style.styleBlocks[className]:
                    if not _styleBlock.condition or _styleBlock.condition(self):
                        self.styleBlock = _styleBlock
                        self._calculateStyle(_styleBlock)
                        # the rest of the style blocks is ignored, so break the "for" cycle
                        break
                else:
                    # empty style
                    return
        # finalize the calculated style
        for attr in calculatedStyle:
            value = calculatedStyle[attr]
            if isinstance(value, Value):
                value = value.value
                value.setData(self)
                calculatedStyle[attr] = value.value
    
    def _calculateStyle(self, styleBlock):
        markup = styleBlock.markup

        # copy attributes from <styleBlock> to <self.calculatedStyle>
        for attr in styleBlock.attrs:
            self.calculatedStyle[attr] = styleBlock.attrs[attr]
        
        if styleBlock.use:
            # note the reversed iterator
            for defName in reversed(styleBlock.use):
                _styleBlock = library.getStyleBlock(
                    defName, self.__class__.__name__, styleBlock.styleId
                )
                if _styleBlock:
                    for attr in _styleBlock.attrs:
                        if not attr in self.calculatedStyle:
                            self.calculatedStyle[attr] = _styleBlock.attrs[attr]
                    if not markup:
                        markup = _styleBlock.markup
        if markup:
            self.markup = markup
    
    def clone(self):
        item = self.__class__()
        return item