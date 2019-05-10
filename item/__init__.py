from grammar.value import Value
from grammar.library import library


class Item:
    
    def __init__(self):
        # For example, a parent for a facade is a footprint
        self.parent = None
        # A style block (an instance of grammar.Item) that has markup definition.
        # If <self.markupStyle> is not <None>, it means that resulting style for
        # the element has a markup definition
        self.markupStyle = None
        # A style block (an instance of grammar.Item) that defines the style for the item
        # within a markup definition.
        # If <self.styleBlock> is not <None>, it means that the item is defined in a markup,
        # and <self.styleBlock> is the style block for the item in the markup definition
        self.styleBlock = None
        self.calculatedStyle = {}
        # The attribute below is used in calculation of <self.calculatedStyle>.
        # See the related code in <self.calculatedStyle(..)>
        self.auxMarkupStyle = {}
    
    def init(self):
        self.parent = None
        self.calculatedStyle.clear()
        self.auxMarkupStyle.clear()
        self.markupStyle = None
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
            # If <self.styleBlock> is not None, it means that the item is defined in the markup
            
            # evaluate the condition if it's available
            if styleBlock.condition and not styleBlock.condition(self):
                # empty style
                return
            
            self._calculateStyle(styleBlock)
        else:
            if className in style.styleBlocks:
                for _styleBlock in style.styleBlocks[className]:
                    if not _styleBlock.condition or _styleBlock.condition(self):
                        self._calculateStyle(_styleBlock)
                        break
                else:
                    # empty style
                    return
        # finalize the calculated style
        for attr in calculatedStyle:
            value = calculatedStyle[attr]
            if isinstance(value, Value):
                value = value.value
                if hasattr(value, "item"):
                    value.item = self
                calculatedStyle[attr] = value.value
    
    def _calculateStyle(self, styleBlock):
        markupStyle = styleBlock.markup

        # copy attributes from <styleBlock> to <self.calculatedStyle>
        for attr in styleBlock.attrs:
            self.calculatedStyle[attr] = styleBlock.attrs[attr]
        
        if styleBlock.use:
            # note the reverse iterator
            for defName in reversed(styleBlock.use):
                _styleBlock = library.getStyleBlock(
                    defName, self.__class__.__name__, styleBlock.styleId
                )
                if _styleBlock:
                    for attr in _styleBlock.attrs:
                        if not attr in self.calculatedStyle:
                            self.calculatedStyle[attr] = _styleBlock.attrs[attr]
                    if not markupStyle:
                        markupStyle = _styleBlock
        if markupStyle:
            self.markupStyle = markupStyle
    
    def clone(self):
        item = self.__class__()
        return item