from grammar.value import Value


def _finalizeCalculatedStyle(style, item):
    # finalize the calculated style
    for attr in tuple(style.keys()):
        value = style[attr]
        if isinstance(value, Value):
            if value.fromItem:
                value.value.item = item
                _value = value.value.value
                if _value is None:
                    # Value does not exist for the attribute,
                    # in that case we use the value from the previous style block stored before
                    v = value
                    while True:
                        prev = v.prev
                        if prev:
                            _value = prev.value.value
                            if _value is None:
                                v = prev
                            else:
                                break
                        else:
                            # unset the attribute
                            del style[attr]
                            break
                # perform cleanup: set the attribute <prev> to None
                v = value
                while True:
                    prev = v.prev
                    if prev:
                        v.prev = None
                        v = prev
                    else:
                        break
            else:
                _value = value.value.value
            if not _value is None:
                style[attr] = _value


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

    def calculateStyle(self, styleDefs):
        """
        Calculates a specific style for the item out of the set of style definitions <styleDefs>
        
        Args:
            styleDefs (grammar.Grammar): a set of style definitions
        """
        style = self.calculatedStyle
        # The variable <markupStyle> is used to have the reference to
        # a style block that has the markup definition that has overridden
        # the preceding markups
        markupStyle = None
        
        if self.parent:
            # If <self.parent> is not None, then <self.styleBlock> is not None as well,
            # i.e. the item is defined in the markup
            styleBlock = self.styleBlock
            
            # Traverse the hierarchy of styles from the down to the top and calculate
            # the resulting style
            parent = self.parent
            auxMarkupStyle = self.auxMarkupStyle
            while parent:
                for styleBlock in parent.defs[self.__class__.__name__]:
                    # The basic idea behind this for-cycle is that
                    # we do not override an attribute in <style> since
                    # we go from the bottom to the top and attributes from higher
                    # style blocks are overriden by attributes from lower style blocks.
                    # The only exception is when the style block <parent> contains style blocks,
                    # that have the same attribute. In that case the attribute that comes later,
                    # overrides the attribute that comes earlier. <self.auxMarkupStyle> is used
                    # to keep track of that particular case.
                    
                    # evaluate the condition if it's available
                    if not styleBlock.condition or styleBlock.condition(self):
                        for attr in styleBlock.attrs:
                            _notAttrInStyle = not attr in style
                            if _notAttrInStyle or attr in auxMarkupStyle:
                                if _notAttrInStyle:
                                    # keep the attribute temporarily in <auxMarkupStyle>
                                    auxMarkupStyle[attr] = 1
                                value = styleBlock.attrs[attr]
                                if isinstance(value, Value) and value.fromItem:
                                    if attr in style:
                                        # remember the previous value for that case
                                        value.prev = style[attr]
                                style[attr] = value
                auxMarkupStyle.clear()
                parent = parent.parent
            
            if styleBlock.markup:
                # the style for the item contains markup definition
                markupStyle = styleBlock
        else:
            # scan the top level style block
            for styleBlock in styleDefs.defs[self.__class__.__name__]:
                # evaluate the condition if it's available
                if not styleBlock.condition or styleBlock.condition(self):
                    if styleBlock.markup:
                        markupStyle = styleBlock
                    for attr in styleBlock.attrs:
                        value = styleBlock.attrs[attr]
                        if isinstance(value, Value) and value.fromItem:
                            if attr in style:
                                # remember the previous value for that case
                                value.prev = style[attr]
                        style[attr] = value
            
        _finalizeCalculatedStyle(style, self)
        
        if markupStyle:
            self.markupStyle = markupStyle
    
    def clone(self):
        item = self.__class__()
        return item