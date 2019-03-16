from . import Item
from grammar.value import Value


class Footprint(Item):
    
    def init(self, part):
        self.part = part
        self.calculatedStyle.clear()
        
    @classmethod
    def getItem(cls, itemFactory, part):
        item = itemFactory.getItem(cls)
        item.init(part)
        return item
    
    def calculateStyle(self, styleDefs):
        style = self.calculatedStyle
        markupStyle = self.markupStyle
        if markupStyle:
            pass
        else:
            # scan the top level style block
            for styleBlock in styleDefs.defs[self.__class__.__name__]:
                for attr in styleBlock.attrs:
                    value = styleBlock.attrs[attr]
                    if isinstance(value, Value) and value.fromItem:
                            if attr in style:
                                # remember the previous value for that case
                                value.prev = style[attr]
                    style[attr] = value
            # finalize the calculated style
            for attr in tuple(style.keys()):
                value = style[attr]
                if isinstance(value, Value):
                    if value.fromItem:
                        value.value.item = self
                        _value = value.value.value
                        if _value is None:
                            # Value does not exist for the attribute,
                            # in that case we use the value from the previous style block
                            # stored before
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
                        # perform cleanup
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
    
    def attr(self, attr):
        return self.part.tags.get(attr)