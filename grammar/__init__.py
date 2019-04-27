

class Item:
    
    def __init__(self, markup, condition, defs, attrs):
        self.markup = markup
        self.condition = condition
        self.defs = defs
        self.attrs = attrs
        self.build()
    
    def build(self):
        if self.defs:
            defs = {}
            for item in self.defs:
                self.setParent(item)
                className = item.__class__.__name__
                if not className in defs:
                    defs[className] = []
                defs[className].append(item)
            self.defs = defs
        if self.markup:
            for item in self.markup:
                self.setParent(item)
    
    def setParent(self, item):
        item.parent = self


class Grammar(Item):
    """
    The top level element for the building style
    """
    
    def __init__(self, defs):
        super().__init__(None, None, defs, {})

    def setParent(self, item):
        item.parent = None


class Footprint(Item):
    
    def __init__(self, markup=None, condition=None, defs=None, **attrs):
        super().__init__(markup, condition, defs, attrs)


class Facade(Item):
    
    def __init__(self, markup=None, condition=None, defs=None, **attrs):
        super().__init__(markup, condition, defs, attrs)


class Roof(Item):
    
    flat = "flat"
    
    gabled = "gabled"
    
    def __init__(self, markup=None, condition=None, defs=None, **attrs):
        super().__init__(markup, condition, defs, attrs)


class Div(Item):
    
    def __init__(self, markup=None, condition=None, defs=None, **attrs):
        super().__init__(markup, condition, defs, attrs)


class Level(Item):
    
    def __init__(self, markup=None, condition=None, defs=None, **attrs):
        super().__init__(markup, condition, defs, attrs)


class Window(Item):
    
    def __init__(self, markup=None, condition=None, defs=None, **attrs):
        super().__init__(markup, condition, defs, attrs)


class WindowPanel:
    
    def __init__(self, relativeSize, openable):
        self.relativeSize = relativeSize
        self.openable = openable


class Balcony(Item):
    
    def __init__(self, markup=None, condition=None, defs=None, **attrs):
        super().__init__(markup, condition, defs, attrs)


class Door(Item):
    
    def __init__(self, markup=None, condition=None, defs=None, **attrs):
        super().__init__(markup, condition, defs, attrs)
        

class Chimney(Item):
    
    def __init__(self, markup=None, condition=None, defs=None, **attrs):
        super().__init__(markup, condition, defs, attrs)


class RoofSide(Item):

    def __init__(self, markup=None, condition=None, defs=None, **attrs):
        super().__init__(markup, condition, defs, attrs)


class Ridge(Item):
    
    def __init__(self, markup=None, condition=None, defs=None, **attrs):
        super().__init__(markup, condition, defs, attrs) 


class Dormer(Item):
    
    def __init__(self, markup=None, condition=None, defs=None, **attrs):
        super().__init__(markup, condition, defs, attrs) 


class Basement(Item):

    def __init__(self, markup=None, condition=None, defs=None, **attrs):
        super().__init__(markup, condition, defs, attrs)
        

def useFrom(itemId):
    return itemId