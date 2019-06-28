from .library import library


class Item:
    
    def __init__(self, defName, use, markup, condition, attrs):
        self.defName = defName
        self.use = use
        self.markup = markup
        self.condition = condition
        self.attrs = attrs
        self.init()
    
    def init(self):
        if self.markup:
            for styleBlock in self.markup:
                self.setParent(styleBlock)
    
    def setParent(self, styleBlock):
        styleBlock.parent = self
    
    def build(self, styleId):
        if self.use:
            self.styleId = styleId
        if self.markup:
            self.buildMarkup(styleId)
    
    def buildMarkup(self, styleId):
        for styleBlock in self.markup:
            styleBlock.build(styleId)


class Grammar:
    """
    The top level element for the building style
    """
    
    def __init__(self, styleBlocks):
        # A placeholder the style blocks without a name, located on the very top of the hierachy.
        # Besides a Footprint, those style blocks can be of the types Facade,
        # RoofSide, Ridge, i.e. the items generated through volume extrusion
        self.styleBlocks = {}
        self.build(styleBlocks)
    
    def build(self, inputStyleBlocks):
        styleBlocks = self.styleBlocks
        
        styleId = library.getStyleId()
        
        for styleBlock in inputStyleBlocks:
            if styleBlock.defName:
                library.addStyleBlock(styleBlock.defName, styleBlock, styleId)
            else:
                className = styleBlock.__class__.__name__
                if not className in styleBlocks:
                    styleBlocks[className] = []
                styleBlocks[className].append(styleBlock)
                styleBlock.parent = None
                styleBlock.build(styleId)

    def setParent(self, item):
        item.parent = None


class Footprint(Item):
    
    def __init__(self, defName=None, use=None, markup=None, condition=None, **attrs):
        super().__init__(defName, use, markup, condition, attrs)
        # The following Python dictionary is created to serve instead of <self.markup>,
        # since the markup in a <Footprint> is a very specific one. It can contain items of the type
        # <Footrpint>, <Facade>, <RoofSide>, <Ridge>, <Roof> in any order.
        self.styleBlocks = {}
        
    def buildMarkup(self, styleId):
        styleBlocks = self.styleBlocks
        for styleBlock in self.markup:
            className = styleBlock.__class__.__name__
            if not className in styleBlocks:
                styleBlocks[className] = []
            styleBlocks[className].append(styleBlock)
            styleBlock.build(styleId)
        # <self.markup> isn't needed anymore
        self.markup = None


class Facade(Item):
    
    def __init__(self, defName=None, use=None, markup=None, condition=None, **attrs):
        super().__init__(defName, use, markup, condition, attrs)


class Roof(Item):
    
    flat = "flat"
    
    gabled = "gabled"
    
    def __init__(self, defName=None, use=None, markup=None, condition=None, **attrs):
        super().__init__(defName, use, markup, condition, attrs)


class Div(Item):
    
    def __init__(self, defName=None, use=None, markup=None, condition=None, **attrs):
        super().__init__(defName, use, markup, condition, attrs)


class Level(Item):
    
    def __init__(self, defName=None, use=None, markup=None, condition=None, **attrs):
        super().__init__(defName, use, markup, condition, attrs)


class Window(Item):
    
    def __init__(self, defName=None, use=None, markup=None, condition=None, **attrs):
        super().__init__(defName, use, markup, condition, attrs)


class WindowPanel:
    
    def __init__(self, relativeSize, openable):
        self.relativeSize = relativeSize
        self.openable = openable


class Balcony(Item):
    
    def __init__(self, defName=None, use=None, markup=None, condition=None, **attrs):
        super().__init__(defName, use, markup, condition, attrs)


class Door(Item):
    
    def __init__(self, defName=None, use=None, markup=None, condition=None, **attrs):
        super().__init__(defName, use, markup, condition, attrs)
        

class Chimney(Item):
    
    def __init__(self, defName=None, use=None, markup=None, condition=None, **attrs):
        super().__init__(defName, use, markup, condition, attrs)


class RoofSide(Item):

    def __init__(self, defName=None, use=None, markup=None, condition=None, **attrs):
        super().__init__(defName, use, markup, condition, attrs)


class Ridge(Item):
    
    def __init__(self, defName=None, use=None, markup=None, condition=None, **attrs):
        super().__init__(defName, use, markup, condition, attrs)


class Dormer(Item):
    
    def __init__(self, defName=None, use=None, markup=None, condition=None, **attrs):
        super().__init__(defName, use, markup, condition, attrs) 


class Basement(Item):

    def __init__(self, defName=None, use=None, markup=None, condition=None, **attrs):
        super().__init__(defName, use, markup, condition, attrs)
        

def useFrom(itemId):
    return itemId