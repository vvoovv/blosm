from .library import library
from .value import Value
from .scope import *

from item.div import Div as ItemDiv
from item.level import Level as ItemLevel
from item.basement import Basement as ItemBasement
from item.window import Window as ItemWindow
from item.balcony import Balcony as ItemBalcony


# style attributes that are evaluated once per building by default
_perBuildingByDefault = {
    "lastLevelHeight": 1,
    "levelHeight": 1,
    "groundLevelHeight": 1,
    "basementHeight": 1,
    "levelHeights": 1,
    "lastRoofLevelHeight": 1,
    "levelHeight": 1,
    "roofLevelHeight0": 1,
    "roofLevelHeights": 1
}


class Item:
    
    def __init__(self, defName, use, markup, condition, attrs):
        self.defName = defName
        self.use = use
        self.markup = markup
        self.condition = condition
        self.attrs = attrs
        self.isLevel = False
        self.init()
    
    def init(self):
        self.initAttrs()
        if self.markup:
            for styleBlock in self.markup:
                self.setParent(styleBlock)
    
    def initAttrs(self):
        attrs = self.attrs
        # restructure <self.attrs>
        for attr in attrs:
            value = attrs[attr]
            # a Python tuple containg two elements
            if isinstance(value, Scope):
                scope = value.scope
                value = value.value
            else:
                value, scope = value, perBuilding if attr in _perBuildingByDefault else perFootprint
            isComplexValue = isinstance(value, Value)
            attrs[attr] = (value, scope, isComplexValue)
    
    def setParent(self, styleBlock):
        styleBlock.parent = self
    
    def build(self, styleId):
        if self.use:
            self.styleId = styleId
            self._applyUse()
        if self.markup:
            self.buildMarkup(styleId)
    
    def buildMarkup(self, styleId):
        for styleBlock in self.markup:
            styleBlock.build(styleId)
    
    def _applyUse(self):
        markup = self.markup
        # note the reversed iterator
        for defName in reversed(self.use):
            _styleBlock = library.getStyleBlock(
                defName, self.__class__.__name__, self.styleId
            )
            if _styleBlock:
                for attr in _styleBlock.attrs:
                    if not attr in self.attrs:
                        self.attrs[attr] = _styleBlock.attrs[attr]
                if not markup:
                    markup = _styleBlock.markup
        # set the markup if it was defined in a definition style block referenced in <self.use>
        if not self.markup and markup:
            self.markup = markup
    
    def __contains__(self, attr):
        return attr in self.attrs


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
        
        # style block with <defName> (i.e. style definitions)
        defStyleBlocks = []
        # ordinary style blocks (without <defName>)
        normalStyleBlocks = []
        
        for styleBlock in inputStyleBlocks:
            if styleBlock.defName:
                library.addStyleBlock(styleBlock.defName, styleBlock, styleId)
                defStyleBlocks.append(styleBlock)
            else:
                className = styleBlock.__class__.__name__
                if not className in styleBlocks:
                    styleBlocks[className] = []
                styleBlocks[className].append(styleBlock)
                styleBlock.parent = None
                normalStyleBlocks.append(styleBlock)
        
        for styleBlock in defStyleBlocks:
            styleBlock.build(styleId)
        
        for styleBlock in normalStyleBlocks:
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
    
    def getItem(self, itemFactory, parent):
        return ItemDiv.getItem(itemFactory, parent, self)


class Level(Item):
    
    def __init__(self, defName=None, use=None, markup=None, condition=None, **attrs):
        super().__init__(defName, use, markup, condition, attrs)
        self.isLevel = True

    def getItem(self, itemFactory, parent):
        return ItemLevel.getItem(itemFactory, parent, self)


class Window(Item):
    
    def __init__(self, defName=None, use=None, markup=None, condition=None, **attrs):
        super().__init__(defName, use, markup, condition, attrs)

    def getItem(self, itemFactory, parent):
        return ItemWindow.getItem(itemFactory, parent, self)


class WindowPanel:
    
    def __init__(self, relativeSize, openable):
        self.relativeSize = relativeSize
        self.openable = openable


class Balcony(Item):
    
    def __init__(self, defName=None, use=None, markup=None, condition=None, **attrs):
        super().__init__(defName, use, markup, condition, attrs)

    def getItem(self, itemFactory, parent):
        return ItemWindow.getItem(itemFactory, parent, self)


class Door(Item):
    
    def __init__(self, defName=None, use=None, markup=None, condition=None, **attrs):
        super().__init__(defName, use, markup, condition, attrs)
        
    def getItem(self, itemFactory, parent):
        return ItemWindow.getItem(itemFactory, parent, self)
        

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
        self.isLevel = True
        
    def getItem(self, itemFactory, parent):
        return ItemBasement.getItem(itemFactory, parent, self)
        

def useFrom(itemId):
    return itemId