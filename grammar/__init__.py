from .library import library
from .value import Value
from .scope import *

from item.div import Div as ItemDiv
from item.level import Level as ItemLevel, CurtainWall as ItemCurtainWall
from item.bottom import Bottom as ItemBottom
from item.window import Window as ItemWindow
from item.balcony import Balcony as ItemBalcony
from item.door import Door as ItemDoor


# style attributes that are evaluated once per building by default
_perBuildingByDefault = {
    "lastLevelHeight": 1,
    "levelHeight": 1,
    "groundLevelHeight": 1,
    "bottomHeight": 1,
    "levelHeights": 1,
    "lastRoofLevelHeight": 1,
    "roofLevelHeight": 1,
    "roofLevelHeight0": 1
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
        self.meta = None
        self.build(styleBlocks)
    
    def build(self, inputStyleBlocks):
        styleBlocks = self.styleBlocks
        
        styleId = library.getStyleId()
        
        # style block with <defName> (i.e. style definitions)
        defStyleBlocks = []
        # ordinary style blocks (without <defName>)
        normalStyleBlocks = []
        
        for styleBlock in inputStyleBlocks:
            className = styleBlock.__class__.__name__
            if className == "Meta":
                self.meta = styleBlock
            elif styleBlock.defName:
                library.addStyleBlock(styleBlock.defName, styleBlock, styleId)
                defStyleBlocks.append(styleBlock)
            else:
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


class Meta:
    
    def __init__(self, **attrs):
        self.attrs = attrs


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
    
    # special markup items to taken out of <self.markup> into <self.styleBlocks>
    specialMarkup = {
        "RoofSide": 1,
        "Ridge": 1
    }
    
    def __init__(self, defName=None, use=None, markup=None, condition=None, **attrs):
        super().__init__(defName, use, markup, condition, attrs)
        # <self.markup> can contain both normal markup items (e.g. air vents for the flat roof) and
        # special markup items (<RoofSide>, <Ridge>).
        # The normal markup items are preserved in <self.markup>.
        # Those special markup items are placed to the following Python dictionary.
        self.styleBlocks = {}
    
    def buildMarkup(self, styleId):
        styleBlocks = self.styleBlocks
        markup = []
        for styleBlock in self.markup:
            className = styleBlock.__class__.__name__
            if className in Roof.specialMarkup:
                if not className in styleBlocks:
                    styleBlocks[className] = []
                styleBlocks[className].append(styleBlock)
            else:
                markup.append(styleBlocks)
            styleBlock.build(styleId)
        if markup:
            self.markup = markup
        else:
            # all items from <self.markup> were placed into <self.styleBlocks>,
            # so <self.markup> isn't needed anymore
            self.markup = None


class Div(Item):
    
    def __init__(self, defName=None, use=None, markup=None, condition=None, **attrs):
        super().__init__(defName, use, markup, condition, attrs)
    
    def getItem(self, itemFactory, parent):
        return ItemDiv.getItem(itemFactory, parent, self)


class Level(Item):
    
    def __init__(self, defName=None, use=None, markup=None, condition=None, indices=None, roof=False, **attrs):
        super().__init__(defName, use, markup, condition, attrs)
        self.indices = indices or (0, -1)
        self.roof = roof
        self.isLevel = True
        self.isBottom = False
        self.isTop = False

    def getItem(self, itemFactory, parent):
        return ItemLevel.getItem(itemFactory, parent, self)


class CurtainWall(Level):
    
    def getItem(self, itemFactory, parent):
        return ItemCurtainWall.getItem(itemFactory, parent, self)


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
        return ItemBalcony.getItem(itemFactory, parent, self)


class Door(Item):
    
    def __init__(self, defName=None, use=None, markup=None, condition=None, **attrs):
        super().__init__(defName, use, markup, condition, attrs)
        
    def getItem(self, itemFactory, parent):
        return ItemDoor.getItem(itemFactory, parent, self)
        

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


class Bottom(Item):

    def __init__(self, defName=None, use=None, markup=None, condition=None, **attrs):
        super().__init__(defName, use, markup, condition, attrs)
        self.isLevel = True
        self.isBottom = True
        self.isTop = False
        
    def getItem(self, itemFactory, parent):
        return ItemBottom.getItem(itemFactory, parent, self)
        

def useFrom(itemId):
    return itemId