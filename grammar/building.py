from .value import Value
from .scope import *

from . import Item

from item.div import Div as ItemDiv
from item.level import Level as ItemLevel
from item.top import Top as ItemTop
from item.bottom import Bottom as ItemBottom
from item.window import Window as ItemWindow
from item.balcony import Balcony as ItemBalcony
from item.entrance import Entrance as ItemEntrance
from item.corner import Corner as ItemCorner


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


class BuildingItem(Item):
    
    def __init__(self, defName, use, markup, condition, attrs):
        super().__init__(defName, use, markup, condition, attrs)
        
        self.isLevel = False
        # A style block may contain a directive (or a command), for example:
        # _building_count_door
        # It means: each time the style block is assigned to an item, increase the building attribute
        # with the name <door> serving as a counter.
        # The scope of the counter is the whole building.
        self.buildingCount = None
        self.init()
    
    def initAttrs(self):
        attrs = self.attrs
        # restructure <self.attrs>
        for attr in attrs:
            value = attrs[attr]
            # a Python tuple containg two elements
            if isinstance(value, Scope):
                # <Scope> only makes sense for complex values
                value.value.value.scope = value.scope
                value = value.value
            isComplexValue = isinstance(value, Value)
            if isComplexValue:
                if attr in _perBuildingByDefault:
                    value.value.scope = perBuilding
            elif attr[0] == '_':
                # the case of a directive (or a command)
                if attr.startswith("_building_count_"):
                    attr = "at" + attr[16:]
                    if self.buildingCount:
                        if len(self.buildingCount) == 1:
                            self.buildingCount = [self.buildingCount[0], attr]
                        else:
                            self.buildingCount.append(attr)
                    else:
                        self.buildingCount = (attr,)
                continue
            elif isinstance(value, str):
                if value == "yes":
                    value = True
                elif value == "no":
                    value = False
                elif value == "none":
                    value = None
            
            attrs[attr] = (value.value if isComplexValue else value, isComplexValue)


class Footprint(BuildingItem):
    
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


class Facade(BuildingItem):
    
    def __init__(self, defName=None, use=None, markup=None, condition=None, **attrs):
        super().__init__(defName, use, markup, condition, attrs)


class Roof(BuildingItem):
    
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


class Div(BuildingItem):
    
    def __init__(self, defName=None, use=None, markup=None, condition=None, **attrs):
        super().__init__(defName, use, markup, condition, attrs)
    
    def getItem(self, parent):
        return ItemDiv(parent, parent.footprint, parent.facade, self)


class Level(BuildingItem):
    
    def __init__(self, defName=None, use=None, markup=None, condition=None, indices=None, roofLevels=False, **attrs):
        super().__init__(defName, use, markup, condition, attrs)
        self.indices = indices or (0, -1)
        self.roof = roofLevels
        self.isLevel = True
        self.isBottom = False
        self.isTop = False

    def getItem(self, parent):
        return ItemLevel(parent, self)


class Window(BuildingItem):
    
    def __init__(self, defName=None, use=None, markup=None, condition=None, **attrs):
        super().__init__(defName, use, markup, condition, attrs)

    def getItem(self, itemFactory, parent):
        return ItemWindow.getItem(itemFactory, parent, self)


class WindowPanel:
    
    def __init__(self, relativeSize, openable):
        self.relativeSize = relativeSize
        self.openable = openable


class Balcony(BuildingItem):
    
    def __init__(self, defName=None, use=None, markup=None, condition=None, **attrs):
        super().__init__(defName, use, markup, condition, attrs)

    def getItem(self, itemFactory, parent):
        return ItemBalcony.getItem(itemFactory, parent, self)


class Entrance(BuildingItem):
    
    def __init__(self, defName=None, use=None, markup=None, condition=None, **attrs):
        super().__init__(defName, use, markup, condition, attrs)
        
    def getItem(self, parent):
        return ItemEntrance(parent, parent.footprint, self)


class Corner(BuildingItem):
    
    def __init__(self, defName=None, use=None, markup=None, condition=None, **attrs):
        super().__init__(defName, use, markup, condition, attrs)
        
    def getItem(self, parent):
        return ItemCorner(parent, parent.footprint, self)
        

class Chimney(BuildingItem):
    
    def __init__(self, defName=None, use=None, markup=None, condition=None, **attrs):
        super().__init__(defName, use, markup, condition, attrs)


class RoofSide(BuildingItem):

    def __init__(self, defName=None, use=None, markup=None, condition=None, **attrs):
        super().__init__(defName, use, markup, condition, attrs)


class Ridge(BuildingItem):
    
    def __init__(self, defName=None, use=None, markup=None, condition=None, **attrs):
        super().__init__(defName, use, markup, condition, attrs)


class Dormer(BuildingItem):
    
    def __init__(self, defName=None, use=None, markup=None, condition=None, **attrs):
        super().__init__(defName, use, markup, condition, attrs)


class Top(BuildingItem):

    def __init__(self, defName=None, use=None, markup=None, condition=None, **attrs):
        super().__init__(defName, use, markup, condition, attrs)
        self.isLevel = True
        self.isBottom = False
        self.isTop = True
        
    def getItem(self, parent):
        return ItemTop(parent, self)


class Bottom(BuildingItem):

    def __init__(self, defName=None, use=None, markup=None, condition=None, **attrs):
        super().__init__(defName, use, markup, condition, attrs)
        self.isLevel = True
        self.isBottom = True
        self.isTop = False
        
    def getItem(self, parent):
        return ItemBottom(parent, self)
        

def useFrom(itemId):
    return itemId
