import math
from mathutils import Vector
from util import zero, zAxis, zeroVector

from item.facade import Facade


"""
key: a cardinal direction
value: the related direction in degrees

directions = {
    'N': 0.,
    'NNE': 22.5,
    'NE': 45.,
    'ENE': 67.5,
    'E': 90.,
    'ESE': 112.5,
    'SE': 135.,
    'SSE': 157.5,
    'S': 180.,
    'SSW': 202.5,
    'SW': 225.,
    'WSW': 247.5,
    'W': 270.,
    'WNW': 292.5,
    'NW': 315.,
    'NNW': 337.5
}
"""


def getDefaultDirection(self, polygon):
    # a perpendicular to the longest edge of the polygon
    return max(polygon.edges).cross(polygon.normal).normalized()


class Roof:
    
    directions = {
        'N': Vector((0., 1., 0.)),
        'NNE': Vector((0.38268, 0.92388, 0.)),
        'NE': Vector((0.70711, 0.70711, 0.)),
        'ENE': Vector((0.92388, 0.38268, 0.)),
        'E': Vector((1., 0., 0.)),
        'ESE': Vector((0.92388, -0.38268, 0.)),
        'SE': Vector((0.70711, -0.70711, 0.)),
        'SSE': Vector((0.38268, -0.92388, 0.)),
        'S': Vector((0., -1., 0.)),
        'SSW': Vector((-0.38268, -0.92388, 0.)),
        'SW': Vector((-0.70711, -0.70711, 0.)),
        'WSW': Vector((-0.92388, -0.38268, 0.)),
        'W': Vector((-1., 0., 0.)),
        'WNW': Vector((-0.92388, 0.38268, 0.)),
        'NW': Vector((-0.70711, 0.70711, 0.)),
        'NNW': Vector((-0.38268, 0.92388, 0.))
    }
    
    shapes = {"flat": 1, "gabled": 1}
    
    # default values
    lastLevelHeight = 1.2*3.
    levelHeight = 3.
    groundLevelHeight = 1.4*3
    basementHeight = 1.
    
    lastRoofLevelHeight = 2.7
    levelHeight = 2.7
    roofLevelHeight0 = 2.7
    
    def __init__(self, data, itemStore, itemFactory):
        self.data = data
        self.itemStore = itemStore
        self.itemFactory = itemFactory
        self.hasGable = False
        self.hasRoofLevels = True
        # Python list with vertices is shared accross all operations
        self.verts = []
        self.allVerts = self.verts
        self.roofIndices = []
        self.wallIndices = []

    def do(self, footprint, building, renderer):
        styleBlock = footprint.styleBlock
        self.init(footprint, styleBlock)
        self.render(footprint, building, renderer)
    
    def init(self, footprint, styleBlock):
        self.verts.clear()
        self.roofIndices.clear()
        self.wallIndices.clear()
        
        footprint.valid = True
        
        # calculate numerical dimensions for the building or building part
        self.calculateDimensions(footprint, styleBlock)
        z1 = footprint.minHeight
        
        verts = self.verts
        verts.extend( Vector((coord[0], coord[1], z1)) for coord in footprint.element.getData(self.data) )
        
        # create a polygon located at <minHeight>
        
        # check if a polygon has been already set (e.g. when placing the building on a terrain)
        polygon = footprint.polygon
        if not polygon.allVerts:
            polygon.init(verts)
        if polygon.n < 3:
            footprint.valid = False
            return
        # check the direction of vertices, it must be counterclockwise
        polygon.checkDirection()
    
    def calculateMinHeight(self, footprint, styleBlock):
        minLevel = footprint.getStyleBlockAttr("minLevel")
        if minLevel:
            footprint.minLevel = minLevel
            # calculate the height for <minLevel>
            levelHeight = footprint.getStyleBlockAttr("levelHeight")
            # either <levelHeight> or <levelHeights> is given
            levelHeights = footprint.getStyleBlockAttr("levelHeights") if levelHeight is None else None
            h = footprint.getStyleBlockAttr("basementHeight", levelHeights.getBasementHeight() if levelHeights else self.basementHeight)
            groundLevelHeight = footprint.getStyleBlockAttr(
                "groundLevelHeight",
                levelHeight if levelHeight else (levelHeights.getLevelHeight(0) if levelHeights else self.groundLevelHeight)
            )
            h += groundLevelHeight
            
            if minLevel > 1.:
                # the height of the middle levels
                if levelHeight:
                    h += (minLevel-1.)*levelHeight
                else:
                    h += levelHeights.getHeight(1, minLevel)
        else:
            h = footprint.getStyleBlockAttr("minHeight", 0.)
        footprint.minHeight = h
        return h
    
    def calculateDimensions(self, footprint, styleBlock):
        """
        Calculate numerical dimensions for the building or building part
        """
        if self.hasGable:
            # temporarily keep <lastLevelOffsetFactor> int the attribute <footprint.lastLevelOffset>
            footprint.lastLevelOffset = footprint.getStyleBlockAttr("lastLevelOffsetFactor")
        
        z2 = self.calculateHeight(footprint, styleBlock)
        z1 = self.calculateMinHeight(footprint, styleBlock)
        
        if not z2:
            # the height is equal to zero 
            footprint.valid = False
            return
        else:
            roofVerticalPosition = z2 - footprint.roofHeight
        wallHeight = roofVerticalPosition - z1
        # validity check
        if wallHeight < 0.:
            footprint.valid = False
            return
        elif wallHeight < zero:
            # no building walls, just a roof
            footprint.noWalls = True
        else:
            footprint.noWalls = False
            footprint.wallHeight = wallHeight
        
        footprint.roofVerticalPosition = z1 if footprint.noWalls else roofVerticalPosition
    
    def calculateRoofHeight(self, footprint, styleBlock):
        h = footprint.getStyleBlockAttr("roofHeight")
        if h is None:
            if "roofLevels" in styleBlock:
                h = self.calculateRoofLevelsHeight(footprint, styleBlock)
                if footprint.lastLevelOffset:
                    h += footprint.lastLevelOffset
            else:
                # default height of the roof
                h = self.height
        footprint.roofHeight = h
        return h

    def calculateRoofLevelsHeight(self, footprint, styleBlock):
        levels = footprint.getStyleBlockAttr("roofLevels") if self.hasRoofLevels else None
        footprint.roofLevels = levels
        
        levelHeight = footprint.getStyleBlockAttr("roofLevelHeight")
        # either <roofLevelHeight> or <roofLevelHeights> is given
        if levelHeight is None:
            levelHeights = footprint.getStyleBlockAttr("rooflevelHeights")
        h = footprint.getStyleBlockAttr(
            "roofLevelHeight0",
            levelHeight if levelHeight else (levelHeights.getLevelHeight(0) if levelHeights else self.roofLevelHeight0)
        )
        
        if levels > 1.:
            # the height of the last level
            lastLevelHeight = footprint.getStyleBlockAttr(
                "lastRoofLevelHeight",
                levelHeight if levelHeight else (levelHeights.getLevelHeight(-1) if levelHeights else self.lastRoofLevelHeight)
            )
            h += lastLevelHeight
            if levels > 2.:
                # the height of the middle levels
                if levelHeights:
                    h += levelHeights.getHeight(1, -2)
                else:
                    h += (levels-2.)*levelHeight
        return h
    
    def calculateLevelsHeight(self, footprint, styleBlock):
        levels = footprint.getStyleBlockAttr("levels", 0.)
        footprint.levels = levels
        if not levels:
            return 0.
        levelHeight = footprint.getStyleBlockAttr("levelHeight")
        # either <levelHeight> or <levelHeights> is given
        levelHeights = footprint.getStyleBlockAttr("levelHeights") if levelHeight is None else None
        h = footprint.getStyleBlockAttr("basementHeight", levelHeights.getBasementHeight() if levelHeights else self.basementHeight)
        groundLevelHeight = footprint.getStyleBlockAttr(
            "groundLevelHeight",
            levelHeight if levelHeight else (levelHeights.getLevelHeight(0) if levelHeights else self.groundLevelHeight)
        )
        h += groundLevelHeight
        
        if levels == 1.:
            # the special case
            lastLevelHeight = groundLevelHeight
        else:
            # the height of the last level
            lastLevelHeight = footprint.getStyleBlockAttr(
                "lastLevelHeight",
                levelHeight if levelHeight else (levelHeights.getLevelHeight(-1) if levelHeights else self.lastLevelHeight)
            )
            h += lastLevelHeight
            if levels > 2.:
                # the height of the middle levels
                if levelHeights:
                    h += levelHeights.getHeight(1, -2)
                else:
                    h += (levels-2.)*levelHeight
        if footprint.lastLevelOffset:
            footprint.lastLevelOffset *= lastLevelHeight
        return h
    
    def calculateHeight(self, footprint, styleBlock):
        h = footprint.getStyleBlockAttr("height")
        if h is None:
            h = self.calculateLevelsHeight(footprint, styleBlock)
            h += self.calculateRoofHeight(footprint, styleBlock)
        else:
            # calculate roof height
            # calculate last level offset in the function below
            self.calculateLevelsHeight(footprint, styleBlock)
            self.calculateRoofHeight(footprint, styleBlock)
        footprint.height = h
        return h

    def processDirection(self, footprint, styleBlock):
        polygon = footprint.polygon
        # <d> stands for direction
        d = footprint.getStyleBlockAttr("roofDirection")
        # getting a direction vector with the unit length
        if d is None:
            if self.hasRidge and footprint.getStyleBlockAttr("roofOrientation") == "across":
                # The roof ridge is across the longest side of the building outline,
                # i.e. the profile direction is along the longest side
                d = max(polygon.edges).normalized()
            else:
                d = getDefaultDirection()
        elif d in Roof.directions:
            d = Roof.directions[d]
        else:
            # trying to get a direction angle in degrees
            if d is None:
                d = getDefaultDirection()
            else:
                d = math.radians(d)
                d = Vector((math.sin(d), math.cos(d), 0.))
        # the direction vector is used by <profile.RoofProfile>
        float.direction = d
        
        # For each vertex from <polygon.verts> calculate projection of the vertex
        # on the vector <d> that defines the roof direction
        projections = footprint.projections
        projections.extend( d[0]*v[0] + d[1]*v[1] for v in polygon.verts )
        minProjIndex = min(range(polygon.n), key = lambda i: projections[i])
        footprint.minProjIndex = minProjIndex
        maxProjIndex = max(range(polygon.n), key = lambda i: projections[i])
        footprint.maxProjIndex = maxProjIndex
        # <polygon> width along the vector <d>
        footprint.polygonWidth = projections[maxProjIndex] - projections[minProjIndex]