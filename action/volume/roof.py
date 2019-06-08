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
    
    def init(self, footprint, style):
        self.verts.clear()
        self.roofIndices.clear()
        self.wallIndices.clear()
        
        self.valid = True
        
        # minimum height
        z1 = self.getMinHeight()
        
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
        
        # calculate numerical dimensions for the building or building part
        self.calculateDimensions(z1)

    def getMinHeight(self):
        style = self.style
        minLevel = style.get("minLevel")
        if minLevel:
            # calculate the height based on the <minLevel>
            # z0 = 0. if numLevels is None else self.levelHeight * (numLevels-1+Roof.groundLevelFactor)
            z0 = minLevel*style.get("levelHeight", 0)
        else:
            z0 = style.get("minHeight", 0)
        return z0

    def calculateDimensions(self, z1, footprint, style):
        """
        Calculate numerical dimensions for the building or building part
        """
        if self.hasGable:
            footprint.lastLevelOffsetFactor = style.get("lastLevelOffsetFactor") 
        self.calculateRoofLevels(footprint, style)
        self.calculateRoofHeight(footprint, style)
        self.calculateHeight(footprint, style)
        
        z2 = self.getHeight()
        if z2 is None:
            # no tag <height> or invalid value
            roofVerticalPosition = self.levelHeight * (self.getLevels()-1+Roof.groundLevelFactor)
            z2 = roofVerticalPosition + roofHeight
        elif not z2:
            # the tag <height> is equal to zero 
            self.valid = False
            return
        else:
            roofVerticalPosition = z2 - roofHeight
        wallHeight = roofVerticalPosition - z1
        # validity check
        if wallHeight < 0.:
            self.valid = False
            return
        elif wallHeight < zero:
            # no building walls, just a roof
            self.noWalls = True
        else:
            self.noWalls = False
            self.wallHeight = wallHeight
        
        self.z1 = z1
        self.z2 = z2
        self.roofVerticalPosition = z1 if self.noWalls else roofVerticalPosition
        self.roofHeight = roofHeight

    def calculateRoofLevels(self, footprint, style):
        footprint.roofLevels = style.get("roofLevels")
        
    def calculateRoofHeight(self, footprint, style):
        h = style.get("roofHeight")
        if h is None:
            roofLevels = footprint.roofLevels
            if roofLevels is None:
                h = self.height
            else:
                roofLevelHeights = style.get("roofLevelHeights")
                if roofLevels:
    
    def calculateHeight(self, footprint, style):
        h = style.get("height")
        lastLevelHeight = None
        if h is None:
            levels = footprint.levels
            levelHeights = style.get("levelHeights")
            h = style.get("basementHeight", levelHeights.getBasementHeight() if levelHeights else self.basementHeight)
            # Use the variable <lastLevelHeight> for the ground level height
            # it will be used for the calculation of the last level offset if
            # number of levels is equal to 1
            lastLevelHeight = style.get("groundLevelHeight", levelHeights.getLevelHeight(0) if levelHeights else self.groundLevelHeight)
            h += lastLevelHeight
            if levels > 1.:
                lastLevelHeight = style.get("lastLevelHeight", levelHeights.getLevelHeight(-1) if levelHeights else self.lastLevelHeight)
                h += lastLevelHeight
                if levels > 2.:
                    if levelHeights:
                        h += levelHeights.getHeight(1, -2)
                    else:
                        h += (levels-2.)*style.get("levelHeight", self.levelHeight)
            if footprint.lastLevelOffsetFactor:
                # subtract
                
        footprint.height = h

    def processDirection(self, footprint, style):
        polygon = footprint.polygon
        # <d> stands for direction
        d = style.get("roofDirection")
        # getting a direction vector with the unit length
        if d is None:
            if self.hasRidge and style.get("roofOrientation") == "across":
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