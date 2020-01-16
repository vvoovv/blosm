import math
from mathutils import Vector
from util import zero


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
    
    # default values
    lastLevelHeight = 1.2*3.
    levelHeight = 3.
    groundLevelHeight = 1.4*3
    basementHeight = 1.
    
    lastRoofLevelHeight = 2.7
    roofLevelHeight = 2.7
    roofLevelHeight0 = 2.7
    
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
    
    def __init__(self, data, itemStore, itemFactory):
        self.data = data
        self.itemStore = itemStore
        self.itemFactory = itemFactory
        self.hasGable = False
        self.hasRoofLevels = True
    
    def do(self, footprint, renderer):
        self.init(footprint)
        if footprint.valid:
            self.render(footprint, renderer)
    
    def init(self, footprint):
        # calculate numerical dimensions for the building or building part
        self.calculateDimensions(footprint)
        if not footprint.valid:
            return
        z1 = footprint.minHeight
        
        # create a polygon located at <minHeight>
        
        # check if a polygon has been already set (e.g. when placing the building on a terrain)
        polygon = footprint.polygon
        if not polygon.allVerts:
            polygon.init( Vector((coord[0], coord[1], z1)) for coord in footprint.element.getData(self.data) )
        if polygon.n < 3:
            footprint.valid = False
            return
        # check the direction of vertices, it must be counterclockwise
        polygon.checkDirection()
    
    def calculateDimensions(self, footprint):
        """
        Calculate numerical dimensions for the building or building part
        """
        if self.hasGable:
            # temporarily keep <lastLevelOffsetFactor> int the attribute <footprint.lastLevelOffset>
            footprint.lastLevelOffset = footprint.getStyleBlockAttr("lastLevelOffsetFactor")
            
        levelHeights = footprint.levelHeights
        z2 = levelHeights.calculateHeight(self)
        z1 = levelHeights.calculateMinHeight()
        
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
    
    def processDirection(self, footprint):
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