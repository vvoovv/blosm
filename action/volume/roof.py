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


def getDefaultDirection(polygon):
    # a perpendicular to the longest edge of the polygon
    return max(polygon.edges).cross(polygon.normal).normalized()


class Roof:
    
    # default values
    levelHeight = 3.
    
    roofLevelHeight = 2.7
    
    topHeight = 0.5
    
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
    
    def __init__(self, data, itemStore, itemFactory):
        self.data = data
        self.itemStore = itemStore
        self.itemFactory = itemFactory
        self.hasGable = False
        self.hasRoofLevels = True
        self.angleToHeight = None
    
    def do(self, footprint, coords):
        self.init(footprint, coords)
        if footprint.valid:
            self.render(footprint)
    
    def init(self, footprint, coords):
        # calculate numerical dimensions for the building or building part
        self.calculateDimensions(footprint)
        if not footprint.valid:
            return
        z1 = footprint.minHeight
        
        # create a polygon located at <z1>
        
        # Check if a polygon has been already set (e.g. when placing the building on a terrain or
        # calculating the area of the whole building footprint)
        polygon = footprint.polygon
        if polygon.allVerts:
            polygon.setHeight(z1)
        else:
            polygon.init( Vector((coord[0], coord[1], z1)) for coord in coords )
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
        if wallHeight < -zero:
            footprint.valid = False
            return
        elif wallHeight < zero:
            # no building walls, just a roof
            footprint.noWalls = True
            footprint.numLevels = 0
        else:
            footprint.noWalls = False
            footprint.wallHeight = wallHeight
        
        footprint.roofVerticalPosition = z1 if footprint.noWalls else roofVerticalPosition
        
        self.validate(footprint)
    
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
                d = getDefaultDirection(polygon)
        elif d in Roof.directions:
            d = Roof.directions[d]
        else:
            # trying to get a direction angle in degrees
            if d is None:
                d = getDefaultDirection(polygon)
            else:
                d = math.radians(d)
                d = Vector((math.sin(d), math.cos(d), 0.))
        # the direction vector is used by <action.volume.roof_profile.RoofProfile>
        footprint.direction = d
        
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
    
    def calculateRoofHeight(self, footprint):
        h = footprint.getStyleBlockAttr("topHeight")
        if h is None:
            h = self.topHeight
        footprint.levelHeights.topHeight = h
        
        h = footprint.getStyleBlockAttr("roofHeight")
        if h is None:
            if not self.angleToHeight is None and "roofAngle" in footprint.styleBlock:
                angle = footprint.getStyleBlockAttr("roofAngle")
                if not angle is None:
                    self.processDirection()
                    h = self.angleToHeight * footprint.polygonWidth * math.tan(math.radians(angle))
                    if self.hasRoofLevels:
                        # still calculate the number of roof levels and their height
                        self.calculateRoofLevelsHeight(footprint)
            if h is None:
                if self.hasRoofLevels:
                    h = self.calculateRoofLevelsHeight(footprint)
                    if h:
                        # The following line means that we need to calculate
                        # the last level offset and the roof height later in the code
                        footprint.roofHeight = None
                        return h
                    else:
                        # default height of the roof
                        h = self.height
                else:
                    # default height of the roof
                    h = self.height
        else:
            # still calculate the number of roof levels and their height
            self.calculateRoofLevelsHeight(footprint)
        footprint.roofHeight = h
        return h

    def calculateRoofLevelsHeight(self, footprint):
        numRooflevels = footprint.getStyleBlockAttr("numRoofLevels")
        if not numRooflevels:
            footprint.roofLevelsHeight = 0.
            return 0.
        footprint.numRoofLevels = numRooflevels
        
        lh = footprint.levelHeights
        
        if lh.levelHeights:
            h = lh.levelHeights.getRoofHeight(0, numRooflevels-1)
        else:
            roofLevelHeight = footprint.getStyleBlockAttr("roofLevelHeight")
            if roofLevelHeight:
                lh.roofLevelHeight = roofLevelHeight
                if not lh.multipleHeights:
                    lh.multipleHeights = True
            else:
                roofLevelHeight = lh.levelHeight
            #
            # the very first roof level
            #
            # <roofLevelHeight0> can't be defined without <roofLevelHeight>
            h = footprint.getStyleBlockAttr("roofLevelHeight0")
            if h:
                lh.roofLevelHeight0 = h
                if not lh.multipleHeights:
                    lh.multipleHeights = True
            else:
                h = roofLevelHeight
            #
            # the roof levels above the very first roof level
            #
            if numRooflevels > 1:
                #
                # the last roof level
                #
                lastRoofLevelHeight = footprint.getStyleBlockAttr("lastRoofLevelHeight")
                if lastRoofLevelHeight:
                    lh.lastRoofLevelHeight = lastRoofLevelHeight
                    if not lh.multipleHeights:
                        lh.multipleHeights = True
                else:
                    lastRoofLevelHeight = roofLevelHeight
                h += lastRoofLevelHeight
                #
                # the levels between the very first roof level and the last roof level
                #
                if numRooflevels > 2:
                    h += (numRooflevels-2)*roofLevelHeight
        footprint.roofLevelsHeight = h
        return h