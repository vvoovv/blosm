from mathutils import Vector
from util import zero, zAxis, zeroVector


class Roof:
    
    def __init__(self, data):
        self.data = data
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
        
        self.footprint = footprint
        self.style = style
        
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
            self.valid = False
            return
        # check the direction of vertices, it must be counterclockwise
        polygon.checkDirection()
        
        # calculate numerical dimensions for the building or building part
        #self.calculateDimensions(z1)

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