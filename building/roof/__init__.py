from mathutils import Vector
from util.osm import parseNumber
from util.polygon import Polygon
from renderer import Renderer

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


class Roof:
    
    assetPath = "roofs.blend"
    
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
    
    def __init__(self):
        self.verts = []
        self.roofIndices = []
        self.wallIndices = []
    
    def init(self, element, minHeight, osm):
        self.verts.clear()
        self.roofIndices.clear()
        self.wallIndices.clear()
        
        self.element = element
        
        verts = self.verts
        self.verts.extend( Vector((coord[0], coord[1], minHeight)) for coord in element.getData(osm) )
        self.polygon = Polygon(
            tuple(range(len(verts))),
            verts
        )
        # check the direction of vertices, it must be counterclockwise
        self.polygon.checkDirection()
    
    def getHeight(self):
        h = parseNumber(
            self.element.tags.get("roof:height", self.defaultHeight),
            self.defaultHeight
        )
        self.h = h
        return h
    
    def render(self, r):
        wallIndices = self.wallIndices
        roofIndices = self.roofIndices
        if not (roofIndices or wallIndices):
            return
        
        bm = r.bm
        verts = [bm.verts.new(v) for v in self.verts]
        
        if wallIndices:
            materialIndex = r.getWallMaterialIndex(self.element)
            for f in (bm.faces.new(verts[i] for i in indices) for indices in wallIndices):
                f.material_index = materialIndex
        
        if roofIndices:
            materialIndex = r.getRoofMaterialIndex(self.element)
            for f in (bm.faces.new(verts[i] for i in indices) for indices in roofIndices):
                f.material_index = materialIndex