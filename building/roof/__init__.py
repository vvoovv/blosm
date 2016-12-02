from mathutils import Vector
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
    
    def init(self, element, osm):
        self.element = element
        self.polygon = Polygon(
            element.getData(osm) if element.t is Renderer.polygon else element.getOuterData(osm)
        )
        # check the direction of vertices, it must be counterclockwise
        self.polygon.checkDirection()
    
    def render(self, r):
        bm = r.bm
        sidesIndices = self.sidesIndices
        verts = [bm.verts.new(v) for v in self.polygon.allVerts]
        f = bm.faces.new(verts[i] for i in self.polygon.indices)
        f.material_index = r.getMaterialIndex(self.element)
        
        materialIndex = r.getSideMaterialIndex(self.element)
        for f in (bm.faces.new(verts[i] for i in indices) for indices in sidesIndices):
            f.material_index = materialIndex