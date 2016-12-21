import math
from mathutils import Vector
from . import Roof
from util.osm import parseNumber


gabled = (
    (
        (0., 0.),
        (0.5, 1.),
        (1., 0.)
    ),
    {
        "numSamples": 10,
        "angleToHeight": 0.5
    }
)

round = (
    (
        (0., 0.),
        (0.01, 0.098),
        (0.038, 0.191),
        (0.084, 0.278),
        (0.146, 0.354),
        (0.222, 0.416),
        (0.309, 0.462),
        (0.402, 0.49),
        (0.5, 0.5),
        (0.598, 0.49),
        (0.691, 0.462),
        (0.778, 0.416),
        (0.854, 0.354),
        (0.916, 0.278),
        (0.962, 0.191),
        (0.99, 0.098),
        (1., 0.)
    ),
    {
        "numSamples": 1000,
        "angleToHeight": None
    }
)

gambrel = (
    (
        (0., 0.),
        (0.2, 0.6),
        (0.5, 1.),
        (0.8, 0.6),
        (1., 0.)
    ),
    {
        "numSamples": 10,
        "angleToHeight": None
    }
)

saltbox = (
    (
        (0., 0.),
        (0.35, 1.),
        (0.65, 1.),
        (1., 0.)
    ),
    {
        "numSamples": 100,
        "angleToHeight": 0.35
    }
)


class RoofProfile(Roof):
    
    defaultHeight = 10.
    
    def __init__(self, data):
        super().__init__()
        self.projections = []
        
        profile = data[0]
        self.profile = profile
        for attr in data[1]:
            setattr(self, attr, data[1][attr])
        
        # quantize <profile> with <numSamples>
        _profile = tuple(math.ceil(p[0]*self.numSamples) for p in profile)
        profileQ = []
        index = 0
        for i in range(self.numSamples):
            if i >= _profile[index+1]:
                index += 1  
            profileQ.append(index)
        profileQ.append(index)
        self.profileQ = profileQ

    def init(self, element, minHeight, osm):
        super().init(element, minHeight, osm)
        self.projections.clear()
    
    def make(self, bldgMaxHeight, roofMinHeight, bldgMinHeight, osm):
        verts = self.verts
        polygon = self.polygon
        indices = polygon.indices
        
        if not self.projections:
            self.processDirection()
        
        _v = verts[indices[0]]
        for i in range(polygon.n):
            v = verts[indices[i]]
            self.sampleProfile(i, roofMinHeight)
            _v = v
        
        return True
    
    def sampleProfile(self, index, roofMinHeight):
        proj = self.projections
        p = self.profile
        x = (proj[index] - proj[self.minProjIndex]) / self.polygonWidth
        i = self.profileQ[
            math.floor(x * self.numSamples)
        ]
        h1 = p[i][1]
        h2 = p[i+1][1]
        h = h1 + (h2 - h1) / (p[i+1][0] - p[i][0]) * (x - p[i][0])
        v = self.verts[self.polygon.indices[index]]
        self.verts.append(Vector((v.x, v.y, roofMinHeight + self.h * h)))
        
    
    def getHeight(self):
        element = self.element
        tags = element.tags
        
        if "roof:height" in tags:
            h = parseNumber(tags["roof:height"], self.defaultHeight)
        elif not self.angleToHeight is None and "roof:angle" in tags:
            angle = parseNumber(tags["roof:angle"])
            if angle is None:
                h = self.defaultHeight
            else:
                self.processDirection()
                h = self.angleToHeight * self.polygonWidth * math.tan(math.radians(angle))
        else:
            h = self.defaultHeight
        
        self.h = h
        return h
    
    def render(self, r):
        tuple(r.bm.verts.new(v) for v in self.verts)