import math
from mathutils import Vector
from . import Roof
from util import zero
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
        p = self.profile
        polygon = self.polygon
        indices = polygon.indices
        
        if not self.projections:
            self.processDirection()
        
        _v = verts[indices[0]]
        _pIndex, onProfile, _pCoord, vertIndex = self.sampleProfile(0, roofMinHeight)
        for i in range(1, polygon.n):
            v = verts[indices[i]]
            pIndex, onProfile, pCoord, vertIndex = self.sampleProfile(i, roofMinHeight)
            # Create vertices for profile reference points located
            # with the indices between _pIndex and pIndex
            if pIndex != _pIndex:
                for _i in (range(_pIndex+1, pIndex if onProfile else pIndex+1)
                    if pIndex > _pIndex else
                    range(_pIndex-1 if onProfile else _pIndex, pIndex, -1)):
                    multiplier = (p[_i][0] - _pCoord) / (pCoord - _pCoord)
                    verts.append(Vector((
                        _v.x + multiplier * (v.x - _v.x),
                        _v.y + multiplier * (v.x - _v.x),
                        roofMinHeight + self.h * p[_i][1]
                    )))
            _v = v
            _pIndex = pIndex
            _pCoord = pCoord
        # TODO: the closing part
        
        return True
    
    def sampleProfile(self, index, roofMinHeight):
        verts = self.verts
        proj = self.projections
        p = self.profile
        # Coordinate along roof profile (i.e. along the roof direction)
        # the roof direction is calculated in self.processDirection(..);
        # the coordinate can possess the value from 0. to 1.
        pCoord = (proj[index] - proj[self.minProjIndex]) / self.polygonWidth
        # index in <self.profileQ>
        pIndex = self.profileQ[
            math.floor(pCoord * self.numSamples)
        ]
        coef = pCoord - p[pIndex][0]
        # check if x is equal zero
        if coef < zero:
            pCoord, h = p[pIndex]
            onProfile = True
        elif abs(p[pIndex + 1][0] - pCoord) < zero:
            # increase <profileIndex> by one
            pIndex += 1
            pCoord, h = p[pIndex]
            onProfile = True
        else:
            onProfile = False
            h1 = p[pIndex][1]
            h2 = p[pIndex+1][1]
            h = h1 + (h2 - h1) / (p[pIndex+1][0] - p[pIndex][0]) * coef
        v = verts[self.polygon.indices[index]]
        vertIndex = len(verts)
        verts.append(Vector((v.x, v.y, roofMinHeight + self.h * h)))
        return pIndex, onProfile, pCoord, vertIndex
        
    
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