import math
from mathutils import Vector
from . import Roof
from util import zero
from util.osm import parseNumber


gabledRoof = (
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

roundRoof = (
    (
        (0., 0.),
        (0.01, 0.195),
        (0.038, 0.383),
        (0.084, 0.556),
        (0.146, 0.707),
        (0.222, 0.831),
        (0.309, 0.924),
        (0.402, 0.981),
        (0.5, 1.),
        (0.598, 0.981),
        (0.691, 0.924),
        (0.778, 0.831),
        (0.854, 0.707),
        (0.916, 0.556),
        (0.962, 0.383),
        (0.99, 0.195),
        (1., 0.)
    ),
    {
        "numSamples": 1000,
        "angleToHeight": None
    }
)

gambrelRoof = (
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

saltboxRoof = (
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
    
    defaultHeight = 3.
    
    def __init__(self, data):
        super().__init__()
        self.projections = []
        
        profile = data[0]
        self.profile = profile
        self.lastProfileIndex = len(profile) - 1
        for attr in data[1]:
            setattr(self, attr, data[1][attr])
        
        self.lEndZero = not profile[0][1]
        self.rEndZero = not profile[-1][1]
        
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
        wallIndices = self.wallIndices
        noWalls = bldgMinHeight is None
        
        def createProfileVertices(
            i,
            v1, pIndex1, onProfile1, pCoord1, vertIndex1,
            v2, pIndex2, onProfile2, pCoord2, vertIndex2
        ):
            """
            Create vertices for profile reference points located
            with the indices between _pIndex and pIndex
            """
            skip1 = noWalls and onProfile1 and\
                ((self.lEndZero and not pIndex1) or\
                (self.rEndZero and pIndex1 == self.lastProfileIndex))
            skip2 = noWalls and onProfile2 and\
                ((self.lEndZero and not pIndex2) or\
                (self.rEndZero and pIndex2 == self.lastProfileIndex))
            
            if skip1 and skip2 and pIndex1 == pIndex2:
                return
            _wallIndices = [vertIndex1]
            if not skip1:
                _wallIndices.append(indices[i-1])
            if not skip2:
                _wallIndices.append(indices[i])
            _wallIndices.append(vertIndex2)
            
            vertIndex = len(verts)
            if pIndex2 != pIndex1:
                for _i in (
                    range(pIndex2-1 if onProfile2 else pIndex2, pIndex1, -1)
                    if pIndex2 > pIndex1 else
                    range(pIndex2+1, pIndex1 if onProfile1 else pIndex1+1)
                ):
                    multiplier = (p[_i][0] - pCoord1) / (pCoord2 - pCoord1)
                    verts.append(Vector((
                        v1.x + multiplier * (v2.x - v1.x),
                        v1.y + multiplier * (v2.y - v1.y),
                        roofMinHeight + self.h * p[_i][1]
                    )))
                    _wallIndices.append(vertIndex)
                    vertIndex += 1
            wallIndices.append(_wallIndices)
            
        
        if not self.projections:
            self.processDirection()
        
        _v = v0 = verts[indices[0]]
        _pIndex, _onProfile, _pCoord, _vertIndex =\
            pIndex0, onProfile0, pCoord0, vertIndex0 =\
            self.sampleProfile(0, roofMinHeight, noWalls)
        for i in range(1, polygon.n):
            v = verts[indices[i]]
            pIndex, onProfile, pCoord, vertIndex = self.sampleProfile(i, roofMinHeight, noWalls)
            createProfileVertices(
                i,
                _v, _pIndex, _onProfile, _pCoord, _vertIndex,
                 v,  pIndex,  onProfile,  pCoord,  vertIndex
            )        
            _v = v
            _pIndex = pIndex
            _onProfile = onProfile
            _pCoord = pCoord
            _vertIndex = vertIndex
        # the closing part
        createProfileVertices(
            0,
            v,  pIndex,  onProfile,  pCoord,  vertIndex,
            v0, pIndex0, onProfile0, pCoord0, vertIndex0
        )
        
        return True
    
    def sampleProfile(self, index, roofMinHeight, noWalls):
        verts = self.verts
        indices = self.polygon.indices
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
        distance = pCoord - p[pIndex][0]
        # check if x is equal zero
        if distance < zero:
            if self.lEndZero and noWalls and not pIndex:
                return 0, True, 0., indices[index]
            pCoord, h = p[pIndex]
            onProfile = True
        elif abs(p[pIndex + 1][0] - pCoord) < zero:
            # increase <profileIndex> by one
            pIndex += 1
            if self.rEndZero and noWalls and pIndex == self.lastProfileIndex:
                return self.lastProfileIndex, True, 1., indices[index]
            pCoord, h = p[pIndex]
            onProfile = True
        else:
            onProfile = False
            h1 = p[pIndex][1]
            h2 = p[pIndex+1][1]
            h = h1 + (h2 - h1) / (p[pIndex+1][0] - p[pIndex][0]) * distance
        v = verts[indices[index]]
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