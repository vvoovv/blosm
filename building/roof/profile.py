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


class ProfileVert:
    
    def __init__(self, roof, i, roofMinHeight, noWalls):
        self.i = i
        verts = roof.verts
        indices = roof.polygon.indices
        proj = roof.projections
        p = roof.profile
        d = roof.direction
        v = verts[indices[i]]
        onProfile = False
        createVert = True
        # Coordinate along roof profile (i.e. along the roof direction)
        # the roof direction is calculated in roof.processDirection(..);
        # the coordinate can possess the value from 0. to 1.
        x = (proj[i] - proj[roof.minProjIndex]) / roof.polygonWidth
        # Coordinate (with an offset) across roof profile;
        # a perpendicular to <roof.direction> is equal to <Vector((-d[1], d[0], 0.))
        self.y = -v[0]*d[1] + v[1]*d[0]
        # index in <roof.profileQ>
        index = roof.profileQ[
            math.floor(x * roof.numSamples)
        ]
        distance = x - p[index][0]
        # check if x is equal zero
        if distance < zero:
            onProfile = True
            if roof.lEndZero and noWalls and not index:
                createVert = False
                index = 0
                x = 0.
                vertIndex = indices[i]
            else:
                x, h = p[index]
        elif abs(p[index + 1][0] - x) < zero:
            onProfile = True
            # increase <index> by one
            index += 1
            if roof.rEndZero and noWalls and index == roof.lastProfileIndex:
                createVert = False
                index = roof.lastProfileIndex
                x = 1.
                vertIndex = indices[i]
            else:
                x, h = p[index]
        else:
            h1 = p[index][1]
            h2 = p[index+1][1]
            h = h1 + (h2 - h1) / (p[index+1][0] - p[index][0]) * distance
        if createVert:
            vertIndex = len(verts)
            verts.append(Vector((v.x, v.y, roofMinHeight + roof.h * h)))
        self.index = index
        self.onProfile = onProfile
        self.x = x
        self.vertIndex = vertIndex


class RoofProfile(Roof):
    
    defaultHeight = 3.
    
    def __init__(self, data):
        super().__init__()
        self.projections = []
        
        profile = data[0]
        self.profile = profile
        numProfilesPoints = len(profile)
        self.lastProfileIndex = numProfilesPoints - 1
        # profile slots
        self.slots = [ [] for _ in range(numProfilesPoints) ]
        
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
        for slot in self.slots:
            slot.clear()
    
    def make(self, bldgMaxHeight, roofMinHeight, bldgMinHeight, osm):
        polygon = self.polygon
        noWalls = bldgMinHeight is None
        
        if not self.projections:
            self.processDirection()
        
        i = i0 = self.minProjIndex
        # <pv> stands for profile vertex
        _pv = pv0 = ProfileVert(self, i, roofMinHeight, noWalls)
        while True:
            i = polygon.next(i)
            if i == i0:
                break
            pv = ProfileVert(self, i, roofMinHeight, noWalls)
            self.createProfileVertices(_pv, pv, roofMinHeight, noWalls)        
            _pv = pv
        # the closing part
        self.createProfileVertices(pv, pv0, roofMinHeight, noWalls)
        return True
    
    def createProfileVertices(self, pv1, pv2, roofMinHeight, noWalls):
        """
        Create vertices for profile reference points located
        with the indices between <pv1.index> and <pv2.index>
        """
        verts = self.verts
        indices = self.polygon.indices
        p = self.profile
        wallIndices = self.wallIndices
        
        index1 = pv1.index
        index2 = pv2.index
        
        skip1 = noWalls and pv1.onProfile and\
            ((self.lEndZero and not index1) or\
            (self.rEndZero and index1 == self.lastProfileIndex))
        skip2 = noWalls and pv2.onProfile and\
            ((self.lEndZero and not index2) or\
            (self.rEndZero and index2 == self.lastProfileIndex))
        
        if skip1 and skip2 and pv1.index == pv2.index:
            return
        _wallIndices = [pv1.vertIndex]
        if not skip1:
            _wallIndices.append(indices[pv1.i])
        if not skip2:
            _wallIndices.append(indices[pv2.i])
        _wallIndices.append(pv2.vertIndex)
        
        v1 = verts[indices[pv1.i]]
        v2 = verts[indices[pv2.i]]
        
        vertIndex = len(verts)
        if index1 != index2:
            for _i in (
                range(index2-1 if pv2.onProfile else index2, index1, -1)
                if index2 > index1 else
                range(index2+1, index1 if pv1.onProfile else index1+1)
            ):
                multiplier = (p[_i][0] - pv1.x) / (pv2.x - pv1.x)
                verts.append(Vector((
                    v1.x + multiplier * (v2.x - v1.x),
                    v1.y + multiplier * (v2.y - v1.y),
                    roofMinHeight + self.h * p[_i][1]
                )))
                _wallIndices.append(vertIndex)
                vertIndex += 1
        wallIndices.append(_wallIndices)
    
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