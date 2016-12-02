import math
from mathutils import Vector
from util.osm import parseNumber
from . import Roof


class RoofSkillion(Roof):
    
    defaultHeight = 2.
    
    def init(self, element, osm):
        super().init(element, osm)
        self.projections = None
    
    def make(self, bldgMaxHeight, roofMinHeight, bldgMinHeight, osm):
        polygon = self.polygon
        if not self.projections:
            self.processDirection()
        projections = self.projections
        maxProj = projections[self.maxProjIndex]
        # update <polygon.allVerts> with vertices sheared along z-axis
        tan = self.h/self.polygonLength
        for _i in range(polygon.n):
            i = polygon.indices[_i]
            vert = polygon.allVerts[i].copy()
            vert.z = roofMinHeight + (maxProj - projections[_i]) * tan
            polygon.allVerts[i] = vert
        # <polygon.normal> won't be used, so it won't be updated
        
        self.sidesIndices = polygon.sidesShortestProjection(self.maxProjIndex) if bldgMinHeight is None else\
            polygon.sidesPrism(bldgMinHeight)
        return True
    
    def getHeight(self):
        element = self.element
        tags = element.tags
        
        if "roof:height" in tags:
            h = parseNumber(tags["roof:height"], self.defaultHeight)
        elif "roof:angle" in tags:
            angle = parseNumber(tags["roof:height"])
            if angle is None:
                h = self.defaultHeight
            else:
                self.processDirection(element, True)
                h = self.polygonLength * math.tan(math.radians(angle))
        else:
            h = self.defaultHeight
        
        self.h = h
        return h

    def processDirection(self):
        polygon = self.polygon
        # <d> stands for direction
        d = self.element.tags.get(
            "roof:slope:direction",
            self.element.tags.get("roof:direction")
        )
        # getting a direction vector with the unit length
        if d is None:
            d = self.getDefaultDirection()
        elif d in Roof.directions:
            d = Roof.directions[d]
        else:
            # trying to get a direction angle in degrees
            d = parseNumber(d)
            if d is None:
                d = self.getDefaultDirection()
            else:
                d = math.radians(d)
                d = Vector((math.sin(d), math.cos(d), 0.))
        
        # For each vertex from <polygon.verts> calculate projection of the vertex
        # on the vector <d> that defines the roof direction
        projections = [d.dot(v) for v in polygon.verts]
        self.projections = projections
        maxProjIndex = max(range(polygon.n), key = lambda i: projections[i])
        self.maxProjIndex = maxProjIndex
        self.polygonLength = projections[maxProjIndex] - min(projections)
    
    def getDefaultDirection(self):
        polygon = self.polygon
        # a perpendicular to the longest edge of the polygon
        return max(self.polygon.edges).cross(polygon.normal).normalized()