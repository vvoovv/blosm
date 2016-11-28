import math
from util import zAxis
from util.polygon import Polygon
from util.osm import parseNumber
from renderer import Renderer
from . import Roof


class RoofSkillion:
    
    defaultHeight = 2.
    
    def init(self, element, osm):
        self.element = element
        self.polygon = Polygon(
            element.getData(osm) if element.t is Renderer.polygon else element.getOuterData(osm)
        )
        # check the direction of vertices, it must be counterclockwise
        self.polygon.checkDirection()
        self.projections = None
    
    def make(self):
        polygon = self.polygon
        if not self.projections:
            self.processDirection()
        # update <polygon.allVerts> with vertices sheared along z-axis
        tan = self.h/self.polygonLength
        for i in range(polygon.n):
            _i = polygon.indices[i]
            polygon.allVerts[_i] = polygon.allVerts[_i] + (self.maxProj - self.projections[i]) * tan * zAxis
        # <polygon.normal> won't be used, so it won't be updated
    
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
        d = self.element.tags.get("roof:direction")
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
                d = (math.sin(d), math.cos(d), 0.)
        
        # For each vertex from <polygon.verts> calculate projection of the vertex
        # on the vector <d> that defines the roof direction
        projections = [d.dot(v) for v in polygon.verts]
        self.projections = projections
        self.maxProj = max(projections)
        self.polygonLength = self.maxProj - min(projections)
    
    def getDefaultDirection(self):
        polygon = self.polygon
        # a perpendicular to the longest edge of the polygon
        return polygon.normal.cross( max(self.polygon.edges) ).normalized()