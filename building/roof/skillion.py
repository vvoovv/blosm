import math
from mathutils import Vector
from util import zero
from util.osm import parseNumber
from . import Roof


class RoofSkillion(Roof):
    
    defaultHeight = 2.
    
    def __init__(self):
        super().__init__()
        self.projections = []
    
    def init(self, element, minHeight, osm):
        super().init(element, minHeight, osm)
        self.projections.clear()
    
    def make(self, bldgMaxHeight, roofMinHeight, bldgMinHeight, osm):
        verts = self.verts
        polygon = self.polygon
        indices = polygon.indices
        n = polygon.n
        wallIndices = self.wallIndices
        
        self.roofIndices.append( tuple(range(0, n)) )
        
        if not self.projections:
            self.processDirection()
        
        projections = self.projections
        minZindex = self.maxProjIndex
        maxProj = projections[minZindex]
        tan = self.h/self.polygonLength
        # update <polygon.verts> with vertices moved along z-axis
        for i in range(polygon.n):
            verts[indices[i]].z = roofMinHeight + (maxProj - projections[i]) * tan
        # <polygon.normal> won't be used, so it won't be updated
        
        indexOffset = len(verts)
        if bldgMinHeight is None:
            # <roofMinHeight> is exactly equal to the height of the bottom part of the building
            # check height of the neighbors of the vertex with the index <minZindex>
            
            # index of the left neighbor
            leftIndex = polygon.prev(minZindex)
            # index of the right neighbor
            rightIndex = polygon.next(minZindex)
            if verts[ indices[leftIndex] ].z - roofMinHeight < zero:
                # Not only the vertex <minZindex> preserves its height,
                # but also its left neighbor
                rightIndex = minZindex
            elif verts[ indices[rightIndex] ].z - roofMinHeight < zero:
                # Not only the vertex <minZindex> preserves its height,
                # but also its right neighbor
                leftIndex = minZindex
            else:
                leftIndex = rightIndex = minZindex
            
            # starting from <rightIndex> walk counterclockwise along the polygon vertices till <leftIndex>
            
            # the current vertex index
            index = polygon.next(rightIndex)
            verts.append(Vector((
                verts[indices[index]].x,
                verts[indices[index]].y,
                roofMinHeight
            )))
            # a triangle that start at the vertex <rightIndex>
            wallIndices.append((indices[rightIndex], indexOffset, indices[index]))
            while True:
                prevIndex = index
                index = polygon.next(index)
                if index == leftIndex:
                    break
                # create a quadrangle
                verts.append(Vector((
                    verts[indices[index]].x,
                    verts[indices[index]].y,
                    roofMinHeight
                )))
                wallIndices.append((indexOffset, indexOffset + 1, indices[index], indices[prevIndex]))
                indexOffset += 1
            # a triangle that starts at the vertex <leftIndex> (all vertices for it are already available)
            wallIndices.append((indexOffset, indices[index], indices[prevIndex]))
        else:
            verts.extend(Vector((v.x, v.y, bldgMinHeight)) for v in polygon.verts)
            # the starting side
            wallIndices.append((indexOffset + n - 1, indexOffset, indices[0], indices[-1]))
            wallIndices.extend(
                (indexOffset + i - 1, indexOffset + i, indices[i], indices[i-1]) for i in range(1, n)
            )
        
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