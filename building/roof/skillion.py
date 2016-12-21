import math
from mathutils import Vector
from util import zero
from util.osm import parseNumber
from . import Roof


class RoofSkillion(Roof):
    """
    A class to deal with buildings or building parts with a skillion roof
    
    Direction vector of the roof is pointing to the lower part of the roof,
    perpendicular to the horinontal line that the roof plane contains.
    In other words the direction vector is pointing from the top to the bottom of the roof.
    """
    
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
        
        # simply take <polygon> indices for the roof
        self.roofIndices.append( tuple(indices[i] for i in range(0, n)) )
        
        if not self.projections:
            self.processDirection()
        
        projections = self.projections
        minZindex = self.maxProjIndex
        maxProj = projections[minZindex]
        tan = self.h/self.polygonWidth
        # update <polygon.verts> with vertices moved along z-axis
        for i in range(n):
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
            
            # Starting from <rightIndex> walk counterclockwise along the polygon vertices
            # till <leftIndex>
            
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
            # vertices for the bottom part
            verts.extend(Vector((v.x, v.y, bldgMinHeight)) for v in polygon.verts)
            # the starting wall side
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
            angle = parseNumber(tags["roof:angle"])
            if angle is None:
                h = self.defaultHeight
            else:
                self.processDirection()
                h = self.polygonWidth * math.tan(math.radians(angle))
        else:
            h = self.defaultHeight
        
        self.h = h
        return h