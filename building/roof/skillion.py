"""
This file is part of blender-osm (OpenStreetMap importer for Blender).
Copyright (C) 2014-2017 Vladimir Elistratov
prokitektura+support@gmail.com

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from mathutils import Vector
from util import zero
from . import Roof
from .profile import RoofProfile


class RoofSkillion(Roof):
    """
    A class to deal with buildings or building parts with a skillion roof
    
    Direction vector of the roof is pointing to the lower part of the roof,
    perpendicular to the horizontal line that the roof plane contains.
    In other words the direction vector is pointing from the top to the bottom of the roof.
    """
    
    defaultHeight = 2.
    
    def __init__(self):
        super().__init__()
        self.hasRidge = False
        self.projections = []
        self.angleToHeight = 1.
    
    def init(self, element, data, minHeight, osm):
        super().init(element, data, minHeight, osm)
        self.projections.clear()
    
    def make(self, bldgMaxHeight, roofMinHeight, bldgMinHeight, osm):
        verts = self.verts
        polygon = self.polygon
        indices = polygon.indices
        n = polygon.n
        wallIndices = self.wallIndices
        
        # simply take <polygon> indices for the roof
        self.roofIndices.append(indices)
        
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
            # the starting wall face
            wallIndices.append((indexOffset + n - 1, indexOffset, indices[0], indices[-1]))
            wallIndices.extend(
                (indexOffset + i - 1, indexOffset + i, indices[i], indices[i-1]) for i in range(1, n)
            )
        
        return True
    
    def getHeight(self, op):
        return RoofProfile.getHeight(self, op)