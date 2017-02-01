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
from util import zAxis, zeroVector


class Polygon:
    
    def __init__(self, allVerts, indices=None, removeStraightAngles=True):
        self.allVerts = allVerts
        # Not all vertices from <allVerts> will be used to create BMesh vertices,
        # since they may have a straight angle.
        # Later new vertices may be added to <allVerts>, for each of those vertices
        # a BMesh vertex will be created. To distinguish between those two groups of <allVerts>,
        # we need to keep the border between them as <self.indexOffset>
        self.indexOffset = len(allVerts)
        self.indices = indices
        self.n = len(indices if indices else allVerts)
        self.removeStraightAngles()
        # normal to the polygon
        self.normal = zAxis
    
    def prev(self, index):
        """
        Returns the previous index for <index>
        
        Args:
            index (int): A number between 0 and <self.n - 1>
        """
        return index - 1 if index else self.n - 1
    
    def next(self, index):
        """
        Returns the next index for <index>
        
        Args:
            index (int): A number between 0 and <self.n - 1>
        """
        return (index+1) % self.n
    
    def checkDirection(self):
        """
        Check direction of the polygon vertices and
        force their direction to be counterclockwise
        """
        verts = self.allVerts
        indices = self.indices
        # find indices of vertices with the minimum y-coordinate (the lowest vertices)
        minY = verts[ min(indices, key = lambda i: verts[i].y) ].y
        _indices = [i for i in range(self.n) if verts[indices[i]].y == minY]
        # For the vertices with minimum y-coordinate,
        # find the index of the vertex with maximum x-coordinate (he rightmost lowest vertex)
        _i = _indices[0] if len(_indices) == 1 else max(_indices, key = lambda i: verts[indices[i]].x)
        i = indices[_i]
        # the edge entering the vertex <verts[i]>
        v1 = verts[i] - verts[ indices[_i-1] ]
        # the edge leaving the vertex <verts[i]>
        v2 = verts[ indices[(_i+1) % self.n] ] - verts[i]
        # Check if the vector <v2> is to the left from the vector <v1>;
        # in that case the direction of vertices is counterclockwise,
        # it's clockwise in the opposite case.
        if v1.x * v2.y - v1.y * v2.x < 0.:
            # clockwise direction, reverse <indices>
            self.indices = tuple(reversed(indices))
    
    @property
    def verts(self):
        """
        A Python generator for the polygon vertices
        """
        for i in self.indices:
            yield self.allVerts[i]
    
    @property
    def edges(self):
        """
        A Python generator for the polygon edges represented as vectors
        """
        verts = self.allVerts
        indices = self.indices
        # previous vertex
        _v = verts[indices[-1]]
        for i in indices:
            v = verts[i]
            yield v - _v
            _v = v
    
    @property
    def center(self):
        """
        Returns geometric center of the polygon
        """
        return sum(tuple(self.verts), zeroVector())/self.n
    
    def sidesPrism(self, z, indices):
        """
        Create sides for the prism with the height <z - <polygon height>>,
        that is based on the polygon.
        
        Vertices for the top part of the prism are appended to <self.allVerts>.
        Vertex indices for the prism sides are appended to <indices>
        
        Args:
            z (float): Vertical location of top part of the prism
            indices (list): A python list to append vertex indices for the prism sides
        """
        verts = self.allVerts
        _indices = self.indices
        indexOffset = len(verts)
        # verts
        verts.extend(Vector((v.x, v.y, z)) for v in self.verts)
        # the starting side
        indices.append((_indices[-1], _indices[0], indexOffset, indexOffset + self.n - 1))
        indices.extend(
            (_indices[i-1], _indices[i], indexOffset + i, indexOffset + i - 1) for i in range(1, self.n)
        )
    
    def removeStraightAngles(self):
        """
        Given <verts> constituting a polygon, removes vertices forming a straight angle
        """
        verts = self.allVerts
        indices = self.indices
        # Create the Python list <newIndices> only if it's really necessary,
        # i.e. there are straight angles in the polygon 
        newIndices = None
        # <v> denotes the beginning of the vector along a polygon side
        v = verts[indices[-1] if indices else -1]
        # <v_> denotes the end of the vector along the polygon side
        v_ = verts[indices[0] if indices else 0]
        # x and y components of the vector <u> = <v_> - <v>
        ux = v_.x - v.x
        uy = v_.y - v.y
        # the last index
        i_ = self.n-1
        for i in range(self.n):
            v = v_
            # x and y components of the vector <_u> along the previous side of the polygon
            _ux = ux
            _uy = uy
            v_ = verts[
                (indices[0] if indices else 0)\
                if i == i_ else\
                (indices[i+1] if indices else i+1)
            ]
            ux = v_.x - v.x
            uy = v_.y - v.y
            # dot product of the vectors <_u> and <u>
            dot = _ux*ux + _uy*uy
            # Check if tangent of angle between the vectors <_u> and <u> is nearly equal to zero;
            # the tangent can be calculated as ration of cross product and dot product
            if dot and abs((_ux*uy - _uy*ux)/dot) < Polygon.straightAngleTan:
                if newIndices is None:
                    # Found the first straight angle in the polygon.
                    # Copy indices for the non-straight angles that we encountered before to <newIndices>
                    newIndices = [indices[_i] for _i in range(i)] if indices else [_i for _i in range(i)]
            elif not newIndices is None:
                # We encountered a straight angle before,
                # therefore we and the current vertex index to <newIndices>
                newIndices.append(indices[i] if indices else i)
        if newIndices is None:
            # no straight angles found
            if not indices:
                self.indices = tuple(range(self.n))
        else:
            # set new indices without straight angles to <self.indices>
            self.indices = newIndices
            # calculate the new number of vertices in the polygon
            self.n = len(newIndices)