"""
This file is part of blender-osm (OpenStreetMap importer for Blender).
Copyright (C) 2014-2018 Vladimir Elistratov
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
from util import zero, zAxis, zeroVector


class PolygonOLD:
    
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
    
    @property
    def area(self):
        verts = self.allVerts
        indices = self.indices
        vertFirst = verts[indices[0]]
        vertLast = verts[indices[-1]]
        # the shoelace formula https://en.wikipedia.org/wiki/Shoelace_formula
        return 0.5 * abs(
            sum( (v0[0]*v1[1] - v1[0]*v0[1]) for (v0, v1) in\
                ( (verts[indices[i]],verts[indices[i+1]]) for i in range(self.n-1))
            ) + vertLast[0]*vertFirst[1] - vertFirst[0]*vertLast[1]
        )
    
    def extrude(self, z, indices):
        """
        Extrude the polygon along <z>-axis to the target height <z>
        
        Extruded vertices are appended to <self.allVerts>.
        Vertex indices for the extruded sides are appended to <indices>
        
        Args:
            z (float): The target height of the extruded part
            indices (list): A python list to append vertex indices for the exruded sides
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

    def inset(self, distances, indices, height=None, negate=False):
        """
        Args:
            distances (float | list | tuple): inset values
            indices (list): A python list to append vertex indices for the inset faces
        """
        verts = self.allVerts
        _indices = self.indices
        indexOffset = _indexOffset = len(verts)
        
        translate = height*self.normal if not height is None else None
        
        #distancePerEdge = False if len(distances)==1 else True
        distancePerEdge = False
        
        if distancePerEdge:
            distance1 = distances[-1]
            distance2 = distances[0]
        else:
            #distance1 = distances[0]
            #distance2 = distance1
            distance1 = distances
            distance2 = distances
        _d = distance1
        prevVert = verts[_indices[0]]
        edge1 = _edge1 = Edge(verts[_indices[0]] - verts[_indices[-1]], self.normal)
        prevIndex1 = _indices[-1]
        prevIndex2 = indexOffset + self.n - 1
        
        for i in range(self.n-1):
            index1 = _indices[i]
            vert = verts[_indices[i+1]]
            # vector along the edge
            edge2 = Edge(vert - prevVert, self.normal)
            if distancePerEdge:
                distance1 = distance2
                distance2 = distances[i]
            self.insetVert(i, edge1, edge2, distance1, distance2, translate, negate)
            index2 = indexOffset
            indexOffset += 1
            if distance1:
                indices.append((prevIndex1, index1, index2, prevIndex2))
            prevVert = vert
            edge1 = edge2
            prevIndex1 = index1
            prevIndex2 = index2
            i += 1
        if not distancePerEdge or _d:
            edge2 = _edge1
            distance2 = _d
            self.insetVert(-1, edge1, edge2, distance1, distance2, translate, negate)
            indices.append((prevIndex1, _indices[-1], indexOffset, prevIndex2))
        # new values for the polygon indices
        #self.indices = tuple(_indexOffset + i for i in range(self.n))
    
    def insetVert(self, index, edge1, edge2, d1, d2, translate=None, negate=False):
        vert = self.allVerts[self.indices[index]]
        
        if not d1 and not d2 and translate:
            vert = vert + translate
        else:
            if negate:
                d1 = -d1
                d2 = -d2
            
            # cross product between edge1 and edge1
            cross = edge1.vec.cross(edge2.vec)
            # To check if have a concave (>180) or convex angle (<180) between edge1 and edge2
            # we calculate dot product between cross and axis
            # If the dot product is positive, we have a convex angle (<180), otherwise concave (>180)
            dot = cross.dot(self.normal)
            convex = True if dot>0 else False
            # sine of the angle between <-edge1.vec> and <edge2.vec>
            sin = cross.length
            isLine = True if sin<zero and convex else False
            if not isLine:
                sin = sin if convex else -sin
                # cosine of the angle between <-edge1.vec> and <edge2.vec>
                cos = -(edge1.vec.dot(edge2.vec))
            
            # extruded counterpart of <vert>
            vert = vert - d1*edge1.normal - (d2+d1*cos)/sin*edge1.vec
            if translate:
                vert = vert + translate
            self.allVerts.append(vert)


class Polygon:
    
    def __init__(self):
        self.allVerts = []
        self.indices = None
        # normal to the polygon
        self.normal = zAxis
        self._maxEdgeIndex = None
        self.reversed = False
    
    def init(self, allVerts):
        """
        Args:
            allVerts (generator): Polygon vertices
        """
        self._maxEdgeIndex = None
        if self.reversed:
            self.reversed = False
        self.allVerts.clear()
        self.allVerts.extend(allVerts)
        # Not all vertices from <allVerts> will be used to create BMesh vertices,
        # since they may have a straight angle.
        # Later new vertices may be added to <allVerts>, for each of those vertices
        # a BMesh vertex will be created. To distinguish between those two groups of <allVerts>,
        # we need to keep the border between them as <self.indexOffset>
        #self.indexOffset = len(allVerts)
        self.n = len(self.allVerts)
        self.removeStraightAngles()
    
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
        if self.directionCondition(v1, v2):
            # clockwise direction, reverse <indices>
            self.indices = tuple(reversed(indices))
            self.reversed = True
    
    def directionCondition(self, v1, v2):
        return v1.x * v2.y - v1.y * v2.x < 0.
    
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
    
    def center(self, z=0.):
        """
        Returns geometric center of the polygon
        """
        center = sum(self.verts, zeroVector())/self.n
        center[2] = z
        return center
    
    def centerBB(self, z=0.):
        """
        Return the center of the polygon bounding box alligned along the global X and Y axes
        """
        return Vector((
            ( min(self.verts, key=lambda v: v[0])[0] + max(self.verts, key=lambda v: v[0])[0] )/2.,
            ( min(self.verts, key=lambda v: v[1])[1] + max(self.verts, key=lambda v: v[1])[1] )/2.,
            z
        ))

    def middleOfTheLongestSide(self, z=0.):
        """
        Return the middle of the polygon's longest side 
        it can be used as a base point for generatrix for half-dome (common russian element) 
        and half-pyramid (common gothic element)   
        """
        verts = self.allVerts
        indices = self.indices
        maxEdgeIndex = self.maxEdgeIndex
        middle = ( verts[indices[maxEdgeIndex+1]] + verts[indices[maxEdgeIndex]] )/2.
        return Vector((middle[0],middle[1],z))
    
    def area(self):
        verts = self.allVerts
        indices = self.indices
        vertFirst = verts[indices[0]]
        vertLast = verts[indices[-1]]
        # the shoelace formula https://en.wikipedia.org/wiki/Shoelace_formula
        return 0.5 * abs(
            sum( (v0[0]*v1[1] - v1[0]*v0[1]) for (v0, v1) in\
                ( (verts[indices[i]],verts[indices[i+1]]) for i in range(self.n-1))
            ) + vertLast[0]*vertFirst[1] - vertFirst[0]*vertLast[1]
        )
    
    def extrude(self, z, indices):
        """
        Extrude the polygon along <z>-axis to the target height <z>
        
        Extruded vertices are appended to <self.allVerts>.
        Vertex indices for the extruded sides are appended to <indices>
        
        Args:
            z (float): The target height of the extruded part
            indices (list): A python list to append vertex indices for the exruded sides
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
        ux = v_[0] - v[0]
        uy = v_[1] - v[1]
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
            ux = v_[0] - v[0]
            uy = v_[1] - v[1]
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
                # therefore we add the current vertex index to <newIndices>
                newIndices.append(indices[i] if indices else i)
        if newIndices is None:
            # no straight angles found
            self.indices = tuple(range(self.n))
        else:
            # set new indices without straight angles to <self.indices>
            self.indices = newIndices
            # calculate the new number of vertices in the polygon
            self.n = len(newIndices)

    def inset(self, distances, indices, height=None, negate=False):
        """
        Args:
            distances (float | list | tuple): inset values
            indices (list): A python list to append vertex indices for the inset faces
        """
        verts = self.allVerts
        _indices = self.indices
        indexOffset = _indexOffset = len(verts)
        
        translate = height*self.normal if not height is None else None
        
        #distancePerEdge = False if len(distances)==1 else True
        distancePerEdge = False
        
        if distancePerEdge:
            distance1 = distances[-1]
            distance2 = distances[0]
        else:
            #distance1 = distances[0]
            #distance2 = distance1
            distance1 = distances
            distance2 = distances
        _d = distance1
        prevVert = verts[_indices[0]]
        edge1 = _edge1 = Edge(verts[_indices[0]] - verts[_indices[-1]], self.normal)
        prevIndex1 = _indices[-1]
        prevIndex2 = indexOffset + self.n - 1
        
        for i in range(self.n-1):
            index1 = _indices[i]
            vert = verts[_indices[i+1]]
            # vector along the edge
            edge2 = Edge(vert - prevVert, self.normal)
            if distancePerEdge:
                distance1 = distance2
                distance2 = distances[i]
            self.insetVert(i, edge1, edge2, distance1, distance2, translate, negate)
            index2 = indexOffset
            indexOffset += 1
            if distance1:
                indices.append((prevIndex1, index1, index2, prevIndex2))
            prevVert = vert
            edge1 = edge2
            prevIndex1 = index1
            prevIndex2 = index2
            i += 1
        if not distancePerEdge or _d:
            edge2 = _edge1
            distance2 = _d
            self.insetVert(-1, edge1, edge2, distance1, distance2, translate, negate)
            indices.append((prevIndex1, _indices[-1], indexOffset, prevIndex2))
        # new values for the polygon indices
        #self.indices = tuple(_indexOffset + i for i in range(self.n))
    
    def insetVert(self, index, edge1, edge2, d1, d2, translate=None, negate=False):
        vert = self.allVerts[self.indices[index]]
        
        if not d1 and not d2 and translate:
            vert = vert + translate
        else:
            if negate:
                d1 = -d1
                d2 = -d2
            
            # cross product between edge1 and edge1
            cross = edge1.vec.cross(edge2.vec)
            # To check if have a concave (>180) or convex angle (<180) between edge1 and edge2
            # we calculate dot product between cross and axis
            # If the dot product is positive, we have a convex angle (<180), otherwise concave (>180)
            dot = cross.dot(self.normal)
            convex = True if dot>0 else False
            # sine of the angle between <-edge1.vec> and <edge2.vec>
            sin = cross.length
            isLine = True if sin<zero and convex else False
            if not isLine:
                sin = sin if convex else -sin
                # cosine of the angle between <-edge1.vec> and <edge2.vec>
                cos = -(edge1.vec.dot(edge2.vec))
            
            # extruded counterpart of <vert>
            vert = vert - d1*edge1.normal - (d2+d1*cos)/sin*edge1.vec
            if translate:
                vert = vert + translate
            self.allVerts.append(vert)
    
    @property
    def maxEdgeIndex(self):
        """
        Returns -1 if the last edge is the longest one
        """
        if self._maxEdgeIndex is None:
            verts = self.allVerts
            indices = self.indices
            self._maxEdgeIndex = max(
                range(-1, self.n-1),
                key = lambda i: (verts[indices[i+1]]-verts[indices[i]]).length_squared
            )
        return self._maxEdgeIndex
    
    def setHeight(self, height):
        """
        Set height for the polygon
        """
        # check if the polygon already has that <height>
        if self.allVerts[self.indices[0]][2] == height:
            return
        for i in self.indices:
            self.allVerts[i][2] = height


class Edge:
    
    def __init__(self, vec, polygonNormal):
        vec.normalize()
        self.vec = vec
        normal = vec.cross(polygonNormal)
        normal.normalize()
        self.normal = normal


class PolygonCW(Polygon):
    """
    A polygon with clockwise order of vertices used for the holes in a multipolygon
    """
    def directionCondition(self, v1, v2):
        return v1.x * v2.y - v1.y * v2.x > 0.