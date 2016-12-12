from mathutils import Vector
from util import zAxis, zeroVector


class Polygon:
    
    def __init__(self, indices, allVerts):
        n = len(indices)
        self.n = n
        self.allVerts = allVerts
        self.indices = indices
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
            # clockwise direction, reverse <indices> in place
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