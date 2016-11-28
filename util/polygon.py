from util import zAxis


class Polygon:
    
    def __init__(self, indices, allVerts=None):
        n = len(indices)
        self.n = n
        if allVerts:
            self.allVerts = allVerts
            self.indices = indices
        else:
            self.allVerts = indices
            self.indices = list(range(n))
        # normal to the polygon
        self.normal = zAxis
    
    def checkDirection(self):
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
        # in that case the direction of vertices is counterclockwise, it's clockwise in the opposite case.
        if v1.x * v2.y - v1.y * v2.x < 0.:
            # clockwise direction, reverse <indices> in place
            indices.reverse()
    
    @property
    def verts(self):
        for i in self.indices:
            yield self.allVerts[i]
    
    @property
    def edges(self):
        verts = self.allVerts
        indices = self.indices
        # previous vertex
        _v = verts[indices[-1]]
        for i in indices:
            v = verts[indices[i]]
            yield v - _v
            _v = v