import mathutils

def _intersect_line2_line2(A, B):
    d = B.v.y * A.v.x - B.v.x * A.v.y
    if d == 0:
        return None

    dy = A.p.y - B.p.y
    dx = A.p.x - B.p.x
    ua = (B.v.x * dy - B.v.y * dx) / d
    if not A.intsecttest(ua):
        return None
    ub = (A.v.x * dy - A.v.y * dx) / d
    if not B.intsecttest(ub):
        return None

    return mathutils.Vector((A.p.x + ua * A.v.x, A.p.y + ua * A.v.y))

class Edge2:
    def __init__(self, p1, p2, norm=None, verts=None, center=mathutils.Vector((0,0))):  # evtl. conversion from 3D to 2D
        """
        Args:
            p1 (mathutils.Vector | int): Index of the first edge vertex if <verts> is given or
                the vector of the first edge vertex otherwise
            p2 (mathutils.Vector | int): Index of the second edge vertex if <verts> is given or
                the vector of the seconf edge vertex otherwise
            norm (mathutils.Vector): Normalized edge vector
            verts (list): Python list of vertices
        """
        if verts:
            self.i1 = p1
            self.i2 = p2
            p1 = verts[p1]-center
            p2 = verts[p2]-center
        self.p1 = mathutils.Vector((p1[0], p1[1]))
        self.p2 = mathutils.Vector((p2[0], p2[1]))
        if norm:
            self.norm = mathutils.Vector((norm[0], norm[1]))
        else:
            norm = self.p2-self.p1
            norm.normalize()
            self.norm = norm

class Ray2:
    def __init__(self, _p, _v):
        self.p = _p
        self.p1 = _p
        self.p2 = _p+_v
        self.v = _v

    def intsecttest(self,u):
        return u>=0.0

    def intersect(self,other):
        return _intersect_line2_line2(self,other)

class Line2:
    def __init__(self, p1, p2=None, ptype=None):
        if p2 is None: # p1 is a LineSegment2, a Line2 or a Ray2
            self.p = p1.p1.copy()
            self.v = (p1.p2-p1.p1).copy()
        elif ptype == 'pp':
            self.p = p1.p.copy()
            self.v = p2-p1
        elif ptype == 'pv':
            self.p = p1.copy()
            self.v = p2.copy()
        self.p1 = self.p
        self.p2 = self.p+self.v

    def intsecttest(self,u):
        return True

    def intersect(self,other):
        intsect = _intersect_line2_line2(self,other)
        return intsect

    def distance(self,other): # other is a vector
        nearest = mathutils.geometry.intersect_point_line(other, self.p, self.p+self.v)[0]
        dist = (other-nearest).length
        return dist
 