import mathutils

# -------------------------------------------------------------------------
# from https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/,
# works fine with mathutils.Vector
def ccw(A,B,C):
    return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)
# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
# -------------------------------------------------------------------------

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

    def length_squared(self):
        return (self.p2-self.p1).length_squared

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

def fitCircle3Points(points):
    N = len(points)
    # circle through three points usingg complex math, see answer in 
    # https://stackoverflow.com/questions/28910718/give-3-points-and-a-plot-circle
    x = complex(points[0].x, points[0].y)
    y = complex(points[N//2].x, points[N//2].y)
    z = complex(points[-1].x, points[-1].y)
    w = z-x
    w /= y-x
    c = (x-y)*(w-abs(w)**2)/2j/w.imag-x
    x0 = -c.real
    y0 = -c.imag
    R = abs(c+x)
    return mathutils.Vector((x0,y0)), R
