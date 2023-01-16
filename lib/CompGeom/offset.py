#   Offseting of polygons and lines, based on:
#
#         Xiaorui Chen and Sara McMains. Polygon Offsetting by Computing Winding Numbers
#         Proceedings of IDETC/CIE 2005. ASME 2005 International Design Engineering Technical Conferences &
#         Computers and Information in Engineering Conference
#         https://mcmains.me.berkeley.edu/pubs/DAC05OffsetPolygon.pdf
#
# Polygons:
#             def offsetPolygon(polygon, offset, precision=8)
#                 polygon:   A Python list of polygon 2D vertices of type matutils.Vector, tuple or of the
#                            internal class Vertex.
#                 offset:    The distance to offset. Positive values will grow the polygon, negative values will shrink it.
#                 precision: If integer: Number of vertices per quadrant for arcs, default precision=8 .
#                            If float: Maximum error between chord and circle of arc.
#                 return:    A Python list of the offset polygon(s). The type of the vertices of type matutils.Vector.
#                            The outer border is ordered counter-clockwise and holes are ordered clockwise.
#
#             Example usage:
#                           poly = [Vector((0.0,0.0)),Vector((1.0,0.1)),Vector((1.0,1.0)),Vector((0.75,0.4))]
#                           offPoly = offsetPolygon(poly,0.1)
#
# Polylines:
#             def offsetLine(line, offset, precision=8):
#                 line:      A Python list of 2D vertices of type matutils.Vector, tuple or the internal
#                            class Vertex.
#                 offset:    The distance to offset. Only positive values are allowed.
#                 precision: If integer: Number of vertices per quadrant for arcs, default precision=8 .
#                            If float: Maximum error between chord and circle of arc.
#                 return:    A Python list of the offset polygon(s). The type of the vertices of type matutils.Vector.
#                            The outer border is ordered counter-clockwise and holes are ordered clockwise.
#
#             Example usage:
#                           line = [Vector((2,5)),Vector((4,10)),Vector((0,10)),Vector((0,0)),Vector((10,0)),
#                                   Vector((10,10)),Vector((6,10)),Vector((8,5))]
#                           offPoly = offsetLine(line,3.)

from itertools import tee, islice, cycle
from collections import defaultdict
from mathutils import Vector
from math import ceil, isclose, sqrt, sin, cos, acos, atan2, pi

# Internal class -------------------------------------------------
class Vertex():
    def __init__(self,x,y):
        self._x = x
        self._y = y

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def length(self):
        return sqrt(self._x * self._x + self._y * self._y)

    @property
    def length_squared(self):
        return self._x * self._x + self._y * self._y

    def __getitem__(self, key):
        if key == 0: return self._x
        elif key == 1: return self._y
        else: raise KeyError('Invalid key: %s. Valid keys are 0 and 1 for x and y' % key)

    def __add__(self, other):
        return Vertex(self._x + other._x, self._y + other._y)

    def __sub__(self, other):
        return Vertex(self._x - other._x, self._y - other._y)

    def __mul__(self, val):
        return Vertex(self._x * val, self._y * val)

    def __div__(self, val):
        return Vertex(self._x / val, self._y / val)

    def __truediv__(self, val):
        return Vertex(self._x / val, self._y / val)

    def __eq__(self,other):
        if not isinstance(other, Vertex): return False
        d = self - other
        return isclose(d[0],0.,abs_tol=1.e-5) and isclose(d[1],0.,abs_tol=1.e-5)

    def __repr__(self):
        return 'Vertex: (%5.3f, %5.2f)'%(self.x,self.y)

    def isclose(self,other,abs_tol):
        d = (self - other).length
        return isclose(d,0.,abs_tol=abs_tol)

    def __hash__(self):
        return hash( (self._x,self._y) )

    def normalize(self):
        return self/sqrt(self._x * self._x + self._y * self._y)

    def normal(self):
        return Vertex(self._y,-self._x)

    def asTuple(self):
        return (self._x,self._y)

    def asVector(self):
        return Vector((self._x,self._y))

# helper functions -----------------------------------------------
def _cyclePair(iterable):
    # iterable -> (p0,p1), (p1,p2), (p2, p3), ..., (pn, p0)
    prevs, nexts = tee(iterable)
    prevs = islice(cycle(prevs), len(iterable) - 1, None)
    return zip(prevs,nexts)

def _cycleTriples(iterable):
    # iterable -> (pn,p0,p1), (p0,p1,p2), (p1,p2, p3), ..., (pn-1,pn,p0)
    preds, this, succs = tee(iterable, 3)
    preds = islice(cycle(preds), len(iterable) - 1, None)
    succs = islice(cycle(succs), 1, None)
    return zip(preds, this, succs)

def _isCCW(poly,eps=0.):
    # True, if polygon oriented counter-clockwise
    return sum( (p2[0]-p1[0])*(p2[1]+p1[1]) for p1,p2 in _cyclePair(poly)) < eps

def _ccw(a,b,c,eps=0.):
	# Returns the orientation of the triangle a, b, c.
	# Return True if a,b,c are oriented counter-clockwise.
	return (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]) < eps

def _isPointInTriangle(p, a,b,c):
	to = _ccw(a,b,c)
	return _ccw(a,b,p) == to and _ccw(b,c,p) == to and _ccw(a,p,c) == to

def _isectLineLine_params(p1, p2, q1, q2):
    # intersect line - line
	d = (q2[1] - q1[1]) * (p2[0] - p1[0]) - (q2[0] - q1[0]) * (p2[1] - p1[1])
	n1 = (q2[0] - q1[0]) * (p1[1] - q1[1]) - (q2[1] - q1[1]) * (p1[0] - q1[0])
	n2 = (p2[0] - p1[0]) * (p1[1] - q1[1]) - (p2[1] - p1[1]) * (p1[0] - q1[0])
	if d == 0: return None
	u_a = float(n1) / d
	u_b = float(n2) / d
	return (u_a, u_b)

def _isectSegRay(p1, p2, q1, q2):
    ll = _isectLineLine_params(p1, p2, q1, q2)
    if ll == None: return None
    if ll[0] < 0. or ll[0] > 1.: return None
    if ll[1] < 0.: return None
    return Vertex( p1[0] + ll[0] * (p2[0] - p1[0]) , p1[1] + ll[0] * (p2[1] - p1[1]) )

def _isectSegSeg(p1,p2,p3,p4):
    # Get the intersection of segment <p1,p2> with <p3,p4>.
    # None if no intersection or end-point intersection.
    d1, d2 = (p2[0]-p1[0],p2[1]-p1[1]), (p4[0]-p3[0],p4[1]-p3[1])#p2-p1, p4-p3
    cross = d1[0] * d2[1] - d2[0] * d1[1] # d1.cross(d2)
    if cross == 0.:
        return None
    d3 = (p1[0]-p3[0],p1[1]-p3[1]) # p1-p3
    t1 = (d2[0]*d3[1] - d2[1]*d3[0])/cross
    t2 = (d1[0]*d3[1] - d1[1]*d3[0])/cross
    # if 0. <= t1 <= 1. and 0. <= t2 <= 1:
    EPS = 1.e-3
    if -EPS <= t1 <= 1.+EPS and -EPS <= t2 <= 1+EPS:
        return Vertex( (p1[0] + d1[0]*t1),(p1[1] + d1[1]*t1) ) 
    else:
        return None

def _arcDecorator(p0,c,p1,radius,precision=8):
    # Create an arc by line segments, <precision> is the maximum distance from any line segment
    # to the arc it is approximating. Line segments are circumscribed by the arc (all end
    # points lie on the arc path).
    # p1:           start point
    # c:            center of arc
    # p2:           end point
    # radius:       radius of the arc
    # precision:    If integer: Number of vertices per quadrant for arcs, default precision=8 .
    #               If float: Maximum error between chord and circle of arc.
    # Return:       Python list of arc vertices, where the vertices p0 and p1 are included.
    startAngle = atan2(p0.y-c.y, p0.x-c.x)
    endAngle   = atan2(p1.y-c.y, p1.x-c.x)

    if _ccw(p0,c,p1,1.e-5): # counter-clockwise
        orientation = 1.
        if startAngle >= endAngle:
            startAngle -= 2.0 * pi
    else:
        orientation = -1.
        if startAngle <= endAngle:
            startAngle += 2.0 * pi

    totalAngle = abs(startAngle - endAngle)
    if isinstance(precision,float):
        segmentSubAngle = abs(2. * acos(1. - precision / abs(radius)))
        segmentCount = ceil(totalAngle / segmentSubAngle)
        angleInc = totalAngle / segmentCount
    else:
        segmentSubAngle = pi / 2.0 / precision
        segmentCount = ceil(totalAngle / segmentSubAngle)
        angleInc = totalAngle / segmentCount

    arcList = []
    currAngle = 0.0
    rad = abs(radius)
    while currAngle <= totalAngle:
        angle = startAngle + orientation * currAngle
        x = c[0] + (rad * cos(angle))
        y = c[1] + (rad * sin(angle))
        arcList.append( Vertex(x,y) )
        currAngle += angleInc
    return arcList
    
def _offsetPoly(poly,offset,precision=8):
    raw = []
    for p0,p1,p2 in _cycleTriples(poly):
        is_convex = _ccw(p0,p1,p2)
        # compute unit normals of segments
        un0 = (p1-p0).normalize().normal()
        un1 = (p2-p1).normalize().normal()
        off1 = p1 + un0*offset
        off2 = p1 + un1*offset
        if is_convex == (offset > 0):
            raw.append(off1)
            raw.append(p1)
            raw.append(off2)
        else:
            if off1.isclose(off2,1.e-3):
                raw.append(off1)
            else:
                raw.extend( _arcDecorator(off1,p1,off2,offset,precision) )
    return raw

def _decompose(poly,offset):
    # Decompose a possibly self-intersecting polygon into multiple simple polygons.
    vertices = [p for p in poly]
    isectVerts = []
    # find self-intersections
    ints = defaultdict(list)
    for i in range(len(vertices)):
        for j in range(i+1, len(vertices)):
            a = vertices[i]
            b = vertices[(i+1)%len(vertices)]
            c = vertices[j]
            d = vertices[(j+1)%len(vertices)]

            x = _isectSegSeg(a, b, c, d)
            if x:
                if x.isclose(a,1.e-5): x = a
                if x.isclose(b,1.e-5): x = b
                if x.isclose(c,1.e-5): x = c
                if x.isclose(d,1.e-5): x = d
                if x not in (a,b):
                    ints[(a.asTuple(),b.asTuple())].append( x )
                    if x not in isectVerts: isectVerts.append(x)
                if x not in (c,d):
                    ints[(c.asTuple(),d.asTuple())].append( x )
                    if x not in isectVerts: isectVerts.append(x)

    # add self-intersection vertices to poly vertices
    for k, v in ints.items():
        _inorderExtend(vertices, Vertex(*k[0]), Vertex(*k[1]), v)

    # roll vertices so that we start with the first intersection vertex.
    first = vertices.index(isectVerts[0])
    raw = vertices[first:] + vertices[:first+1]
    raw.reverse()

    if True:
        # build a list of loops
        loops = []
        while raw:
            # build up a list of seen points until we re-visit one - a loop!
            seen = []
            for p in raw + [raw[0]]:
                if p not in seen:
                    seen.append(p)
                else:
                    break
            loop = seen[seen.index(p):]

            # remove the loop from pts
            for p in loop:
                raw.remove(p)
            loops.append(loop)
        return loops

def _findPointInPoly(poly):
    # Subject 2.06 in https://web.archive.org/web/20120615051829/http://www.exaflop.org/docs/cgafaq/cga2.html
    if len(poly)==3:
        return sum(poly,Vertex(0.,0.))/3.

    # find a convex vertex v
    is_ccw  = _isCCW(poly)
    for a, v, b in _cycleTriples(poly):
        if _ccw(a,v,b) != is_ccw:
            break

    # For each other vertex p ... 
    p_inside = [ p for p in poly if p not in [a,v,b] and _isPointInTriangle(p, a,v,b) ]

    # ... if p is inside avb, pv is internal: return its midpoint
    if p_inside:
        return (p_inside[0] + v)/2.
    # else (no point is inside), return midpoint of ab
    else: 
        return (a + b)/2.

def _windingNumber(p, poly):
    # compute winding number of point
    #http://softsurfer.com/Archive/algorithm_0103/algorithm_0103.htm
    wn = 0
    if True:
        for a,b in _cyclePair(poly):
            if a[1] <= p[1] and b[1] >= p[1]:
                i = _isectSegRay(a,b,p,p+Vertex(1.,0.))
                if i and i[0] > p[0]:
                    wn -= 1
            if a[1] >= p[1] and b[1] <= p[1]:
                i = _isectSegRay(a,b,p,p+Vertex(1.,0.))
                if i and i[0] > p[0]:
                    wn += 1
    return wn

def _inorderExtend(seq, v1, v2, points):
    # Extend a sequence <seq> by <points> that are
    # between the points v1, v2
    k, r = None, False
    if   v1[0] < v2[0]: k, r = lambda i: i[0], True
    elif v1[0] > v2[0]: k, r = lambda i: i[0], False
    elif v1[1] < v2[1]: k, r = lambda i: i[1], True
    else:               k, r = lambda i: i[1], False
    l = sorted(points, key=k, reverse=r)
    i = next((i for i, p in enumerate(seq) if p == v2), -1)
    assert(i>=0)
    for e in l:
        seq.insert(i, e)
    return seq
# ----------------------------------------------------------------

def offsetPolygon(polygon, offset, precision=8):
    # polygon:   A Python list of polygon 2D vertices of type matutils.Vector, tuple or the internal
    #            class Vertex.
    # offset:    The distance to offset. Positive values will grow the polygon, negative values will shrink it.
    # precision: If integer: Number of vertices per quadrant for arcs, default precision=8 .
    #            If float: Maximum error between chord and circle of arc.
    # return:    A Python list of the offset polygon(s). The type of the vertices of type matutils.Vector.
    #            The outer border is ordered counter-clockwise and holes are ordered clockwise.

    if offset == 0: return polygon

    if isinstance(polygon[0],Vector):
        poly = [Vertex(v[0],v[1]) for v in polygon]
    elif isinstance(polygon[0],tuple):
        poly = [Vertex(v[0],v[1]) for v in polygon]
    else:
        poly = polygon

    # assert that polygon is ordered counter-clockwise
    if _isCCW(poly):
        offPoly = _offsetPoly(poly,offset,precision)
    else:
        offPoly = _offsetPoly(poly[::-1],offset,precision)

    # Decompose possibly self-intersecting <offset> polygon into multiple simple polygons (loops) 
    loops = _decompose(offPoly,offset)
    result = []
    for loop in loops:
        if len(loop) >= len(offPoly):
            continue
        p = _findPointInPoly(loop)
        wn = _windingNumber(p,offPoly)

        if False or (offset < 0 and wn < 0) or (offset > 0 and wn in [-1,0]):
                result.append([Vector((v[0],v[1])) for v in loop])
    return result

def offsetLine(line, offset, precision=8):
    # line:      A Python list of 2D vertices of type matutils.Vector, tuple or the internal
    #            class Vertex.
    # offset:    The distance to offset. Only positive values are allowed.
    # precision: If integer: Number of vertices per quadrant for arcs, default precision=8 .
    #            If float: Maximum error between chord and circle of arc.
    # return:    A Python list of the offset polygon(s). The type of the vertices of type matutils.Vector.
    #            The outer border is ordered counter-clockwise and holes are ordered clockwise.

    if offset == 0: return line
    offset = abs(offset)

    if isinstance(line[0],Vector):
        poly = [Vertex(v[0],v[1]) for v in line]
    elif isinstance(line[0],tuple):
        poly = [Vertex(v[0],v[1]) for v in line]
    else:
        poly = line

    poly += poly[::-1][1:-1]

    offPoly = _offsetPoly(poly,offset,precision)
    loops = _decompose(offPoly,offset)
    result = []
    for loop in loops:
        if len(loop) >= len(offPoly):
            continue
        p = _findPointInPoly(loop)
        wn = _windingNumber(p,offPoly)

        if False or (offset < 0 and wn < 0) or (offset > 0 and wn in [-1,0]):
            result.append([Vector((v[0],v[1])) for v in loop])
    return result
