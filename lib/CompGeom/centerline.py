from itertools import tee, accumulate
from bisect import bisect_left
from mathutils import Vector
from lib.CompGeom.delaunay_voronoi import computeDelaunayTriangulation

def pairs(iterable):
    # s -> (s0,s1), (s1,s2), (s2, s3), ...
    p1, p2 = tee(iterable)
    next(p2, None)
    return zip(p1,p2)

class LinearInterpolator():
    def __init__(self, x, y):
        # <x> and <Y> must be of equal length (at least two elements) and x must
        # attributes
        self.x = x
        self.y = y
        self.length = len(x)

        # precalculate slopes
        intervals = zip(x, x[1:], y, y[1:])
        self.slopes = [(y2 - y1) / (x2 - x1) for x1, x2, y1, y2 in intervals]

    def __call__(self, x):
        i = bisect_left(self.x, x) - 1
        if i == -1:
            i = 0
        elif i == self.length - 1:
            i = -1
        return self.y[i] + self.slopes[i] * (x - self.x[i])

# Algorithm based on the idea in:
#         Elzbieta Lewandowicz and Paweł Flisek
#         A Method for Generating the Centerline of an Elongated Polygon
#         on the Example of a Watercourse
#         May 2020, International Journal of Geo-Information 9(5):304

def _pointInPolygon(poly, p):
    # Location of a point vis a vis a polygon.
    #
    # poly:     Python list of vertices of type mathutils.Vector, arbitrary order
    # p:        Vertex of type mathutils.Vector
    # Return:   'IN' if <p> is in, 'ON' if <p> is on, or 'OUT' if <p> is out
    #           of the polygon <poly>.
    # 
    # Based on algorithm 7 from:
    #     Kai Horman and Alexander Agathos,
    #     "The point in polygon problem for arbitrary polygons".
    #     Computational Geometry: Theory and Applications, 
    #     Volume 20 Issue 3, November 2001
    # See: https://www.sciencedirect.com/science/article/pii/S0925772101000128

    # shift polygon so that p is (0,0)
    pi = [(v[0]-p[0],v[1]-p[1]) for v in poly]

    if pi[0] == (0,0):#Vector((0,0)):
        return 'ON' # on vertex

    wNr = 0
    for p1,p2 in zip(pi,pi[1:]+[pi[0]]):
        if p2[1] == 0.:
            if p2[0] == 0.:
                return 'ON' # on vertex
            else:
                if p1[1] == 0. and (p2[0] > 0.) == (p1[0] < 0.):
                    return 'ON' # on edge
        # if crossing horizontal line
        if (p1[1] < 0. and p2[1] >= 0.) or (p1[1] >= 0. and p2[1] < 0.):
            if p1[0] >= 0.:
                if p2[0] > 0.:
                    # modify w
                    wNr += 1 if p2[1] > p1[1] else -1
                else:
                    det = p1[0] * p2[1] - p2[0] * p1[1]
                    if abs(det) < 5.e-3: return 'ON' # on edge
                    # if right crossing
                    if (det > 0. and p2[1] > p1[1]) or (det < 0 and p2[1] < p1[1]):
                        # modify w
                        wNr += 1 if p2[1] > p1[1] else -1
            else:
                if p2[0] > 0.:
                    det = p1[0] * p2[1] - p2[0] * p1[1]
                    if abs(det) < 5.e-3: return 'ON' # on edge
                    # if right crossing
                    if (det > 0 and p2[1] > p1[1]) or (det < 0 and p2[1] < p1[1]):
                        # modify w
                        wNr += 1 if p2[1] > p1[1] else -1
    if (wNr % 2) != 0:
        return 'IN'    # in polygon
    else:
        return 'OUT'   # out of polygon

def centerlineOf(curve0, curve1):
    # Computes the centerline of two almost parallel curves.
    # curve0,curve1: Python lists of vertices (type mathutils.Vector).
    #                Both should be alomst parallel and ordered in the
    #                same direction
    # return:        Python list of centerline vertices (type mathutils.Vector).
    points = curve0 + curve1
    poly = curve0 + curve1[::-1]
    triangles = computeDelaunayTriangulation(points)

    edges = set()   # set() to remove duplicates
    for tri in triangles:
        for e1, e2 in [[0,1],[1,2],[2,0]]: # for all edges of triangle
            t1,t2 = tri[e1], tri[e2]
            i1,i2 = (t1,t2) if t1<t2 else (t2,t1) # lower index first
            if i1 < len(curve0) <= i2: # Restrict to TIN edges
                edges.add((i1,i2))

    centerline = [(points[i1]+points[i2])/2. for i1,i2 in sorted(edges)]
    # Check if points are in polygon
    centerline = [p for p in centerline if _pointInPolygon(poly,p) in ('IN','ON')]
    return centerline

def interpolateCurve(curve,sampleDist=10.):
    x = [v[0] for v in curve]
    y = [v[1] for v in curve]
    xy = [complex(xx,yy) for xx,yy in zip(x,y)]
    vD = list( accumulate([0]+[(v2-v1).length for v1,v2 in pairs(curve)]) )
    interp = LinearInterpolator(vD,xy)
    samples = max(10,int(vD[-1]/sampleDist))
    step = vD[-1]/(samples-1)
    xy = [interp(d*step) for d in range(samples)]

    interpCurve = [Vector((v.real,v.imag)) for v in xy]
    return interpCurve

def centerlineInterOf(curve0, curve1,sampleDist=10.):
    # Computes the centerline of two almost parallel curves.
    # curve0,curve1: Python lists of vertices (type mathutils.Vector).
    #                Both should be alomst parallel and ordered in the
    #                same direction
    # return:        Python list of centerline vertices (type mathutils.Vector).
    curve0 = interpolateCurve(curve0,sampleDist)
    curve1 = interpolateCurve(curve1,sampleDist)
    points = curve0 + curve1
    poly = curve0 + curve1[::-1]
    triangles = computeDelaunayTriangulation(points)

    edges = set()   # set() to remove duplicates
    for tri in triangles:
        for e1, e2 in [[0,1],[1,2],[2,0]]: # for all edges of triangle
            t1,t2 = tri[e1], tri[e2]
            i1,i2 = (t1,t2) if t1<t2 else (t2,t1) # lower index first
            if i1 < len(curve0) <= i2: # Restrict to TIN edges
                edges.add((i1,i2))

    centerline = [(points[i1]+points[i2])/2. for i1,i2 in sorted(edges)]
    # Check if points are in polygon
    centerline = [p for p in centerline if _pointInPolygon(poly,p) in ('IN','ON')]
    return centerline