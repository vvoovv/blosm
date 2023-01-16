from mathutils import Vector
import numpy as np
from random import shuffle
from itertools import *

MULT_EPSILON = 1 + 1e-14

def cyclePairs(lst):
    this, nexts = tee(lst)
    nexts = islice(cycle(nexts), len(lst) + 1, None)
    return zip(this,nexts)

# Create smallest enclosing circle of a list of points
# The points are expected as mathutils.Vector
# Returns tuple (center,radius)
def circumCircle(points):

    # c is tuple (center,radius)
    def isInCircle(c, p):
        return c is not None and (p-c[0]).length <= c[1] * MULT_EPSILON

    def makeDiameter(a, b):
        c = (a+b)/2
        r0 = (c-a).length
        r1 = (c-b).length
        return (c, max(r0, r1))

    # one boundary point known
    def addFromOnePoint(points, p):
        c = (p, 0.0)
        for i, q in enumerate(points):
            if not isInCircle(c, q):
                if c[1] == 0.0:
                    c = makeDiameter(p, q)
                else:
                    c = addFromTwoPoints(points[:i+1], p, q)
        return c

    # two boundary points known
    def addFromTwoPoints(points, p, q):
        circ = makeDiameter(p, q)
        left  = None
        right = None
        
        # for each point not in the two-point circle
        for r in points:
            if isInCircle(circ, r):
                continue
            
            # form a circumcircle and classify it on left or right side
            cross = (q-p).cross(r-p)
            c = circumCircle(p, q, r)
            if c is None:
                continue
            elif cross > 0.0 and (left is None or (q-p).cross(c[0]-p) > (q-p).cross(left[0]-p)):
                left = c
            elif cross < 0.0 and (right is None or (q-p).cross(c[0]-p) < (q-p).cross(right[0]-p)):
                right = c
        
        # Select which circle to return
        if left is None and right is None:
            return circ
        elif left is None:
            return right
        elif right is None:
            return left
        else:
            return left if (left[1] <= right[1]) else right

    def circumCircle(a, b, c):
        # Mathematical algorithm from Wikipedia: Circumscribed circle
        ax, bx, cx = a[0],b[0],c[0]
        ay, by, cy = a[1],b[1],c[1]
        ox = (min(ax, bx, cx) + max(ax, bx, cx)) / 2
        oy = (min(ay, by, cy) + max(ay, by, cy)) / 2
        ax -= ox;  ay -= oy
        bx -= ox;  by -= oy
        cx -= ox;  cy -= oy
        d = (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) * 2.0
        if d == 0.0:
            return None
        x = ox + ((ax*ax + ay*ay) * (by - cy) + (bx*bx + by*by) * (cy - ay) + (cx*cx + cy*cy) * (ay - by)) / d
        y = oy + ((ax*ax + ay*ay) * (cx - bx) + (bx*bx + by*by) * (ax - cx) + (cx*cx + cy*cy) * (bx - ax)) / d
        center = Vector((x,y))
        ra, rb, rc = (center-a).length, (center-b).length, (center-c).length
        return (center, max(ra, rb, rc))

    # randomize order
    shuffled = points
    shuffle(shuffled)
	
    # progressively add points to circle or recompute circle
    c = None
    for (i, p) in enumerate(shuffled):
        if c is None or not isInCircle(c, p):
            c = addFromOnePoint(shuffled[ : i + 1], p)
    return c

class CyrusBeckClipper():
    # The polygon in expected as a Python list of vertices of type mathutis.Vector.
    # Works only for convex polygons.
    def __init__(self, polygon):
        self.poly = polygon
        # Be sure that it is ordered counter-clockwise.
        area = sum( (v2[0]-v1[0])*(v2[1]+v1[1]) for v1,v2 in cyclePairs(self.poly))
        if area > 0.:
            print('reverted')
            self.poly.reverse()

        self.eV = [v2-v1 for v1,v2 in cyclePairs(self.poly)] # edge vectors of polygon

    # In literature, the normal pointing to the outside of the polygon's edge is used for
    # a dot prodcut with other vectors. The normal of the egde vector E(ex,ey) is then
    # defined as N(ey,-ex). The (2D) dot product with a vector V(vx,vy) then becomes:
    #   N dot V = ey*vx - ex*vy
    # But this is identical to the (2D) cross product of the edge vector with V:
    #   E cross V = ey*vx - ex*vy
    # Like this, the computation of the normal N can be avoided.
    def clipSegment(self, p0, p1):
        import matplotlib.pyplot as plt
        assert p0 != p1, 'degenerated segment'
        D = p1 - p0
        tE, tL = 0., 1.
        for ei, pi in zip(self.eV,self.poly):
            p0i = p0 - pi
            denom = ei[1]*D[0] - ei[0]*D[1] # 2D cross product
            if denom:
                numer = ei[1]*p0i[0] - ei[0]*p0i[1] # 2D cross product
                t = numer/-denom
                tE, tL = (max(tE, t), tL) if denom < 0. else (tE, min(tL, t))
        if (tE,tL) == (0.,1.):  # inside poly
            return 1, p0, p1
        elif tE <= tL:    # clipped
            return 0, p0 + D*tE, p0 + D*tL
        else:   #outside
            return -1, None, None

# Define region codes for SCClipper
_INSIDE = 0  # 0000 
_LEFT = 1    # 0001 
_RIGHT = 2   # 0010 
_BOTTOM = 4  # 0100 
_TOP = 8     # 1000     

class SCClipper():
    # Cohen-Sutherland algorithm
    
    def __init__(self, xMin, xMax, yMin, yMax):
        # float32 is used by mathutils.Vector, here it must be the same
        # precision to allow comparisons
        self.xMin = np.float32(xMin)
        self.xMax = np.float32(xMax)
        self.yMin = np.float32(yMin)
        self.yMax = np.float32(yMax)
        self.bottom = [Vector((xMin,yMin))]
        self.right = [Vector((xMax,yMin))]
        self.top = [Vector((xMax,yMax))]
        self.left = [Vector((xMin,yMax)),Vector((xMin,yMin))]

    def positionCode(self, p): 
        code = _INSIDE 
        if   p[0] < self.xMin: code |= _LEFT 
        elif p[0] > self.xMax: code |= _RIGHT 
        if   p[1] < self.yMin: code |= _BOTTOM 
        elif p[1] > self.yMax: code |= _TOP 
        return code

    def checkForBorder(self,p):
        if   p[0]==self.xMin: self.left.append(p)
        elif p[0]==self.xMax: self.right.append(p)
        elif p[1]==self.yMin: self.bottom.append(p)
        elif p[1]==self.yMax: self.top.append(p)

    def clip(self,p1,p2):
        code1 = self.positionCode(p1)
        code2 = self.positionCode(p2)
        accepted = False

        while True:
            if code1 == 0 and code2 == 0:   # inside scene
                accepted = True
                break
            elif (code1 & code2) != 0:      # outside scene
                break
            else:                           # needs clipping
                p = Vector((1.,1.))
                dP = p2-p1
                codeOut = code1 if code1 else code2

                if codeOut & _TOP:       # above scene
                    p[0] = p1[0] + ( (dP[0] / dP[1]) * (self.yMax - p1[1]) )
                    p[1] = self.yMax 
                elif codeOut & _BOTTOM:  # below scene
                    p[0] = p1[0] + ( (dP[0] / dP[1]) * (self.yMin - p1[1]) )
                    p[1] = self.yMin 
                elif codeOut & _RIGHT:  # right of scene
                    p[0] = self.xMax 
                    p[1] = p1[1] + ( (dP[1] / dP[0]) * (self.xMax - p1[0]) )
                elif codeOut & _LEFT:  # left of scene
                    p[0] = self.xMin 
                    p[1] = p1[1] + ( (dP[1] / dP[0]) * (self.xMin - p1[0]) )

                if codeOut == code1: 
                    p1 = p 
                    code1 = self.positionCode(p1)    
                else: 
                    p2 = p 
                    code2 = self.positionCode(p2) 

        if accepted:
            self.checkForBorder(p1)
            self.checkForBorder(p2)

        return accepted, p1, p2

    def getPolygon(self):
        # in counter-clockwise order, including start vertice also at end
        return sorted(self.bottom,key=lambda p:p[0])  + sorted(self.right,key=lambda p:p[1]) + \
               sorted(self.top,key=lambda p:-p[0]) + sorted(self.left,key=lambda p:-p[1]) 

# simple check for self-intersections   ------------------------------------
def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

# iterate cyclic over non-contiguous segments
def iterCircular(iterable):
    A,B,C,D = tee(iterable, 4)
    B = islice(cycle(B), 1, None)
    C = islice(cycle(C), 2, None)
    D = islice(cycle(D), 3, None)
    return zip(A, B, C, D)

def repairSimpleSelfIntersection(poly):
    if len(poly) <= 3:                                  # Repairs this type of intersection
        return False, poly                              # 
    polyIndx = range(len(poly))                         #        \  /
    wasRepaired = False                                 #         \/
    for a,b,c,d in iterCircular(polyIndx):              #         /\
        if intersect(poly[a],poly[b],poly[c],poly[d]):  #        /  \
            poly[b],poly[c] = poly[c],poly[b]           #       O----O
            wasRepaired = True          
    return wasRepaired,poly

import sys
def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush() 
