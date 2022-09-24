# ----------------------------------------------------------------
# PolyLine holds an ordered list of vertices that describe a line.
# The vertices are of the class mathutils.Vector.
#
# The line parameter <t> describes a position onto the line. It is
# constructed ba a compound of the last vertex number and the percentage
# of the following segment. <t> measures the distance from the start
# vertex, given by the <view> of the polyline. The vertex at the position
# can be found by the method t2v().
#
#          t=1.4 (fwd)                 t= 2.8 (rev)
#        ------------>         <--------------------------
#       |             |       |                           |
#       |             V       V                           |
#       o---------o---x-----o-x-------o---------o---------o
#       0         1         2         3         4         5
#
# The line parameter <d> describes a position onto the line. It measures
# the cumulated length of the polyline from the start vertex, given by
# the <view> of the polyline. to the position. The vertex at the position
# can be found by the method d2v().
#
#            d (fwd)                      d (rev)
#        ---------------->         <----------------------
#       |                 |       |                       |
#       |                 V       V                           |
#       o-------------o---x-----o-x-------o-----o---------o
#       0             1         2         3     4         5
#
# ----------------------------------------------------------------

from mathutils import Vector
from itertools import tee, accumulate, product
import numpy as np
from bisect import bisect_left
from math import floor,ceil
import matplotlib.pyplot as plt

from lib.pygeos.shared import Coordinate
from lib.CompGeom.OffsetGenerator import OffsetGenerator

# helper functions -----------------------------------------------
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

def unitDoubleParabola(nrPoints):
    x = np.linspace(0.,1.,nrPoints)
    return np.where(x <= 0.5, 2.*x*x, 1.-2.*(1.-x)*(1.-x))

def unitHalfParabola(nrPoints):
    x = np.linspace(0.,1.,nrPoints)
    return np.where(x <= 0.5, 1.-4*(0.5-x)*(0.5-x), 1.)

# ----------------------------------------------------------------

class PolyLine():
    fwd = slice(None,None,1)
    rev = slice(None,None,-1)
    def __init__(self, vertList):
        # Initilaizes an instance from a list of vertices of the
        # classes mathutils.Vector, pyGEOS Coordinate or of
        # tuples of two floats for x and y values.
        self.verts = None
        if isinstance(vertList[0],Vector):
            self.verts = vertList
        elif isinstance(vertList[0],Coordinate):
            self.verts = [Vector((v.x,v.x)) for v in vertList]
        elif isinstance(vertList[0],tuple):
            self.verts = [Vector(v) for v in vertList]
        else:
            raise TypeError('PolyLine: Unexpected input type.')

        self.view = PolyLine.fwd
        self.prepareLineParameters()

    def setView(self,view):
        # <view> can either be <PolyLine.fwd> or <PolyLine.rev>
        # All operations of the polyline use the vertex order forward or
        # reverse, according to this setting.
        self.view = view
        self.tInterp = None
        self.dInterp = None
        self.prepareLineParameters()

    def prepareLineParameters(self):
        vD = list( accumulate([0]+[(v2-v1).length for v1,v2 in pairs(self.verts[self.view])]) )
        vI = [i for i in range(len(self.verts))]
        x = [v[0] for v in self.verts[self.view]]
        y = [v[1] for v in self.verts[self.view]]
        xy = [complex(xx,yy) for xx,yy in zip(x,y)]
        self.tInterp = LinearInterpolator(vI,xy)
        self.dInterp = LinearInterpolator(vD,xy)
        self.dtInterp = LinearInterpolator(vD,vI)

    def t2v(self,t):
        # Computes the vertex given by the line parameter <t>.
        xy = self.tInterp(t)
        return Vector((xy.real,xy.imag))

    def d2v(self,d):
        # Computes the vertex given by the line parameter <t>.
        xy = self.dInterp(d)
        return Vector((xy.real,xy.imag))

    def d2t(self,d):
        return self.dtInterp(d)

    def __len__(self):
        # Return the length of the polyline
        return len(self.verts)

    def clone(self):
        # Create a copy of the polyLine
        return PolyLine([v for v in self.verts])

    def __getitem__(self, indx):
        # Acces a vertex by index
        return self.verts[self.view][indx]

    def __iter__(self):
        # Iterator of the vertices list
        return iter(self.verts[self.view])

    def length(self):
        # Returns the total geomteric length of the polyline.
        return sum( (v2-v1).length for v1, v2 in pairs(self.verts) )

    def segments(self):
        # Returns a list of vertex tuples for all segments of the polyline
        return [(v1,v2) for v1, v2 in pairs(self.verts[self.view])]

    def segment(self,indx):
        # Returns the indx'th segment of the polyline
        assert indx < len(self.verts)-2, 'indx out of limits (%d ... %d)'%(0,len(self.verts)-2)
        return (self.verts[self.view][indx],self.verts[self.view][indx+1])

    def lengthOfSegment(self,i):
        # Returns the length of the ith segment of the line
        assert i<len(self.verts)+1
        return (self.verts[i+1]-self.verts[i]).length

    def unitVectors(self):
        # Returns a list of the unit vectors of the polyline's segments
        return [(v2-v1)/(v2-v1).length for v1, v2 in pairs(self.verts[self.view])]

    def unitEndVec(self,fwd):
        s = slice(None,None,1) if fwd else slice(None,None,-1)
        d = self.verts[s][1]-self.verts[s][0]
        return d/d.length

    def trimmed(self,tS,tE):
        # Returns the slice of the line between and including the
        # vertices given by the line parameters <tS> and <tE>
        assert tS < tE
        itS = floor(tS)
        itE = floor(tE)

        if itS == tS:
            iS = itS
            pS = []
        else:
            iS = itS+1
            pS = [self.t2v(tS)]

        if itE == tE:
            iE = itE+1
            pE = []
        else:
            iE = itE+1
            pE = [self.t2v(tE)]

        if iS == iE:
            return PolyLine(pS + pE)
        else:
            return PolyLine(pS + self.verts[self.view][iS:iE] + pE)

        # N = len(self.verts)
        # if tS==0.:
        #     pS = []
        #     iS = -1
        # else:
        #     if floor(tS) != tS:
        #         iS = floor(tS)
        #         v0,v1 = self.verts[self.view][iS], self.verts[self.view][iS+1]
        #         pS = [v0 + (v1 - v0) * (tS % 1)]
        #     else:
        #         pS = []
        #         iS = floor(tS)-1

        # if tE==0.:
        #     pE = []
        #     iE = N-1
        # else:
        #     t = N-1-tE
        #     if floor(t) != t:
        #         iE = floor(t)
        #         v0,v1 = self.verts[self.view][iE], self.verts[self.view][iE+1]
        #         pE = [v0 + (v1 - v0) * (t % 1)]
        #     else:
        #         pE = []
        #         iE = floor(t)+1

        # inter = [v for v in self.verts[self.view][iS+1:iE+1]]
        # return PolyLine(pS+inter+pE)

    def orthoProj(self,p):
        # Projects the point <p> onto the polyline. The return values
        # are the projected point and the line parameter t. The line
        # is evaluated from start to end. A projection before the first
        # vertex on the infinite line of the first edge, or after the 
        # last vertex on the infinite line of the last edge are accepted.
        edgeNr = 0
        passedEnd = True
        passedStart = False
        for p1,p2 in pairs(self.verts[self.view]):
            edgeVector = p2-p1
            t = (p-p1).dot(edgeVector)/edgeVector.dot(edgeVector)
            r = self.t2v(t+edgeNr)
            if  0. <= t < 1.:
                passedEnd = False
                break
            if t < 0.:
                break
            edgeNr += 1
            passedStart = True

        if t<0:
            if passedStart:
                # the projection is between two edges, use nearest vertex
                t = edgeNr
        else:
            t += edgeNr-1 if passedEnd else edgeNr
        return self.t2v(t), t


    def offsetPointAt(self,t,dist):
        # Computes the position of a point <p> on the line using
        # the line parameter <t>. This point is offset perpendicular
        # to the segment it lies on by the distance <dist>. The result
        # is on the left of the line when <dist> is positive and on
        # their right else.
        t0 = floor(t)
        if t0 >= len(self.verts)-1:
            v0,v1 = self.verts[self.view][-2:]
        elif t0 < 0.:
            v0,v1 = self.verts[self.view][:2]
        else:
            v0,v1 = self.verts[self.view][t0:t0+2]
        p = self.t2v(t)
        d = v1-v0
        u = d/d.length
        pOffs = p + Vector((-u[1],u[0])) * dist
        return pOffs

    @staticmethod
    def intersectInfinite(p1,p2,p3,p4,limTan=0.0175):
        # Finds the intersection of two infinite lines, the first one given by
        # the vertices <p1> and <p2> and the second one by the vertices <p3>
        # and <p4>.
        # Returns the intersection point <p> and the polyline parameters <t1>
        # and <t2> relative to the given segments. When the lines are almost
        # parallel, this is when the tangent of the angle between them is less
        # than <limTan> (0.0175 corresponds to ~1°), None is returned.
        d1, d2 = p2-p1, p4-p3
        cross = d1.cross(d2)
        dot = d1.dot(d2)
        if abs(cross/dot) < limTan: # tangent of ~1°, almost parallel
            return None
        d3 = p1-p3
        t1 = (d2[0]*d3[1] - d2[1]*d3[0])/cross
        t2 = (d1[0]*d3[1] - d1[1]*d3[0])/cross
        p = p1 + d1*t1
        return p, t1, t2

    @staticmethod
    def intersection(poly1,poly2):
        # Finds the intersection between two polylines <poly1> and <poly2>.
        # Returns the intersection point <iP>, the line parameters <t1> and
        # <t2> of the intersection on the lines and the unit vectors <uV1>
        # and <uV2> of the intersecting segments.
        # When there is no intersection between the polyline segments, the
        # intersection may be on the infinite lines through the first segments.
        # In this case, <t1> and/or <t2> may be negative. Whene these infinte
        # lines are almost paralllel (angle between them less than 1°), or when
        # they intersect after the last segments of the polylines, None is returned.
        segs1, segs2 = poly1.segments(), poly2.segments()
        found = False

        # Intersections for the first few segments are most probable.
        # The combinations of the indices are constructed so that these
        # are first tested.
        combinations = list(product(range(len(segs1)),range(len(segs2))))
        combinations = sorted(combinations,key=lambda x: sum(x))
        for i1,i2 in combinations:
            p1, p2 = segs1[i1]
            p3, p4 = segs2[i2]
            isect = PolyLine.intersectInfinite(p1,p2,p3,p4)
            if isect is None: continue

            # Accept intersection of infinite lines (negative line parameters)
            # only between first segments.
            iP, t1, t2 = isect
            if (i1,i2) == (0,0): 
                if t1 > 1. or t2 > 1.:
                    continue # out of segment
            else:
                if t1 < 0. or t1 > 1. or t2 < 0. or t2 > 1.:
                    continue # out of segment
            found = True
            break # valid intersection between polylines  found

        if found:
            d1, d2 = p2-p1, p4-p3
            return iP, i1+t1, i2+t2, d1/d1.length, d2/d2.length
        else:
            return None

    def intersectSegment(self, p1, p2):
        for v1,v2 in pairs(self.verts[self.view]):
            d1, d2 = v2-v1, p2-p1
            cross = d1.cross(d2)
            if cross == 0.:
                continue
            d3 = v1-p1
            t1 = (d2[0]*d3[1] - d2[1]*d3[0])/cross
            t2 = (d1[0]*d3[1] - d1[1]*d3[0])/cross
            if 0. <= t1 <= 1. and 0. <= t2 <= 1:
                return v1 + d1*t1, t1
        return None

    def parallelOffset(self, dist):
        # Returns the PolyLine that is offset perpendicularly by
        # the distance <dist>. The result is on the left of
        # the line when <dist> is positive and on their right else.
        offsetter = OffsetGenerator()
        offsetVerts = offsetter.parallel_offset(self.verts[self.view],dist)
        return PolyLine(offsetVerts)

    def buffer(self,leftW,rightW):
        # Expands the line to a polygon, to the left by the distance
        # <leftW> and to the right by <rightW>.
        offsetter = OffsetGenerator()
        offsetL = offsetter.parallel_offset(self.verts[self.view],leftW)
        offsetR = offsetter.parallel_offset(self.verts[self.view],-rightW)
        offsetR.reverse()
        offsetVerts = offsetL + offsetR
        offsetPoly = PolyLine(offsetVerts)
        return offsetPoly

    def parabolicOffset(self,dist,delta,nrPoints = 30):
        # Prepare transform of segment lengths along polyline to line parameters <t>
        segLengths = np.array([0]+[(v1-v2).length for v1,v2 in pairs(self.verts[self.view])])
        sumLengths = np.cumsum(segLengths)
        t = np.array([i for i in range(len(self.verts))])

        # Create <samples> and transform to line parameter samples <tSamples>
        samples = np.linspace(sumLengths[0],sumLengths[-1],nrPoints)
        tSamples = np.interp(samples,sumLengths,t)

        # Create offset samples offSamples, shifted by dist and added double parabola
        # with width of delta.
        offVerts = []
        para = unitHalfParabola(nrPoints)#unitDoubleParabola(nrPoints)
        # plt.close()
        # plt.plot(para)
        # plt.show()
        for i in range(nrPoints):
            o = self.offsetPointAt(tSamples[i],dist+para[i]*delta)
            offVerts.append(o)
        return offVerts

    def parabolicBuffer(self,leftW,rightW,leftD,rightD ):
        p = self.verts[0]
        # plt.plot(p[0],p[1],'ro',markersize=10,zorder=950)
        offsetter = OffsetGenerator()
        if leftD != 0.:
            offsetL = self.parabolicOffset(leftW-leftD,leftD)
        else:
            offsetL = offsetter.parallel_offset(self.verts[self.view],leftW)
        if rightD != 0.:
            offsetR = self.parabolicOffset(-(rightW-rightD),-rightD)
        else:
            offsetR = offsetter.parallel_offset(self.verts[self.view],-rightW)
        offsetR.reverse()
        offsetVerts = offsetL + offsetR
        offsetPoly = PolyLine(offsetVerts)
        return offsetPoly

    def plot(self,color):
        for v1,v2 in pairs(self.verts[self.view]):
            plt.plot([v1.x,v2.x],[v1.y,v2.y],color)
            plt.plot(v1.x,v1.y,'k.')
            plt.plot(v2.x,v2.y,'k.')

