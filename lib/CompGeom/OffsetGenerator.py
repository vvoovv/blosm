from itertools import tee
from math import isclose, pi, sin, cos, atan2
from mathutils import Vector

import matplotlib.pyplot as plt

# Simplified from
# Xu-Zheng Liu, Jun-Hai Yong, Guo-Qin Zheng, Jia-Guang Sun.
# An offset algorithm for polyline curves.
# Computers in Industry, Elsevier, 2007, 15p. ￿inria-00518005
# https://hal.inria.fr/inria-00518005/document
#
# Additional assumptions:
# - Way-sections never have self-intersections.
# - The left border of a way-section never intersects the right border.
# - The way-segments are joined round, by a fillet

# helper functions -----------------------------------------------
def segmentIntersection(p1,p2,p3,p4):
    d1, d2 = p2-p1, p4-p3 
    denom = d2.y*d1.x - d2.x*d1.y
    if denom == 0.: # parallel
        return None
    d3 = p1 - p3
    t1 = (d2.x*d3.y - d2.y*d3.x)/denom 
    t2 = (d1.x*d3.y - d1.y*d3.x)/denom              
    return p1 + d1*t1, t1, t2

def pairs(iterable):
	# s -> (s0,s1), (s1,s2), (s2, s3), ...
    p1, p2 = tee(iterable)
    next(p2, None)
    return zip(p1,p2)
# ----------------------------------------------------------------

class OffsetGenerator():
    def __init__(self,resolution=8):
        self.filletAngleQuantum = pi / 2.0 / resolution
        self.distance = 0.
        self.pline0 = []
        self.offset = []
        self.connected = []

    def reset(self):
        self.pline0 = []
        self.offset = []
        self.connected = []

    def offsetSegments(self):
        for v0,v1 in self.pline0:
            d = v1-v0
            u = d/d.length * self.distance
            uv = Vector((-u[1],u[0]))
            self.offset.append( (v0+uv,v1+uv) )

    def connectOffsetSegments(self):
        if not self.offset: self.connected = []
        if len(self.offset) == 1: self.connected =  [*self.offset[0]]

        self.connected = [self.offset[0][0]]
        for seg,offs in zip(self.pline0,pairs(self.offset)):
            # end of segment seg will be origin for fillet, if any
            (p1,p2),(p3,p4) = offs
            p12 = p2-p1
            p34 = p4-p3
            if isclose(p12.cross(p34), 0.): 
                # case 1: offset segments are collinear and consecutive
                self.connected.append(p2)
            else:
                # case 2
                p, t12, t34 = segmentIntersection(p1,p2,p3,p4)
                # find type of intersection point p
		        # TIP means 'true intersection point' : ie. the intersection point lies within the segment.
                # FIP means 'false intersection point' : ie the intersection point lies outside the segment.
                # PFIP means 'positive false intersection point' : ie the intersection point lies beyond the
                # segment in the direction of the segment
                TIP12 = 0 <= t12 <= 1
                TIP34 = 0 <= t34 <= 1
                FIP12 = not TIP12
                FIP34 = not TIP34
                PFIP12 = FIP12 and t12 > 0
			
                if TIP12 and TIP34:
                    # case 2a
                    self.connected.append(p)
                elif FIP12 and FIP34:
                    # case 2b
                    if PFIP12:
                        # TODO: test for mitre limit
                        # self.connected.append(p)
                        fillet = self.constructFillet(p2,seg[1],p3,abs(self.distance))
                        if fillet:
                            self.connected.extend( fillet )
                        else:
                            self.connected.append(p)
                    else:
                        self.connected.append(p2)
                        self.connected.append(p3)
                else:
                    # case 2c
                    self.connected.append(p2)
                    self.connected.append(p3)   
        # case 3
        self.connected.append(self.offset[-1][-1])

    def constructFillet(self,p0,o,p1,radius):
        # p1: start point
        # o:  origin
        # p2: end point
        d1, d2 = o-p0, p1-o 
        denom = d2.y*d1.x - d2.x*d1.y
        # 1 : clockwise, 0: linear, -1 : counter-clockwise
        orientation = 1 if denom > 0. else -1 if denom < 0. else 0.

        d0 = p0 - o
        startAngle = atan2(d0[1], d0[0])
        d1 = p1 - o
        endAngle = atan2(d1[1], d1[0])

        if orientation > 0.: # counter-clockwise
            if startAngle <= endAngle:
                startAngle += 2.0 * pi
        elif startAngle >= endAngle:
            # counter-clockwise
            startAngle -= 2.0 * pi

        totalAngle = abs(startAngle - endAngle)
        nSegs = int(totalAngle / self.filletAngleQuantum + 0.5)
        if nSegs < 1:
            return []
        angleInc = totalAngle / nSegs

        filletList = []
        currAngle = 0.0
        while currAngle < totalAngle:
            angle = startAngle - orientation * currAngle
            x = o[0] + (radius * cos(angle))
            y = o[1] + (radius * sin(angle))
            filletList.append( Vector((x,y)) )
            currAngle += angleInc

        return filletList

    def cleanSelfIntersections(self):
        # At least three consecutive segments (4 vertices) are required for self-intersection
        if len(self.connected) < 4:
            return self.connected

        EPSILON = 0.001
        cleaned = []
        lastValid = -1
        self.connected = [self.pline0[0][0]] + self.connected + [self.pline0[-1][1]]
        for i, (p1, p2) in enumerate(pairs(self.connected)):
            if i > lastValid:# and i<len(self.connected)-2:
                cleaned.append(p1)
                p0 = NonelastValid0 = None
                for j, (p3, p4) in enumerate(pairs(self.connected[i + 2:])):
                    isectRes = segmentIntersection(p1,p2,p3,p4)
                    if isectRes is not None:
                        p, t1, t2 = isectRes
                        if EPSILON <= t1 <= 1.-EPSILON and EPSILON <= t2 <= 1.+EPSILON:
                            p0 = p
                            lastValid0 = j+ i + 2
                if p0:
                    cleaned.append(p0)
                    lastValid = lastValid0
                            # cleaned.append(p)
                            # lastValid = j+ i + 2
                            # break
        cleaned.append(self.connected[-1])
        return cleaned[1:-1]
         
    def parallel_offset(self, polyline, distance):
        self.reset()
        self.distance = distance
        self.pline0 = [(v1,v2) for v1,v2 in zip(polyline,polyline[1:])]
        self.offsetSegments()
        self.connectOffsetSegments()
        cleaned = self.cleanSelfIntersections()
        # if len(cleaned) > 0:
        #     plt.close()
        #     for i,(v1,v2) in enumerate(self.pline0):
        #         plt.plot([v1[0],v2[0]],[v1[1],v2[1]],'k')
        #         plt.plot(v1[0],v1[1],'k.')
        #         plt.plot(v2[0],v2[1],'k.')
        #         plt.text(v1[0],v1[1],str(i))

        #     # for i,(v1,v2) in enumerate(self.offset):
        #     #     plt.plot([v1[0],v2[0]],[v1[1],v2[1]],'b')
        #     #     plt.text(v1[0],v1[1],str(i))

        #     for i,(v1,v2) in enumerate( zip(self.connected,self.connected[1:]) ):
        #         plt.plot([v1[0],v2[0]],[v1[1],v2[1]],'k')
        #         plt.plot(v1[0],v1[1],'k.')
        #         plt.plot(v2[0],v2[1],'k.')
        #         plt.text(v1[0],v1[1],str(i))

        #     for v1,v2 in zip(cleaned,cleaned[1:]):
        #         plt.plot([v1[0],v2[0]],[v1[1],v2[1]],'g',linewidth = 5)
        #         plt.plot(v1[0],v1[1],'kx',markersize=8)
        #         plt.plot(v2[0],v2[1],'kx',markersize=8)

        #     # for v in cleaned:
        #     #     print(v)
        #     plt.title('Self-Intersection '+str(len(cleaned)))
        #     plt.gca().axis('equal')
        #     plt.show()
        #     test=1
        return cleaned


