from itertools import tee, islice, cycle
from collections import defaultdict
from mathutils import Vector

def pairs(iterable):
    # s -> (s0,s1), (s1,s2), (s2, s3), ...
    p1, p2 = tee(iterable)
    next(p2, None)
    return zip(p1,p2)

def cyclePairs(iterable):
    this, nexts = tee(iterable)
    nexts = islice(cycle(nexts), len(iterable) + 1, None)
    return zip(this,nexts)

def pointInPolygon(poly, p):
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
                    if det == 0: return 'ON' # on edge
                    # if right crossing
                    if (det > 0. and p2[1] > p1[1]) or (det < 0 and p2[1] < p1[1]):
                        # modify w
                        wNr += 1 if p2[1] > p1[1] else -1
            else:
                if p2[0] > 0.:
                    det = p1[0] * p2[1] - p2[0] * p1[1]
                    if det == 0: return 'ON' # on edge
                    # if right crossing
                    if (det > 0 and p2[1] > p1[1]) or (det < 0 and p2[1] < p1[1]):
                        # modify w
                        wNr += 1 if p2[1] > p1[1] else -1
    if (wNr % 2) != 0:
        return 'IN'    # in polygon
    else:
        return 'OUT'   # out of polygon

def isectSegSeg(p1,p2,p3,p4):
    # Get the intersection of segment <p1,p2> with <p3,p4>.
    # None if no intersection or end-point intersection.
    d1, d2 = (p2[0]-p1[0],p2[1]-p1[1]), (p4[0]-p3[0],p4[1]-p3[1])#p2-p1, p4-p3
    cross = d1[0] * d2[1] - d2[0] * d1[1] # d1.cross(d2)
    if cross == 0.:
        return None
    d3 = (p1[0]-p3[0],p1[1]-p3[1]) # p1-p3
    t1 = (d2[0]*d3[1] - d2[1]*d3[0])/cross
    t2 = (d1[0]*d3[1] - d1[1]*d3[0])/cross
    if 0. < t1 < 1. and 0. < t2 < 1:
        return ((p1[0] + d1[0]*t1),(p1[1] + d1[1]*t1)) #(p1 + d1*t1).freeze()
    else:
        return None

def inorderExtend(seq, v1, v2, points):
    # Extend a sequence <seq> by <points> that are
    # between the points v1, v2
    k, r = None, False
    if   v1[0] < v2[0]: k, r = lambda i: i[0], True
    elif v1[0] > v2[0]: k, r = lambda i: i[0], False
    elif v1[1] < v2[1]: k, r = lambda i: i[1], True
    else:               k, r = lambda i: i[1], False
    l = [ (p,'ON') for p in sorted(points, key=k, reverse=r) ]
    i = next((i for i, p in enumerate(seq) if p[0] == v2), -1)
    assert(i>=0)
    for e in l:
        seq.insert(i, e)
    return seq

def plotPolygon(poly,vertsOrder,lineColor='k',fillColor='k',width=1.,fill=False,alpha = 0.2,order=100):
    import matplotlib.pyplot as plt
    x = [n[0] for n in poly] + [poly[0][0]]
    y = [n[1] for n in poly] + [poly[0][1]]
    if fill:
        plt.fill(x[:-1],y[:-1],color=fillColor,alpha=alpha,zorder = order)
    plt.plot(x,y,lineColor,linewidth=width,zorder=order)
    if vertsOrder:
        for i,(xx,yy) in enumerate(zip(x[:-1],y[:-1])):
            plt.text(xx,yy,str(i),fontsize=12)

def plotLine(line,vertsOrder,lineColor='k',width=1.,order=100):
    import matplotlib.pyplot as plt
    x = [n[0] for n in line]
    y = [n[1] for n in line]
    plt.plot(x,y,lineColor,linewidth=width,zorder=order)
    if vertsOrder:
        for i,(xx,yy) in enumerate(zip(x[:-1],y[:-1])):
            plt.text(xx,yy,str(i),fontsize=12)

def plotEnd():
    import matplotlib.pyplot as plt
    plt.gca().axis('equal')
    plt.show()

class LinePolygonClipper():
    def __init__(self, poly):
        self.poly = [tuple(v) for v in poly]

    def clipLine(self, lineV):
        line = [tuple(v) for v in lineV]

        # Classify and insert the line vertices
        lineClass = [(p, pointInPolygon(self.poly,p)) for p in line]
        # for v in lineClass:
        #     print(v)
        # print(' ')

        # Find intersection points
        isectsL = defaultdict(list)
        for p1, p2 in cyclePairs(self.poly):
            for v1, v2 in pairs(line):
                isect = isectSegSeg(p1,p2,v1,v2)
                if isect:
                    isectsL[(v1,v2)].append(isect)

        # Insert the intersection points.
        # A vertex can appear twice in sequence in a vertex ring,
        # but not more (restricion implemented in _inorderExtend)
        for segment, isects in isectsL.items():
            inorderExtend(lineClass, segment[0], segment[1], isects)
        # for v in lineClass:
        #     print(v)

        # Now collect the line fragemnts that are in the polygon
        fragments = []
        totalLength = 0.
        for v1,v2 in pairs(lineClass):
            if v1[1] in ['ON','IN'] and v2[1] in ['ON','IN']:
                p1, p2 = Vector(v1[0]), Vector(v2[0])
                fragments.append( (p1,p2) )
                totalLength += (p2-p1).length

        nrOfON = sum(1 for v in lineClass if v[1] == 'ON' )
        return fragments, totalLength, nrOfON