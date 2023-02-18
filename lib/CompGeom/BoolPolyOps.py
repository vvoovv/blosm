# Based on:
# Avraham Margalit and Gary Knott.
# An Algorithm for Computing the Union, Intersection or Difference of Two Polygons.
# Computers & Graphics VoI. 13, No 2, pp 167-183, 1989
#
# Computes the boolean operation of two simple polygons without holes.
#
# result = boolPolyOp(polyA, polyB, operation)
# polyA:        Python list of vertices of type mathutils.Vector, arbitrary order
# polyB:        Python list of vertices of type mathutils.Vector, arbitrary order
# operation:    'union', 'intersection', 'difference'
# result:       Python list of resulting polygon(s)
#
# Example usage:
#             polyA = [Vector((3,1)),Vector((6,1)),Vector((5,4))]
#             polyB = [Vector((1,4)),Vector((5,2)),Vector((3,5))]
#             result = boolPolyOp(polyA, polyB, 'intersection')
#             result
#             > [[Vector((4.33...053, 3.0)), Vector((4.0, 2.5)), Vector((5.0, 2.0))]]

from itertools import tee, islice, cycle
from collections import defaultdict
from mathutils import Vector

# helper functions ------------------------------------------------------------
def _iterEdges(poly):
        p1, p2= tee(poly)
        p2 = islice(cycle(p2), 1, None)
        return zip(p1,p2)

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

def _borderState(poly,v1,v2):
    # If <v1,v2> is an edge of <poly>, 
    # then they are on the border
    for p1,p2 in _iterEdges(poly):
        if v1 in (p1,p2) and v2 in (p1,p2):
            return 'ON'

    # else check state using a midpoint
    mid = ((v1[0] + v2[0])/2.,(v1[1] + v2[1])/2.)   # (v1 + v2) / 2.0
    return _pointInPolygon(poly,mid)

def _inorderExtend(seq, v1, v2, points):
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

def _orientation(poly):
    area = sum( (p2[0]-p1[0])*(p2[1]+p1[1]) for p1,p2 in zip(poly,poly[1:]+[poly[0]]))
    return area < 0.    # counter-clockwise

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
    if 0. < t1 < 1. and 0. < t2 < 1:
        return ((p1[0] + d1[0]*t1),(p1[1] + d1[1]*t1)) #(p1 + d1*t1).freeze()
    else:
        return None

def _extendFragments(fragments, ring, poly, fragmentType):
    for v1, v2 in _iterEdges(ring):
        # if one endpoint is of type fragmentType ...
        if v1[1] == fragmentType or v2[1] == fragmentType:
            # ... insert it
            fragments[v1[0]].append( v2[0] )           
        # ... else we have a boundary fragment
        elif v1[1] == 'ON' and v2[1] == 'ON':
            state = _borderState(poly,v1[0],v2[0])
            if state == fragmentType or state == 'ON':
                fragments[v1[0]].append( v2[0] )
#------------------------------------------------------------------------------

def boolPolyOp(polyAv, polyBv, operation):
    # converted to tuples because mathutils.Vector made issues
    polyA = [tuple(v) for v in polyAv]
    polyB = [tuple(v) for v in polyBv]

    # polyA, polyB: Python lists of vertices, no duplicates
    # operation: 'union', 'intersection', 'difference'
    assert operation in ('union','intersection','difference'), 'Unknown operation'

    # Step 1: Normalize the orientations of the input polygons
    # For union and intersection, the same orientation on both polygons is required.
    # For difference, different orientation is required.
    sameOrient = _orientation(polyA) == _orientation(polyB)
    if sameOrient != (operation != 'difference'):
        polyB.reverse()

    # Step 2: Classify and insert the vertices into vertex rings
    ringA = [(p, _pointInPolygon(polyB,p)) for p in polyA]
    ringB = [(p, _pointInPolygon(polyA,p)) for p in polyB]

    # Step 3a: Find the intersection points
    isectsA = defaultdict(list)
    isectsB = defaultdict(list)
    for a1, a2 in _iterEdges(ringA):
        for b1, b2 in _iterEdges(ringB):
            isect = _isectSegSeg(a1[0],a2[0],b1[0],b2[0])
            if isect:
                isectsA[(a1[0],a2[0])].append(isect)
                isectsB[(b1[0],b2[0])].append(isect)

    # Step 3b: Insert the intersection points
    # a vertex can appear twice in sequence in a vertex ring,
    # but not more (restricion implemented in _inorderExtend)
    for segment, isects in isectsA.items():
        _inorderExtend(ringA, segment[0], segment[1], isects)
    for segment, isects in isectsB.items():
        _inorderExtend(ringB, segment[0], segment[1], isects)

    # Step 4: Classify the edge fragments
    # type of edge fragments, besides the boundary line fragment (Table 2)
    fragmentTypeA = 'IN'  if operation == 'intersection' else 'OUT'
    fragmentTypeB = 'OUT' if operation == 'union' else 'IN'

    edgeFragments = defaultdict(list)
    _extendFragments(edgeFragments,ringA, polyB, fragmentTypeA)
    _extendFragments(edgeFragments,ringB, polyA, fragmentTypeB)

    # Step 5: Remove antiparallel overlapping boundary fragments
    if operation != 'intersection':
        removeParallel = []
        for key1 in edgeFragments.keys():
            for key2 in edgeFragments[key1]:
                if key2 in edgeFragments and key1 in edgeFragments[key2]:
                    removeParallel.append( (key1,key2) )
        for key1,key2 in removeParallel:
            edgeFragments[key1].remove(key2)
            if not edgeFragments[key1]: del edgeFragments[key1]

    # Step 6: Construct the result polygons and find their types
    output = []
    while edgeFragments:
        firstNode = next(iter(edgeFragments))
        sequence = []
        nextNode = firstNode
        while True:
            thisNode = edgeFragments[nextNode][0]
            sequence.append(thisNode)
            del edgeFragments[nextNode][0]
            if not edgeFragments[nextNode]:
                del edgeFragments[nextNode]
            if thisNode == firstNode:
                break
            nextNode = thisNode
        if sequence:
            sequence = list(dict.fromkeys(sequence))
            output.append(sequence)

    outputV = []
    for s in output:
        outputV.append( [Vector(v).freeze() for v in s])
    return outputV

# def plotPolygon(poly,vertsOrder,lineColor='k',fillColor='k',width=1.,fill=False,alpha = 0.2,order=100):
#     import matplotlib.pyplot as plt
#     x = [n[0] for n in poly] + [poly[0][0]]
#     y = [n[1] for n in poly] + [poly[0][1]]
#     if fill:
#         plt.fill(x[:-1],y[:-1],color=fillColor,alpha=alpha,zorder = order)
#     plt.plot(x,y,lineColor,linewidth=width,zorder=order)
#     if vertsOrder:
#         for i,(xx,yy) in enumerate(zip(x[:-1],y[:-1])):
#             plt.text(xx,yy,str(i),fontsize=12)
