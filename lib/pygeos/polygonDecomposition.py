# -------------------------------------------------------------------------------
# Implementation of a decomposition of a polygon with holes into convex polygons and triangles,
# based on the follwoing papers:
#
#   Fernández, J., Cánovas, L. and Pelegrín, B.
#   Algorithms for the decomposition of a polygon into convex polygons,
#   European Journal of Operational Research, 121, (2), (2000), 330-342
#
# and
#
#   From Fernández, J., Tóth, B., Cánovas, L. et al.
#   A practical algorithm for decomposing polygonal domains into convex
#   polygons by diagonals. TOP 16, 367–387 (2008).
#   
# Dependencies: PyGEOS
#
# Example:
# 
#       ring = [(0,0),(10,0),(12,2),(10,5),(5,4),(0,5),(1,0.5)]
#       hole = [(9,1.5),(9.5,3.5),(6.5,3.0),(9,1.5)]

#       gF = GeometryFactory()
#       ringCoords = [gF.createCoordinate(v) for v in ring+[ring[0]]]
#       holeCoords = [gF.createCoordinate(v) for v in hole+[hole[0]]]
#       P = gF.createPolygon(gF.createLinearRing(ringCoords) )
#       H = gF.createPolygon(gF.createLinearRing(holeCoords))

#       P = P.difference(H)
#       L = polygonDecomposition(P)
# -------------------------------------------------------------------------------

from itertools import *

# imports from PyGEOS, to be adapted if this file is moved to another folder
from .geom import GeometryFactory
from .algorithms import CGAlgorithms, LineIntersector

def _iterPoly(N):
    """
        N: The length of a polygon, where its vertices are expected to be in counter-clockwise order.
        return: Iterator delivers previous, this and next index in forward (counter-clockwise) 
                order, where this is initially the index of the first element.
    """
    for indx in range(N):
        iPrev = (indx-1)%N
        iThis = (indx)%N
        iNext = (indx+1)%N
        yield iPrev,iThis,iNext

def iterForwardFrom(startIndx,N):
    """
        Iterates in forward (counter-clockwise) order along the indices of the outer contour
        of the polygon with N vertices. 
        startIndx: Index to start with.
        N: length of the polygon
        return: Iterator delivers previous, this and next index in forward
                (counter-clockwise) order, where this is initially the index <startIndx>.
    """
    for indx in range(startIndx,startIndx+N+1):
        iPrev = (indx-1)%N
        iThis = (indx)%N
        iNext = (indx+1)%N
        yield iPrev,iThis,iNext

def iterBackwardFrom(startIndx,N):
    """
        Iterates in backward (clockwise) order along the indices of the outer contour
        of the polygon with N vertices. 
        startIndx: Index to start with.
        N: length of the polygon
        return: Iterator delivers previous, this and next index in backward
                (clockwise) order, where this is initially the index <startIndx>.
                Note that previous, this and next turn forward (counter-clockwise)
    """
    for indx in range(startIndx,startIndx-N-1,-1):
        iPrev = (indx-1)%N
        iThis = (indx)%N
        iNext = (indx+1)%N
        yield iPrev,iThis,iNext

def getNotchIndxs(poly):
    """
        poly: The outer contour of the polygon as a list of PyGEOS vertices (class Coordinate). 
              The vertices must be in counter-clockwise order.
        return: A list of all indices of vertices in poly, that are notches in the given polygon.
    """
    N = len(poly)
    return [iThis for iPrev,iThis,iNext in _iterPoly(N) if \
                CGAlgorithms.orientationIndex(poly[iPrev],poly[iThis],poly[iNext]) < 0.]

def getNotches(poly):
    """
        poly: The outer contour of the polygon as a list of PyGEOS vertices (class Coordinate). 
              The vertices must be in counter-clockwise order.
        return: A list of all vertices in poly, that are notches in the given polygon.
    """
    nIndxs = getNotchIndxs(poly)
    return [poly[i] for i in nIndxs]

def nextIndex(startIndx,N):
    """
        Next index in forward (counter-clockwise) order along the indices of the outer contour
        of the polygon with N vertices. 
        startIndx: Index to start with.
        N: length of the polygon
        return: Next index in forward (counter-clockwise) order after startIndx.
    """
    return (startIndx+1)%N

def prevIndex(startIndx,N):
    """
        Previous index in backward (clockwise) order along the indices of the outer contour
        of the polygon with N vertices. 
        startIndx: Index to start with.
        N: length of the polygon
        return: Previous index in backward (clockwise) order before startIndx.
     """
    return (startIndx-1)%N

def subtractPolygons(iPoly1,iPoly2):
    """
        Subtracts iPoly2 from iPoly1. The iPoly2 indices must be a
        subset of the iPoly1 indices. 

        iPoly1: The outer contour of a polygon1 as a list of indices.
                The vertices must be in counter-clockwise order.
        iPoly2: The outer contour of a polygon2 as a list of indices.  
                The vertices must be in counter-clockwise order.
        return: Remaining polygon after removing iPoly2 from iPoly1.
    """
    sIndx = iPoly2[0]
    eIndx = iPoly2[-1]
    if sIndx < eIndx:
        return iPoly1[0:sIndx+1] + iPoly1[eIndx:]
    else:
        return iPoly1[eIndx:sIndx+1] 

def MP1ccw(poly, iniIndx):
    """
        Procedure MP1 in counter-clockwise order, from
            Fernández, J., Cánovas, L. and Pelegrín, B.
            Algorithms for the decomposition of a polygon into convex polygons,
            European Journal of Operational Research, 121, (2), (2000), 330-342

        poly: The outer contour of the polygon as a list of PyGEOS vertices (class Coordinate). 
              The vertices must be in counter-clockwise order.
        iniIndx: Index of initial vertex
        return: A decomposed polygon as a list of indices of poly and a boolean,
                which is True, when the end is reached.
    """
    N = len(poly)
    if N < 4:
        return (poly, True)

    iPoly = [i for i in range(N)]
    iL = [iniIndx, nextIndex(iniIndx,N)]

    # initial vertices indices
    iV1, iV2 = iL[0], iL[1]
    iVim1 = iV1
    iVi = iV2
    iVip1 = nextIndex(iV2,N)
    # add vertices to iL as long as all three conditons are true
    while len(iL) < N:
        if CGAlgorithms.orientationIndex(poly[iVim1], poly[iVi], poly[iVip1]) >= 0 and \
        CGAlgorithms.orientationIndex(poly[iVi], poly[iVip1], poly[iV1]) >= 0 and \
        CGAlgorithms.orientationIndex(poly[iVip1], poly[iV1], poly[iV2]) >= 0:
            iL.append(iVip1)
        else:
            break
        iVim1 = iVi
        iVi = iVip1
        iVip1 = nextIndex(iVip1,N)

    # When iL contains the whole polygon, we are done
    if len(iL) == N:
        return (iL, True)
    else:
        while len(iL) > 2:
            PmL = [poly[i] for i in subtractPolygons(iPoly,iL)]
            L = [poly[i] for i in iL+[iL[0]]]
            notches = [notch for notch in getNotches(PmL) if notch not in L and \
                CGAlgorithms.isPointInRing(notch, L+[L[0]])]
            if not notches:
                break
            iV1 = iL[0]
            iVk = iL[-1]
            iL = iL[:-1]
            for notch in notches:
                sideOfVk = CGAlgorithms.orientationIndex(poly[iVk], poly[iV1], notch)
                if sideOfVk:  # !=0
                    iL = [iV for iV in iL if CGAlgorithms.orientationIndex(poly[iV], poly[iV1],notch) != sideOfVk]

        return (iL, False)

def MP1cw(poly, iniIndx):
    """
        Procedure MP1 in clockwise order, from
            Fernández, J., Cánovas, L. and Pelegrín, B.
            Algorithms for the decomposition of a polygon into convex polygons,
            European Journal of Operational Research, 121, (2), (2000), 330-342

        poly: The outer contour of the polygon as a list of PyGEOS vertices (class Coordinate). 
              The vertices must be in counter-clockwise order.
        iniVerts: List of initial vertices
        return: A decomposed polygon as a list of indices of poly and a boolean,
                which is True, when the end is reached.
    """
    N = len(poly)
    if N < 4:
        return poly, True

    iPoly = [i for i in range(N)]
    iL = [prevIndex(iniIndx,N),iniIndx]

    iV1, iV2 = iL[1], iL[0]
    iVim1 = iV1
    iVi = iV2
    iVip1 = prevIndex(iV2,N)
    while len(iL) < N:
        if CGAlgorithms.orientationIndex(poly[iVim1], poly[iVi], poly[iVip1]) <= 0 and \
        CGAlgorithms.orientationIndex(poly[iVi], poly[iVip1], poly[iV1]) <= 0 and \
        CGAlgorithms.orientationIndex(poly[iVip1], poly[iV1], poly[iV2]) <= 0:
            iL.insert(0,iVip1)
        else:
            break
        iVim1 = iVi
        iVi = iVip1
        iVip1 = prevIndex(iVip1,N)

    # if iL contains the whole polygon, we are done
    if len(iL) == N:
        return iL, True
    else:
        while len(iL) > 2:
            PmL = [poly[i] for i in subtractPolygons(iPoly,iL)]
            L = [poly[i] for i in iL+[iL[0]]]
            notches = [notch for notch in getNotches(PmL) if notch not in L and \
                CGAlgorithms.isPointInRing(notch, L+[L[0]])]
            if not notches:
                break
            iV1 = iL[-1]
            iVk = iL[0]
            iL = iL[1:]
            for notch in notches:
                sideOfVk = CGAlgorithms.orientationIndex(poly[iVk], poly[iV1], notch)
                if sideOfVk:  # !=0
                    iL = [iV for iV in iL if CGAlgorithms.orientationIndex(poly[iV], poly[iV1], notch) != sideOfVk]

        return (iL, False)

def MP5(poly):
    """
        Procedure MP5 from
            From Fernández, J., Tóth, B., Cánovas, L. et al.
            A practical algorithm for decomposing polygonal domains into convex
            polygons by diagonals. TOP 16, 367–387 (2008).

        poly: The outer contour of the polygon as a list of PyGEOS vertices (class Coordinate). 
              The vertices must be in counter-clockwise order.
        return: A decomposed polygon as a list of indices of poly and a boolean,
                which is True, when the end is reached.
    """
    N = len(poly)
    iPoly = [i for i in range(N)]
    if N < 4:
        return iPoly, True

    # In the original version of the algorithm, MP1ccw or MP1cw is called for every notch until a resulting
    # polygon is found. For conscutive sequences of notches like in curved parts of the polygon, this is
    # very inefficient, as there can't be a result between two consecutive notches. We therefore call MP1
    # only, when a notch is followed by a fnon-notch.
    isNotch = [CGAlgorithms.orientationIndex(poly[iPrev],poly[iThis],poly[iNext]) < 0. for iPrev,iThis,iNext in _iterPoly(N)]

    # If no notch found, the polygon is already convex
    if not any(isNotch):
        return iPoly, True

    nextNotchIndxs = [indx for indx,notches in enumerate(zip(isNotch, isNotch[1:]+[isNotch[0]])) \
                            if notches[0] and not notches[1] ]

    for notchIndx in nextNotchIndxs:
        # Obtain a polygon ccwL as a partition of poly using MP1 in
        # counter-clockwise order.
        ccwL,ccwEnd = MP1ccw(poly, notchIndx)

        if ccwEnd:
            return ccwL, True

        # MP1 + notch checking = MP3
        if len(ccwL) > 2:
            notches  = nextNotchIndxs
            if ccwL[0] in notches or ccwL[-1] in notches:
                return ccwL, False
 
    prevNotchIndxs = [indx for indx,notches in enumerate(zip([isNotch[-1]]+isNotch[:-1], isNotch)) \
                            if not notches[0] and notches[1] ]
    prevNotchIndxs.reverse()

    for notchIndx in prevNotchIndxs:
        # Obtain a polygon cwL as a partition of poly using MP1 in
        # clockwise order.
        cwL,cwEnd = MP1cw(poly, notchIndx)

        if cwEnd:
            return cwL, True

        # MP1 + notch checking = MP3
        if len(cwL) > 2:
            notches  = prevNotchIndxs
            if cwL[0] in notches or cwL[-1] in notches:
                return cwL, False

    raise Exception('MP5','This should not happen!')

def getSegmentHoleIntersectionEdges(segment, hole):
    """
     Returns all edges of the given hole that intersect the given segment
     segment: tuple Vi-Vf of the segment's end points
     hole: PyGEOS object of a hole (class Polygon)
     return A list of dictionaries of all intersected edges. The keys of
     the dictionary are: 
        'p': The intersection point.
        'e': The edge as a tuple of its end points.
        'h': PyGEOS object of the hole.
    """
    isectEdges = []
    c = hole.coords
    holeEdges = [(hole.coords[k:k+2]) for k in range(len(hole.coords) - 1)]
    s1,s2 = segment
    intersector = LineIntersector()
    for v1,v2 in holeEdges:
        intersector.computeLinesIntersection(s1,s2,v1,v2)
        if intersector.intersections:
            isectEdges.append( {'p':intersector.intersectionPts[0], 'e':(v1,v2), 'h':hole} )

    return isectEdges

def drawTrueDiagonal(prevDiagonal, holesInC):
    """
        Procedure drawTrueDiagonal from
            From Fernández, J., Tóth, B., Cánovas, L. et al.
            A practical algorithm for decomposing polygonal domains into convex
            polygons by diagonals. TOP 16, 367–387 (2008).
        prevDiagonal: PyGEOS object (class LineString) of the diagonal.
        holesInC: List of holes (PyGEOS Polygons) that are intersected by diagonal.
    """
    diagSegment = [prevDiagonal.coords[0],prevDiagonal.coords[1]]
    Vi = diagSegment[0]

    previousClosestVertex = diagSegment[1]
    while True:
        # Find all the edges of holes which intersect the diagonal and calculate
        # the corresponding intersection points.
        edges = []
        for hole in holesInC:
            edges.extend( getSegmentHoleIntersectionEdges(diagSegment, hole) )

        if not edges:
            break

        # Find the intersection point pC closest to Vi and its edge e
        pC_e = min( [edge for edge in edges], 
            key=lambda e: CGAlgorithms.distancePointPoint(Vi,e['p']))

        # Find closest end-point Es of the edge e
        Es = min( [ei for ei in pC_e['e']], key=lambda ei: CGAlgorithms.distancePointPoint(Vi,ei) )

        if Es == previousClosestVertex:
            return Vi, Es, pC_e['h']

        diagSegment[1] = Es # new diagonal is Vi-Es
        H = pC_e['h']   # hole H' to absorb
        previousClosestVertex = Es

    return Vi, Es, H

def absHol(poly,holes):
    """
        Procedure absHol from
            From Fernández, J., Tóth, B., Cánovas, L. et al.
            A practical algorithm for decomposing polygonal domains into convex
            polygons by diagonals. TOP 16, 367–387 (2008).

        poly: The outer contour of the polygon as a list of PyGEOS vertices (class Coordinate). 
              The vertices must be in counter-clockwise order.
        return: A list of decomposed polygons. Every polygon in this list is itself a
                list of PyGEOS vertices (class Coordinate).
    """
    gF = GeometryFactory()

    LPCP = []
    Q = poly

    while True:
        # Obtain a convex polygon C of the partition of Q using the algorithm MP5
        iC, end  = MP5(Q)
        C = [Q[i] for i in iC]

        # Let diagonal = Vi-Vf be the diagonal generating the polygon C
        diagonal = gF.createLineString([C[0],C[-1]])
        Cgeos = gF.createPolygon(gF.createLinearRing(C+[C[0]]))

        diagonalIsCutByAHole = False
        holesInC = []
        for hole in holes:
            # Check if diagonal is cut by a hole .. ('T********' is DE-9IM code for 'intersects')
            if not diagonalIsCutByAHole and diagonal.relate(hole,'T********'): 
                diagonalIsCutByAHole = True
            # .. or there is a hole inside C
            if not Cgeos.disjoint(hole):
                holesInC.append(hole)

        # If diagonal is cut by a hole or there is a hole inside C
        if diagonalIsCutByAHole or holesInC:
            # If d is not cut by a hole
            if not diagonalIsCutByAHole:
                # diagonal = Vi-Vhole, where Vhole is a vertex of an inside hole
                diagonal = gF.createLineString([C[0],holesInC[0].coords[0]])
            # From this diagonal, find new diagonal (not intersected by any of
            # the holes nor by the border polygon) by the algorithm drawTrueDiagonal.
            Vi, Es, H = drawTrueDiagonal(diagonal, holesInC)

            # ToDo !!!!!!! In rare cases, this diagonal (bridge) crosses the polygon.
            # This shoild be checked and corrected..

            # Absorb H' into Q
            holes.remove(H)
            Vi_indx = iC[0]#Q.index(Vi)
            Hc = H.coords
            Es_indx = Hc.index(Es)
            Q = Q[0:Vi_indx+1] + Hc[Es_indx:-1] + Hc[0:Es_indx+1] + Q[Vi_indx:]
        else:
            LPCP.append(C)
            if end:
                break
            iQ = [i for i in range(len(Q))]
            iQ = subtractPolygons(iQ, iC)
            Q = [Q[i] for i in iQ]

    return LPCP

def polygonDecompositionWithHoles(P):
    """
        P: A PyGEOS polygon (class Polygon) with optional interior geometries as holes. 
        return: A list of decomposed polygons. Every polygon in this list is itself a
        list of PyGEOS vertices (class Coordinate)
    """
    gF = GeometryFactory()

    exteriorRing = P.exterior
    # the exterior polygon must be counter-clockwise
    if not exteriorRing.is_ccw:
        exteriorRing.coords.reverse()
    poly = exteriorRing.coords[:-1]

    holes = []
    for interiorRing in P.interiors:
        # the interior polygons (holes) must be clockwise
        if interiorRing.is_ccw:
            interiorRing.coords.reverse()
        holes.append(gF.createPolygon(interiorRing))

           # call Fernandez polygon decomposition
    return absHol(poly,holes)

def polygonDecomposition(P):
    """
        P: A PyGEOS polygon (class Polygon) without interior geometries as holes. 
        return: A list of decomposed polygons. Every polygon in this list is itself a
        list of PyGEOS vertices (class Coordinate)
    """
    gF = GeometryFactory()

    exteriorRing = P.exterior
    # the exterior polygon must be counter-clockwise
    if not exteriorRing.is_ccw:
        exteriorRing.coords.reverse()
    poly = exteriorRing.coords[:-1]

    LPCP = []
    Q = poly

    while True:
        # Obtain a convex polygon C of the partition of Q using the algorithm MP5
        iC, end  = MP5(Q)
        C = [Q[i] for i in iC]

        LPCP.append(C)
        if end:
            break
        iQ = [i for i in range(len(Q))]
        iQ = subtractPolygons(iQ, iC)
        Q = [Q[i] for i in iQ]

    return LPCP
