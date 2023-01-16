# -*- coding:utf-8 -*-

# ##### BEGIN LGPL LICENSE BLOCK #####
# GEOS - Geometry Engine Open Source
# http://geos.osgeo.org
#
# Copyright (C) 2011 Sandro Santilli <strk@kbt.io>
# Copyright (C) 2005 2006 Refractions Research Inc.
# Copyright (C) 2001-2002 Vivid Solutions Inc.
# Copyright (C) 1995 Olivier Devillers <Olivier.Devillers@sophia.inria.fr>
#
# This is free software you can redistribute and/or modify it under
# the terms of the GNU Lesser General Public Licence as published
# by the Free Software Foundation.
# See the COPYING file for more information.
#
# ##### END LGPL LICENSE BLOCK #####

# <pep8 compliant>

# ----------------------------------------------------------
# Partial port (version 3.7.0) by: Stephen Leger (s-leger)
#
# ----------------------------------------------------------

import logging
logger = logging.getLogger("pygeos.algorithms")
from math import floor, isfinite, sqrt, pi, atan2
from .shared import (
    quicksort,
    GeomTypeId,
    Location,
    Envelope,
    Quadrant,
    Coordinate,
    CoordinateSequence,
    LinearComponentExtracter,
    CoordinateFilter
    )
from .index_bintree import (
    Bintree,
    Interval
    )
from .index_intervaltree import (
    SortedPackedIntervalRTree
    )


# index
EPSILON_SINGLE = 1e-5
EPSILON = 1e-12
EPSILON_SQUARE = EPSILON * EPSILON
USE_FUZZY_LINE_INTERSECTOR = False


class ItemVisitor():
    """
     * A visitor for items in an index.
    """
    def visitItem(self) -> None:
        raise NotImplementedError()


# index/chain


class MonotoneChain():
    """
     * Monotone Chains are a way of partitioning the segments of a linestring to
     * allow for fast searching of intersections.
     *
     * They have the following properties:
     *
     * - the segments within a monotone chain never intersect each other
     * - the envelope of any contiguous subset of the segments in a monotone
     *   chain is equal to the envelope of the endpoints of the subset.
     *
     * Property 1 means that there is no need to test pairs of segments from
     * within the same monotone chain for intersection.
     * Property 2 allows an efficient binary search to be used to find the
     * intersection points of two monotone chains.
     *
     * For many types of real-world data, these properties eliminate
     * a large number of segment comparisons, producing substantial speed gains.
     *
     * One of the goals of this implementation of MonotoneChains is to be
     * as space and time efficient as possible. One design choice that aids this
     * is that a MonotoneChain is based on a subarray of a list of points.
     * This means that new arrays of points (potentially very large) do not
     * have to be allocated.
     *
     * MonotoneChains support the following kinds of queries:
     *
     * - Envelope select: determine all the segments in the chain which
     *   intersect a given envelope
     * - Overlap: determine all the pairs of segments in two chains whose
     *   envelopes overlap
     *
     * This implementation of MonotoneChains uses the concept of internal iterators
     * to return the resultsets for the above queries.
     * This has time and space advantages, since it
     * is not necessary to build lists of instantiated objects to represent the segments
     * returned by the query.
     * However, it does mean that the queries are not thread-safe.
    """
    def __init__(self, coords, start, end, context):

        self.coords = coords
        self._env = None
        self.context = context
        self.start = start
        self.end = end
        self.id = -1

    @property
    def envelope(self):

        if self._env is None:
            self._env = Envelope(
                self.coords[self.start],
                self.coords[self.end])

        return self._env

    def getLineSegment(self, index: int, ls) -> None:
        """
         *  Set given LineSegment with points of the segment starting
         *  at the given index.
        """
        ls.p0 = self.coords[index]
        ls.p1 = self.coords[index + 1]

    def select(self, searchEnv, mcs) -> None:
        self.computeSelect(searchEnv, self.start, self.end, mcs)

    def _computeSelect(self, searchEnv, start0: int, end0: int, mcs) -> None:
        p0 = self.coords[start0]
        p1 = self.coords[end0]
        mcs.tempEnv1.__init__(p0, p1)

        # terminating condition for the recursion
        if end0 - start0 == 1:
            mcs.select(self, start0)
            return

        # nothing to do if the envelopes don't overlap
        if not searchEnv.intersects(mcs.tempEnv1):
            return

        # the chains overlap,so split each in half and iterate (binary search)
        mid = int((start0 + end0) / 2)
        if start0 < mid:
            self._computeSelect(searchEnv, start0, mid, mcs)

        if mid < end0:
            self._computeSelect(searchEnv, mid, end0, mcs)

    def computeOverlaps(self, mc, mco) -> None:
        self._computeOverlaps(self.start, self.end, mc, mc.start, mc.end, mco)

    def _computeOverlaps(self,
            start0: int, end0: int, mc,
            start1: int, end1: int, mco) -> None:

        # terminating condition for the recursion
        if end0 - start0 == 1 and end1 - start1 == 1:
            mco.overlap(self, start0, mc, start1)
            return

        p1 = self.coords[start0]
        p2 = self.coords[end0]
        q1 = mc.coords[start1]
        q2 = mc.coords[end1]
        # nothing to do if the envelopes of these chains don't overlap
        # MonotoneChainOverlapAction
        mco.tempEnv1.__init__(p1, p2)
        mco.tempEnv2.__init__(q1, q2)

        if not mco.tempEnv1.intersects(mco.tempEnv2):
            return

        mid0 = int((start0 + end0) / 2)
        mid1 = int((start1 + end1) / 2)

        if start0 < mid0:
            if start1 < mid1:
                self._computeOverlaps(start0, mid0, mc, start1, mid1, mco)
            if mid1 < end1:
                self._computeOverlaps(start0, mid0, mc, mid1, end1, mco)

        if mid0 < end0:
            if start1 < mid1:
                self._computeOverlaps(mid0, end0, mc, start1, mid1, mco)
            if mid1 < end1:
                self._computeOverlaps(mid0, end0, mc, mid1, end1, mco)

    def __str__(self):
        return "MonotonChain {} {}-{}".format(self.id, self.start, self.end)


class MonotoneChainBuilder():
    """
     * Constructs {MonotoneChain}s
     * for sequences of {Coordinate}s.
    """

    @staticmethod
    def getChains(coords, context=None, mcList=[]) -> list:
        """
         * Fill the provided vector with newly-allocated MonotoneChain objects
         * for the given CoordinateSequence.
        """
        startIndex = []
        MonotoneChainBuilder.getChainStartIndices(coords, startIndex)
        n = len(startIndex)

        if n > 0:
            for i in range(n - 1):
                mc = MonotoneChain(coords, startIndex[i], startIndex[i + 1], context)
                mcList.append(mc)

        logger.debug("MonotoneChainBuilder.getChains(%s) for coords:%s\n%s",
            len(mcList),
            coords,
            "\n".join([str(mc) for mc in mcList]))
            
        return mcList

    @staticmethod
    def getChainStartIndices(coords, startIndexList: list) -> None:
        """
         * Fill the given vector with start/end indexes of the monotone chains
         * for the given CoordinateSequence.
         * The last entry in the array points to the end point of the point
         * array,
         * for use as a sentinel.
        """
        start = 0
        startIndexList.append(start)
        n = len(coords) - 1
        while (True):
            last = MonotoneChainBuilder._findChainEnd(coords, start)
            startIndexList.append(last)
            start = last
            if start >= n:
                break

    @staticmethod
    def _findChainEnd(coords, start: int) -> int:
        """
         * Finds the index of the last point in a monotone chain
         * starting at a given point.
         * Any repeated points (0-length segments) will be included
         * in the monotone chain returned.
         *
         * @return the index of the last point in the monotone chain
         *         starting at start.
         *
         * NOTE: aborts if 'start' is >= pts.getSize()
        """
        safeStart = start
        npts = len(coords)
        
        # skip any zero-length segments at the start of the sequence
        # (since they cannot be used to establish a quadrant)
        while (safeStart < npts - 1 and coords[safeStart] == coords[safeStart + 1]):
            safeStart += 1

        # check if there are NO non-zero-length segments
        if safeStart >= npts - 1:
            return npts - 1

        # determine overall quadrant for chain
        chainQuad = Quadrant.from_coords(coords[safeStart], coords[safeStart + 1])

        last = start + 1
        while last < npts:
            if coords[last - 1] != coords[last]:
                quad = Quadrant.from_coords(coords[last - 1], coords[last])
                # logger.debug("MonotoneChainBuilder._findChainEnd() Quadrants: %s %s", chainQuad, quad)
                if quad != chainQuad:
                    break
            last += 1

        return last - 1


class MonotoneChainSelectAction():

    def __init__(self):
        # geom.LineSegment
        self._selectedSegment = LineSegment()
        # these envelopes are used during the MonotoneChain search process
        self.tempEnv1 = Envelope()

    def select(self, mc, start: int) -> None:
        mc.getLineSegment(start, self._selectedSegment)
        self._select(self._selectedSegment)

    def _select(self, seg) -> None:
        """
         * This is a convenience function which can be overridden
         * to obtain the actual line segment which is selected
        """
        pass


class MonotoneChainOverlapAction():
    """
     * The action for the internal iterator for performing
     * overlap queries on a MonotoneChain
    """
    def __init__(self):
        # geom.LineSegment
        self.overlapSeg1 = LineSegment()
        self.overlapSeg2 = LineSegment()

        # these envelopes are used during the MonotoneChain search process
        self.tempEnv1 = Envelope()
        self.tempEnv2 = Envelope()

    def overlap(self, mc1, start1: int, mc2, start2: int) -> None:
        """
         * This function can be overridden if the original chains are needed
         *
         * @param start1 the index of the start of the overlapping segment
         *               from mc1
         * @param start2 the index of the start of the overlapping segment
         *               from mc2
        """
        mc1.getLineSegment(start1, self.overlapSeg1)
        mc2.getLineSegment(start2, self.overlapSeg2)
        self._overlap(self.overlapSeg1, self.overlapSeg2)

    def _overlap(self, seg1, seg2) -> None:
        """
         * This is a convenience function which can be overridden to
         * obtain the actual line segments which overlap
         * @param seg1
         * @param seg2
        """
        pass


class MonotoneChainIndexer():

    def getChainStartIndices(self, coords, startIndexList):
        start = 0
        startIndexList.append(start)
        while True:
            last = self._findChainEnd(coords, start)
            startIndexList.append(last)
            start = last
            if start >= len(coords) - 1:
                break

    def _findChainEnd(self, coords, start):
        """
         * @return the index of the last point in the monotone chain
        """
        chainQuad = Quadrant.from_coords(coords[start], coords[start + 1])
        last = start + 1
        while last < len(coords):
            quad = Quadrant.from_coords(coords[last - 1], coords[last])
            if quad != chainQuad:
                break
            last += 1
        return last - 1


class LineSegment():
    """
     * A line segment
    """
    def __init__(self, x0=None, y0=None, x1=None, y1=None):

        if x0 is None:
            x0, y0, x1, y1 = 0, 0, 0, 0

        elif y0 is None:
            y1 = x0.p1.y
            x1 = x0.p1.x
            y0 = x0.p0.y
            x0 = x0.p0.x

        elif x1 is None:
            y1 = y0.y
            x1 = y0.x
            y0 = x0.y
            x0 = x0.x

        self.p0 = Coordinate(x0, y0)
        self.p1 = Coordinate(x1, y1)

    def setCoordinates(self, p0, p1):
        self.p0.x = p0.x
        self.p0.y = p0.y
        self.p1.x = p1.x
        self.p1.y = p1.y

    @property
    def length(self):
        return CGAlgorithms.distancePointPoint(self.p0, self.p1)

    def distance(self, other) -> float:
        if type(other).__name__ == 'LineSegment':
            return CGAlgorithms.distanceLineLine(self.p0, self.p1, other.p0, other.p1)
        else:
            return CGAlgorithms.distancePointLine(other, self.p0, self.p1)

    def projectionFactor(self, p) -> float:
        if p == self.p0:
            return 0.0
        if p == self.p1:
            return 1.0
        # Otherwise, use comp.graphics.algorithms Frequently Asked Questions method
        """
          (1)             AC dot AB
                       r = ---------
                             ||AB||^2
                    r has the following meaning:
                    r=0 P = A
                    r=1 P = B
                    r<0 P is on the backward extension of AB
                    r>1 P is on the forward extension of AB
                    0<r<1 P is interiors to AB
        """
        p0, p1 = self.p0, self.p1

        dx = p1.x - p0.x
        dy = p1.y - p0.y
        len2 = dx * dx + dy * dy
        r = ((p.x - p0.x) * dx + (p.y - p0.y) * dy) / len2
        return r

    def orientationIndex(self, seg) -> int:
        orient0 = CGAlgorithms.orientationIndex(self.p0, self.p1, seg.p0)
        orient1 = CGAlgorithms.orientationIndex(self.p0, self.p1, seg.p1)
        # this handles the case where the points are L or collinear
        if orient0 >= 0 and orient1 >= 0:
            return max(orient0, orient1)
        # this handles the case where the points are R or collinear
        if orient0 <= 0 and orient1 <= 0:
            return max(orient0, orient1)
        # points lie on opposite sides ==> indeterminate orientation
        return 0

    def pointAlongOffset(self, t: float, offsetDistance: float, res) -> None:

        dx = self.p1.x - self.p0.x
        dy = self.p1.y - self.p0.y

        # the point on the segment line
        segx = self.p0.x + t * dx
        segy = self.p0.y + t * dy

        length = sqrt(dx * dx + dy * dy)

        ux = 0.0
        uy = 0.0

        if offsetDistance != 0.0:
            if length <= 0.0:
                raise ValueError("Cannot compute offset from zero-length line segment")
            # u is the vector that is the length of the offset,
            # in the direction of the segment
            ux = offsetDistance * dx / length
            uy = offsetDistance * dy / length

        # the offset point is the seg point plus the offset
        # vector rotated 90 degrees CCW
        res.x = segx - uy
        res.y = segy + ux

    def __str__(self):
        return "{} - {}".format(self.p0, self.p1)


class UniqueCoordinateArrayFilter(CoordinateFilter):
    """
     *  A CoordinateFilter that fills a vector of Coordinate const pointers.
     *  The set of coordinates contains no duplicate points.
    """
    def __init__(self, target):
        self._unique = {}
        self.pts = target

    def filter_ro(self, coord):
        if self._unique.get(coord) is None:
            self._unique[coord] = coord
            self.pts.append(coord)


class ReallyLessThen():

    def __init__(self, origin):
        self.origin = origin

    def compare(self, p1, p2) -> bool:
        return self.polarCompare(self.origin, p1, p2) == -1

    def polarCompare(self, o, p, q) -> int:

        dxp = p.x - o.x
        dyp = p.y - o.y
        dxq = q.x - o.x
        dyq = q.y - o.y

        orient = CGAlgorithms.computeOrientation(o, p, q)

        if orient == CGAlgorithms.COUNTERCLOCKWISE:
            return 1
        elif orient == CGAlgorithms.CLOCKWISE:
            return -1

        op = dxp * dxp + dyp * dyp
        oq = dxq * dxq + dyq * dyq
        if op < oq:
            return -1
        elif op > oq:
            return 1
        return 0


# algorithms


class Angle():
    """
     * Utility functions for working with angles.
     * Unless otherwise noted, methods in this class express angles in radians.
    """
    PI_TIMES_2 = 2.0 * pi
    PI_OVER_2 = pi / 2.0
    PI_OVER_4 = pi / 4.0

    @staticmethod
    def angle(p0, p1=None) -> float:
        if p1 is None:
            x = p0.x
            y = p0.y
        else:
            x = p1.x - p0.x
            y = p1.y - p0.y
        return atan2(y, x)

    @staticmethod
    def angleBetween(p0, p1, p2) -> float:
        """
         * Returns the angle between two vectors.
         *
         * The computed angle will be in the range (-Pi, Pi].
         * A positive result corresponds to a counterclockwise rotation
         * from v1 to v2;
         * a negative result corresponds to a clockwise rotation.
         * @param tip1 the tip of v1
         * @param tail the tail of each vector
         * @param tip2 the tip of v2
         * @return the angle between v1 and v2, relative to v1
        """
        a1 = Angle.angle(p1, p0)
        a2 = Angle.angle(p1, p2)
        return Angle.diff(a1, a2)

    @staticmethod
    def angleBetweenOriented(p0, p1, p2) -> float:
        """
         * Returns the oriented smallest angle between two vectors.
         *
         * The computed angle will be in the range (-Pi, Pi].
         * A positive result corresponds to a counterclockwise rotation
         * from v1 to v2;
         * a negative result corresponds to a clockwise rotation.
         * @param tip1 the tip of v1
         * @param tail the tail of each vector
         * @param tip2 the tip of v2
         * @return the angle between v1 and v2, relative to v1
        """
        a1 = Angle.angle(p1, p0)
        a2 = Angle.angle(p1, p2)
        delta = a2 - a1
        if delta <= -pi:
            return delta + Angle.PI_TIMES_2
        if delta > pi:
            return delta - Angle.PI_TIMES_2
        return delta

    @staticmethod
    def normalize(angle: float) -> float:
        """
         * Computes the normalized value of an angle, which is the
         * equivalent angle in the range ( -Pi, Pi ].
         * @param angle the angle to normalize
         * @return an equivalent angle in the range (-Pi, Pi]
        """
        while angle > pi:
            angle -= Angle.PI_TIMES_2

        while angle <= -pi:
            angle += Angle.PI_TIMES_2

        return angle

    @staticmethod
    def normalizePositive(angle: float) -> float:
        """
         * Computes the normalized value of an angle, which is the
         * equivalent angle in the range ( -Pi, Pi ].
         * @param angle the angle to normalize
         * @return an equivalent angle in the range (-Pi, Pi]
        """
        if angle < 0.0:
            while angle < 0.0:
                angle += Angle.PI_TIMES_2

            if angle >= Angle.PI_TIMES_2:
                angle = 0.0
        else:
            while angle >= Angle.PI_TIMES_2:
                angle -= Angle.PI_TIMES_2

            if angle < 0.0:
                angle = 0.0

        return angle

    @staticmethod
    def diff(ang1: float, ang2: float) -> float:
        if ang1 < ang2:
            delta = ang2 - ang1
        else:
            delta = ang1 - ang2

        if delta > pi:
            delta = Angle.PI_TIMES_2 - delta

        return delta


class ConvexHull():
    """
     * Computes the convex hull of a Geometry.
     *
     * The convex hull is the smallest convex Geometry that contains all the
     * points in the input Geometry.
     *
     * Uses the Graham Scan algorithm.
    """
    def __init__(self, geom):
        self._factory = geom._factory
        self.inputCoords = []
        self.extractCoordinates(geom)

    def toCoordinateSequence(self, coords):
        """
         * Create a CoordinateSequence from the Coordinate.ConstVect
         * This is needed to construct the geometries.
         * Here coordinate copies happen
        """
        csf = self._factory.coordinateSequenceFactory
        return csf.create(coords)

    def extractCoordinates(self, geom) -> None:
        filter = UniqueCoordinateArrayFilter(self.inputCoords)
        geom.apply_ro(filter)

    def computeOctPts(self, src, tgt) -> None:
        # Initialize all slots with first input coordinate
        tgt.clear()
        tgt.extend([src[0] for i in range(8)])
        for coord in src:
            if coord.x < tgt[0].x:
                tgt[0] = coord
            if coord.x - coord.y < tgt[1].x - tgt[1].y:
                tgt[1] = coord
            if coord.y > tgt[2].y:
                tgt[2] = coord
            if coord.x + coord.y > tgt[3].x + tgt[3].y:
                tgt[3] = coord
            if coord.x > tgt[4].x:
                tgt[4] = coord
            if coord.x - coord.y > tgt[5].x - tgt[5].y:
                tgt[5] = coord
            if coord.y < tgt[6].y:
                tgt[6] = coord
            if coord.x + coord.y < tgt[7].x + tgt[7].y:
                tgt[7] = coord

    def computeOctRing(self, src, tgt) -> bool:
        self.computeOctPts(src, tgt)
        # Remove consecutive equal Coordinates
        tmp = [co for i, co in enumerate(tgt) if i == 0 or tgt[i - 1] is not co]
        tgt.clear()
        tgt.extend(tmp)
        logger.debug("ConvexHull.computeOctRing() %s", [str(co) for co in tgt])
        if len(tgt) < 3:
            return False
        # close ring
        tgt.append(tgt[0])
        return True

    def reduce(self, coords) -> None:
        """
         * Uses a heuristic to reduce the number of points scanned
         * to compute the hull.
         * The heuristic is to find a polygon guaranteed to
         * be in (or on) the hull, and eliminate all points inside it.
         * A quadrilateral defined by the extremal points
         * in the four orthogonal directions
         * can be used, but even more inclusive is
         * to use an octilateral defined by the points in the
         * 8 cardinal directions.
         *
         * Note that even if the method used to determine the polygon
         * vertices is not 100% robust, this does not affect the
         * robustness of the convex hull.
         *
         * To satisfy the requirements of the Graham Scan algorithm,
         * the resulting array has at least 3 entries.
         *
         * @param pts The vector of const Coordinate pointers
         *            to be reduced (to at least 3 elements)
         *
         * WARNING: the parameter will be modified
        """
        polyPts = []
        if not self.computeOctRing(coords, polyPts):
            # unable to compute interiors polygon for some reason
            return

        # add points defining polygon
        reducedSet = [polyPts[0], polyPts[-1]]
        """
         * Add all unique points not in the interiors poly.
         * CGAlgorithms.isPointInRing is not defined for points
         * actually on the ring, but this doesn't matter since
         * the points of the interiors polygon are forced to be
         * in the reduced set.
        """
        for p in coords:
            if not CGAlgorithms.isPointInRing(p, polyPts):
                reducedSet.append(p)

        self.inputCoords.clear()
        self.inputCoords.extend(reducedSet)

        if len(self.inputCoords) < 3:
            self.padArray3(self.inputCoords)

    def padArray3(self, coords) -> None:
        for i in range(len(coords), 3):
            coords.append(coords[0])

    def preSort(self, coords) -> None:
        # find the lowest point in the set. If two or more points have
        # the same minimum y coordinate choose the one with the minimum x.
        # This focal point is put in array location pts[0].
        for i, p1 in enumerate(coords):
            p0 = coords[0]
            if p1.y < p0.y or (p1.y == p0.y and p1.x < p0.x):
                coords[0], coords[i] = p1, p0

        logger.debug("ConvexHull.preSort() before radial sort: %s", [str(co) for co in coords])

        # sort the points radially around the focal point.
        rls = ReallyLessThen(coords[0])
        quicksort(coords, rls.compare)

        logger.debug("ConvexHull.preSort() after radial sort: %s", [str(co) for co in coords])

    def grahamScan(self, coords, res) -> None:
        res.extend(coords[0:3])
        for i in range(3, len(coords)):
            p = res.pop()
            while (not len(res) == 0) and CGAlgorithms.computeOrientation(res[-1], p, coords[i]) > 0:
                p = res.pop()
            res.append(p)
            res.append(coords[i])
        res.append(coords[0])
        logger.debug("ConvexHull.grahamScan() %s", [str(co) for co in res])

    def lineOrPolygon(self, coords):
        """
         * @param  vertices  the vertices of a linear ring,
         *                   which may or may not be
         *                   flattened (i.e. vertices collinear)
         *
         * @return           a 2-vertex LineString if the vertices are
         *                   collinear; otherwise, a Polygon with unnecessary
         *                   (collinear) vertices removed
        """
        cleaned = []
        self.cleanRing(coords, cleaned)
        if len(cleaned) == 3:
            cleaned = cleaned[0:2]
            cs = self.toCoordinateSequence(cleaned)
            return self._factory.createLineString(cs)

        cs = self.toCoordinateSequence(cleaned)
        lr = self._factory.createLinearRing(cs)
        return self._factory.createPolygon(lr, None)

    def cleanRing(self, coords, cleaned) -> None:
        """
         * Write in 'cleaned' a version of 'input' with collinear
         * vertexes removed.
        """

        npts = len(coords)
        logger.debug("ConvexHull.cleanRing() before: %s %s", npts, [str(co) for co in coords])
        last = coords[-1]
        prev = None
        for i in range(npts - 1):

            curr = coords[i]
            next = coords[i + 1]

            # skip consecutive equal coords
            if curr == next:
                continue

            if prev is not None and self.isBetween(prev, curr, next):
                continue

            cleaned.append(curr)
            prev = curr

        cleaned.append(last)
        logger.debug("ConvexHull.cleanRing() after: %s %s", len(cleaned), [str(co) for co in cleaned])

    def isBetween(self, c1, c2, c3) -> bool:
        """
         * @return  whether the three coordinates are collinear
         *          and c2 lies between c1 and c3 inclusive
        """
        if CGAlgorithms.computeOrientation(c1, c2, c3) != 0:
            return False

        if c1.x != c3.x:
            if c1.x <= c2.x <= c3.x:
                return True
            if c3.x <= c2.x <= c1.x:
                return True

        if c1.y != c3.y:
            if c1.y <= c2.y <= c3.y:
                return True
            if c3.y <= c2.y <= c1.y:
                return True

        return False

    def getConvexHull(self):
        """
         * Returns a Geometry that represents the convex hull of
         * the input geometry.
         * The returned geometry contains the minimal number of points
         * needed to represent the convex hull.
         * In particular, no more than two consecutive points
         * will be collinear.
         *
         * @return if the convex hull contains 3 or more points,
         *         a Polygon; 2 points, a LineString;
         *         1 point, a Point; 0 points, an empty GeometryCollection.
        """
        nInputPts = len(self.inputCoords)
        logger.debug("ConvexHull.getConvexHull() %s", [str(co) for co in self.inputCoords])
        if nInputPts == 0:
            return self._factory.createEmptyGeometry()

        if nInputPts == 1:
            # Return a point
            return self._factory.createPoint(self.inputCoords[0])

        if nInputPts == 2:
            # return a LineString
            cs = self.toCoordinateSequence(self.inputCoords)
            return self._factory.createLineString(cs)

        # use heuristic to reduce points if large
        if nInputPts > 50:
            self.reduce(self.inputCoords)

        # Sort points for Graham scan
        self.preSort(self.inputCoords)

        # Use Graham scan to find convex hull
        cHs = []
        self.grahamScan(self.inputCoords, cHs)

        return self.lineOrPolygon(cHs)


class PointLocator():
    """
     * Computes the topological relationship (Location)
     * of a single point to a Geometry.
     *
     * The algorithm obeys the SFS boundaryDetermination rule to correctly determine
     * whether the point lies on the boundary or not.
     *
     * Notes:
     *  - instances of this class are not reentrant.
     *  - LinearRing objects do not enclose any area
     *    points inside the ring are still in the EXTERIOR of the ring.
    """
    def __init__(self):
        # true if the point lies in or on any Geometry element
        self._isIn = False
        # the number of sub-elements whose boundaries the point lies in
        self.numBoundaries = 0

    def locate(self, coord, geom):
        """
         * Computes the topological relationship (Location) of a single point
         * to a Geometry.
         * It handles both single-element
         * and multi-element Geometries.
         * The algorithm for multi-part Geometries
         * takes into account the boundaryDetermination rule.
         *
         * @return the Location of the point relative to the input Geometry
        """
        if geom.is_empty:
            return Location.EXTERIOR

        self._isIn = False
        self._numBoundaries = 0
        self._computeLocation(coord, geom)
        if self._numBoundaries % 2 == 1:
            return Location.BOUNDARY
        if self._numBoundaries > 0 or self._isIn:
            return Location.INTERIOR
        return Location.EXTERIOR

    def intersects(self, coord, geom):
        """
         * Convenience method to test a point for intersection with
         * a Geometry
         *
         * @param p the coordinate to test
         * @param geom the Geometry to test
         * @return true if the point is in the interiors or boundary of the Geometry
        """
        return self.locate(coord, geom) != Location.EXTERIOR

    def _locateInPoint(self, coord, geom):
        if geom.coord == coord:
            return Location.INTERIOR
        return Location.EXTERIOR

    def _locateInLineString(self, coord, geom):
        coords = geom.coords
        if not geom.isClosed:
            if coord == coords[0] or coord == coords[-1]:
                return Location.BOUNDARY
        if CGAlgorithms.isOnLine(coord, coords):
            return Location.INTERIOR
        return Location.EXTERIOR

    def _locateInLinearRing(self, coord, ring):
        coords = ring.coords
        if CGAlgorithms.isOnLine(coord, coords):
            return Location.BOUNDARY
        if CGAlgorithms.isPointInRing(coord, coords):
            return Location.INTERIOR
        return Location.EXTERIOR

    def _locateInPolygon(self, coord, geom):
        exterior = geom.exterior
        exteriorLoc = self._locateInLinearRing(coord, exterior)
        if exteriorLoc == Location.EXTERIOR:
            return Location.EXTERIOR
        if exteriorLoc == Location.BOUNDARY:
            return Location.BOUNDARY

        for hole in geom.interiors:
            holeLoc = self._locateInLinearRing(coord, hole)
            if holeLoc == Location.INTERIOR:
                return Location.EXTERIOR
            if holeLoc == Location.BOUNDARY:
                return Location.BOUNDARY
        return Location.INTERIOR

    def _computeLocation(self, coord, geom):
        type_id = geom.type_id
        
        if type_id == GeomTypeId.GEOS_POINT:
            loc = self._locateInPoint(coord, geom)
            self._updateLocationInfo(loc)

        elif type_id == GeomTypeId.GEOS_LINESTRING:
            loc = self._locateInLineString(coord, geom)
            self._updateLocationInfo(loc)

        elif type_id == GeomTypeId.GEOS_LINEARRING:
            loc = self._locateInLinearRing(coord, geom)
            self._updateLocationInfo(loc)

        elif type_id == GeomTypeId.GEOS_POLYGON:
            loc = self._locateInPolygon(coord, geom)
            self._updateLocationInfo(loc)

        elif type_id in [
                GeomTypeId.GEOS_MULTIPOINT,
                GeomTypeId.GEOS_MULTILINESTRING,
                GeomTypeId.GEOS_MULTIPOLYGON,
                GeomTypeId.GEOS_GEOMETRYCOLLECTION
                ]:
            for g in geom.geoms:
                self._computeLocation(coord, g)

    def _updateLocationInfo(self, loc):
        if loc == Location.INTERIOR:
            self._isIn = True
        if loc == Location.BOUNDARY:
            self._numBoundaries += 1


class HCoordinate():
    @staticmethod
    def intersection(p1, p2, q1, q2, ret):
        px = p1.y - p2.y
        py = p2.x - p1.x
        pw = p1.x * p2.y - p2.x * p1.y
        
        qx = q1.y - q2.y
        qy = q2.x - q1.x
        qw = q1.x * q2.y - q2.x * q1.y

        x = py * qw - qy * pw
        y = qx * pw - px * qw
        w = px * qy - qx * py

        xInt = x / w
        yInt = y / w

        if not isfinite(xInt) or not isfinite(yInt):
            raise ArithmeticError()

        ret.x, ret.y = xInt, yInt


class RobustDeterminant():
    """
     * RobustDeterminant implements an algorithm to compute the
     * sign of a 2x2 determinant for double precision values robustly.
     * It is a direct translation of code developed by Olivier Devillers.
     *
     * The original code carries the following copyright notice:
     *
     * Author : Olivier Devillers
     * Olivier.Devillers@sophia.inria.fr
     * http://www-sop.inria.fr/prisme/logiciel/determinant.html
     *
     * Olivier Devillers has allowed the code to be distributed under
     * the LGPL (2012-02-16) saying "It is ok for LGPL distribution."
    """

    @staticmethod
    def signOfDet2x2(x1: float, y1: float, x2: float, y2: float) -> int:
        """
            returns -1 if the determinant is negative,
            returns  1 if the determinant is positive,
            retunrs  0 if the determinant is null.
        """
        sign = 1

        # testing null entries
        if x1 == 0.0 or y2 == 0.0:
            if y1 == 0.0 or x2 == 0.0:
                return 0
            elif y1 > 0:
                if x2 > 0:
                    return -sign
                else:
                    return sign
            else:
                if x2 > 0:
                    return sign
                else:
                    return -sign

        if y1 == 0.0 or x2 == 0.0:
            if y2 > 0:
                if x1 > 0:
                    return sign
                else:
                    return -sign
            else:
                if x1 > 0:
                    return -sign
                else:
                    return sign

        #  making y coordinates positive and permuting the entries
        #  so that y2 is the biggest one
        if 0.0 < y1:
            if 0.0 < y2:
                if y1 > y2:
                    sign = -sign
                    x1, x2 = x2, x1
                    y1, y2 = y2, y1
            else:
                if y1 <= -y2:
                    sign = -sign
                    x2 = -x2
                    y2 = -y2
                else:
                    x1, x2 = -x2, x1
                    y1, y2 = -y2, y1
        else:
            if 0.0 < y2:
                if -y1 <= y2:
                    sign = -sign
                    x1 = -x1
                    y1 = -y1
                else:
                    x1, x2 = x2, -x1
                    y1, y2 = y2, -y1
            else:
                if y1 >= y2:
                    x1, x2 = -x1, -x2
                    y1, y2 = -y1, -y2
                else:
                    sign = -sign
                    x1, x2 = -x2, -x1
                    y1, y2 = -y2, -y1

        #  making x coordinates positive
        #  if |x2|<|x1| one can conclude

        if 0.0 < x1:
            if 0.0 < x2:
                if x1 > x2:
                    return sign
            else:
                return sign

        else:
            if 0.0 < x2:
                return -sign
            else:
                if x1 >= x2:
                    sign = -sign
                    x1 = -x1
                    x2 = -x2
                else:
                    return -sign

        #  all entries strictly positive   x1 <= x2 and y1 <= y2
        while (True):
            k = floor(x2 / x1)
            x2 = x2 - k * x1
            y2 = y2 - k * y1

            #  testing if R (new U2) is in U1 rectangle

            if y2 < 0.0:
                return -sign

            if y2 > y1:
                return sign

            #  finding R'
            if x1 > x2 + x2:
                if y1 < y2 + y2:
                    return sign
            else:
                if y1 > y2 + y2:
                    return -sign
                else:
                    x2 = x1 - x2
                    y2 = y1 - y2
                    sign = -sign

            if y2 == 0.0:
                if x2 == 0.0:
                    return 0
                else:
                    return -sign

            if x2 == 0.0:
                return sign

            #  exchange 1 and 2 role.
            k = floor(x1 / x2)
            x1 = x1 - k * x2
            y1 = y1 - k * y2

            #  testing if R (new U1) is in U2 rectangle
            if y1 < 0.0:
                return sign

            if y1 > y2:
                return -sign

            #  finding R'
            if x2 > x1 + x1:
                if y2 < y1 + y1:
                    return -sign
            else:
                if y2 > y1 + y1:
                    return sign
                else:
                    x1 = x2 - x1
                    y1 = y2 - y1
                    sign = -sign

            if y1 == 0.0:
                if x1 == 0.0:
                    return 0
                else:
                    return sign

            if x1 == 0.0:
                return -sign


class RayCrossingCounter():
    """
     * Counts the number of segments crossed by a horizontal ray extending to the right
     * from a given point, in an incremental fashion.
     *
     * This can be used to determine whether a point lies in a {Polygonal} geometry.
     * The class determines the situation where the point lies exactly on a segment.
     * When being used for Point-In-Polygon determination, this case allows short-circuiting
     * the evaluation.
     *
     * This class handles polygonal geometries with any number of exteriors and interiors.
     * The orientation of the exterior and hole rings is unimportant.
     * In order to compute a correct location for a given polygonal geometry,
     * it is essential that <b>all</b> segments are counted which
     * <ul>
     * <li>touch the ray
     * <li>lie in in any ring which may contain the point
     * </ul>
     * The only exception is when the point-on-segment situation is detected, in which
     * case no further processing is required.
     * The implication of the above rule is that segments
     * which can be a priori determined to <i>not</i> touch the ray
     * (i.e. by a test of their bounding box or Y-extent)
     * do not need to be counted.  This allows for optimization by indexing.
     *
     * @author Martin Davis
    """
    def __init__(self, point):
        self._point = point
        self._crossingCount = 0
        self._isPointOnSegment = False

    @staticmethod
    def locatePointInRing(p, ring):
        """
         * Determines the {Location} of a point in a ring.
         * This method is an exemplar of how to use this class.
         *
         * @param p the point to test
         * @param ring an array of Coordinates forming a ring
         * @return the location of the point in the ring
        """
        rcc = RayCrossingCounter(p)

        for i in range(1, len(ring)):
            p1 = ring[i - 1]
            p2 = ring[i]

            rcc.countSegment(p1, p2)

            if rcc._isPointOnSegment:
                return rcc.location

        return rcc.location

    @staticmethod
    def orientationIndex(p1, p2, q) -> int:
        """
         * Returns the index of the direction of the point q
         * relative to a vector specified by p1-p2.
         *
         * @param p1 the origin point of the vector
         * @param p2 the final point of the vector
         * @param q the point to compute the direction to
         *
         * @return 1 if q is counter-clockwise (left) from p1-p2
         * @return -1 if q is clockwise (right) from p1-p2
         * @return 0 if q is collinear with p1-p2
        """
        dx1 = p2.x - p1.x
        dy1 = p2.y - p1.y
        dx2 = q.x - p2.x
        dy2 = q.y - p2.y
        return RobustDeterminant.signOfDet2x2(dx1, dy1, dx2, dy2)

    def countSegment(self, p1, p2):
        """
         * Counts a segment
         *
         * @param p1 an endpoint of the segment
         * @param p2 another endpoint of the segment
        """
        # For each segment, check if it crosses
        # a horizontal ray running from the test point in
        # the positive x direction.
        point = self._point

        # check if the segment is strictly to the left of the test point
        if p1.x < point.x and p2.x < point.x:
            return

        # check if the point is equal to the current ring vertex
        if p2 == point:
            self._isPointOnSegment = True
            return

        # For horizontal segments, check if the point is on the segment.
        # Otherwise, horizontal segments are not counted.
        if p1.y == point.y and p2.y == point.y:
        
            if p1.x > p2.x:
                minx, maxx = p2.x, p1.x
            else:
                minx, maxx = p1.x, p2.x
            
            if maxx >= point.x >= minx:
                self._isPointOnSegment = True

            return

        # Evaluate all non-horizontal segments which cross a horizontal ray
        # to the right of the test pt.
        # To avoid double-counting shared vertices, we use the convention that
        # - an upward edge includes its starting endpoint, and excludes its
        #   final endpoint
        # - a downward edge excludes its starting endpoint, and includes its
        #   final endpoint
        if (p1.y > point.y and p2.y <= point.y) or (p2.y > point.y and p1.y <= point.y):

            # For an upward edge, orientationIndex will be positive when p1->p2
            # crosses ray. Conversely, downward edges should have negative sign.
            sign = RayCrossingCounter.orientationIndex(p1, p2, point)

            if sign == 0:
                self._isPointOnSegment = True
                return

            if p2.y < p1.y:
                sign = -sign

            # The segment crosses the ray if the sign is strictly positive.
            if sign > 0:
                self._crossingCount += 1

    @property
    def location(self):
        """
         * Gets the {Location} of the point relative to
         * the ring, polygon
         * or multipolygon from which the processed segments were provided.
         * <p>
         * This method only determines the correct location
         * if <b>all</b> relevant segments have been processed.
         *
         * @return the Location of the point
        """
        if self._isPointOnSegment:
            return Location.BOUNDARY

        # The point is in the interiors of the ring if the number
        # of X-crossings is odd.
        if (self._crossingCount % 2) == 1:
            return Location.INTERIOR

        return Location.EXTERIOR

    def isPointInPolygon(self):
        """
         * Tests whether the point lies in or on
         * the ring, polygon
         * or multipolygon from which the processed segments were provided.
         * <p>
         * This method only determines the correct location
         * if <b>all</b> relevant segments must have been processed.
         *
         * @return true if the point lies in or on the supplied polygon
        """
        return self.location != Location.EXTERIOR


class CGAlgorithms():

    # shared
    CLOCKWISE = -1
    COLLINEAR = 0
    COUNTERCLOCKWISE = 1

    RIGHT = -1
    LEFT = 0
    STRAIGHT = 1

    @staticmethod
    def isOnLine(p, coords):
        if len(coords) == 0:
            return False
            
        p0 = coords[0]
        for i in range(1, len(coords)):
            p1 = coords[i]
            if LineIntersector._hasIntersection(p, p0, p1):
                return True
            p0 = p1
        return False

    @staticmethod
    def orientationIndex(p1, p2, q):
        """
         * Returns the index of the direction of the point q
         * relative to a vector specified by p1-p2.
         *
         * @param p1 the origin point of the vector
         * @param p2 the final point of the vector
         * @param q the point to compute the direction to
         *
         * @return 1 if q is counter-clockwise (left) from p1-p2
         * @return -1 if q is clockwise (right) from p1-p2
         * @return 0 if q is collinear with p1-p2
        """
        return RayCrossingCounter.orientationIndex(p1, p2, q)

    @staticmethod
    def computeOrientation(p1, p2, q):
        """
         * Computes the orientation of a point q to the directed line
         * segment p1-p2.
         *
         * The orientation of a point relative to a directed line
         * segment indicates which way you turn to get to q after
         * travelling from p1 to p2.
         *
         * @return 1 if q is counter-clockwise from p1-p2
         * @return -1 if q is clockwise from p1-p2
         * @return 0 if q is collinear with p1-p2
        """
        return RayCrossingCounter.orientationIndex(p1, p2, q)

    @staticmethod
    def isCCW(ring):
        """
         * Computes whether a ring defined by an array of Coordinate is
         * oriented counter-clockwise.
         *
         *  - The list of points is assumed to have the first and last
         *    points equal.
         *  - This will handle coordinate lists which contain repeated points.
         *
         * This algorithm is <b>only</b> guaranteed to work with valid rings.
         * If the ring is invalid (e.g. self-crosses or touches),
         * the computed result <b>may</b> not be correct.
         *
         * @param ring an array of coordinates forming a ring
         * @return true if the ring is oriented counter-clockwise.
        """
        # of points without closing endpoint
        nCoords = len(ring) - 1
        if nCoords < 3:
            raise Exception("Ring has fewer than 3 points, so orientation cannot be determined")

        # find highest point
        # Coordinate
        hiPt = ring[0]
        hiIndex = 0
        for i in range(1, nCoords):
            p = ring[i]
            if p.y > hiPt.y:
                hiPt = p
                hiIndex = i

        # find distinct point before highest point
        iPrev = hiIndex - 1
        while ring[iPrev] == hiPt and iPrev != hiIndex:
            iPrev = iPrev - 1
            if iPrev < 0:
                iPrev = nCoords

        # find distinct point after highest point
        iNext = (hiIndex + 1) % nCoords
        while ring[iNext] == hiPt and iNext != hiIndex:
            iNext = (iNext + 1) % nCoords

        # Coordinate
        prev = ring[iPrev]
        next = ring[iNext]
        """
          * This check catches cases where the ring contains an A-B-A
          * configuration of points.
          * This can happen if the ring does not contain 3 distinct points
          * (including the case where the input array has fewer than 4 elements),
          * or it contains coincident line segments.
        """
        if prev is hiPt or next is hiPt or prev is next:
            return False

        disc = CGAlgorithms.orientationIndex(prev, hiPt, next)
        """
         *  If disc is exactly 0, lines are collinear.
         * There are two possible cases:
         *  (1) the lines lie along the x axis in opposite directions
         *  (2) the lines lie on top of one another
         *
         *  (1) is handled by checking if next is left of prev ==> CCW
         *  (2) should never happen, so we're going to ignore it!
         *  (Might want to assert this)
        """
        isCCW = False

        if disc == 0:
            # poly is CCW if prev x is right of next x
            isCCW = prev.x > next.x
        else:
            # if area is positive, points are ordered CCW
            isCCW = disc > 0

        return isCCW

    @staticmethod
    def isPointInRing(p, ring):
        """
         * Tests whether a point lies inside a ring.
         *
         * The ring may be oriented in either direction.
         * A point lying exactly on the ring boundary is considered
         * to be inside the ring.
         *
         * This algorithm does not first check the
         * point against the envelope of the ring.
         *
         * @param p point to check for ring inclusion
         * @param ring is assumed to have first point identical to last point
         * @return true if p is inside ring
         *
         * @see locatePointInRing
        """
        return RayCrossingCounter.locatePointInRing(p, ring) != Location.EXTERIOR

    @staticmethod
    def locatePointInRing(p, ring):
        """
         * Determines whether a point lies in the interiors,
         * on the boundary, or in the exterior of a ring.
         *
         * The ring may be oriented in either direction.
         *
         * This method does <i>not</i> first check the point against
         * the envelope of the ring.
         *
         * @param p point to check for ring inclusion
         * @param ring an array of coordinates representing the ring
         *        (which must have first point identical to last point)
         * @return the {Location} of p relative to the ring
        """
        return RayCrossingCounter.locatePointInRing(p, ring)

    @staticmethod
    def signedArea(ring):

        npts = len(ring)

        if npts < 3:
            return 0.0
        cx, cy = ring[0].x, ring[0].y
        nx, ny = ring[1].x, ring[1].y
        x0 = cx
        nx -= x0
        sum = 0.0
        for i in range(1, npts):
            py, cx, cy = cy, nx, ny
            nx, ny = ring[i].x, ring[i].y
            nx -= x0
            sum += cx * (ny - py)
        return -sum / 2.0

    @staticmethod
    def length(coords):

        if len(coords) < 2:
            return 0.0

        length = 0.0
        p0 = coords[0]

        x0, y0 = p0.x, p0.y

        for i in range(1, len(coords)):
            p = coords[i]
            x1, y1 = p.x, p.y
            dx, dy = x1 - x0, y1 - y0
            length += sqrt(dx * dx + dy * dy)
            x0, y0 = x1, y1

        return length

    @staticmethod
    def distancePointPoint(p0, p1):
        dx = p1.x - p0.x
        dy = p1.y - p0.y
        return sqrt(dx * dx + dy * dy)

    @staticmethod
    def distancePointLine(p, A, B):

        # if start==end, then use pt distance
        if A == B:
            return CGAlgorithms.distancePointPoint(p, A)
        """
            otherwise use comp.graphics.algorithms Frequently Asked Questions method
            (1)            AC dot AB
                       r = ---------
                           ||AB||^2
            r has the following meaning:
            r=0 P = A
            r=1 P = B
            r<0 P is on the backward extension of AB
            r>1 P is on the forward extension of AB
            0<r<1 P is interiors to AB
        """
        bax = B.x - A.x
        bay = B.y - A.y
        pax = p.x - A.x
        pay = p.y - A.y
        d = (bax * bax + bay * bay)
        r = (pax * bax + pay * bay) / d
        if r <= 0.0:
            return CGAlgorithms.distancePointPoint(p, A)
        if r >= 1.0:
            return CGAlgorithms.distancePointPoint(p, B)
        """
            (2)
                 (Ay-Cy)(Bx-Ax)-(Ax-Cx)(By-Ay)
            s = -----------------------------
                            L^2

            Then the distance from C to P = |s|*L.
        """
        s = (pax * bay - pay * bax) / d
        return abs(s) * sqrt(d)

    @staticmethod
    def distanceLineLine(A, B, C, D):
        # check for zero-length segments
        if A == B:
            return CGAlgorithms.distancePointLine(A, C, D)
        if C == D:
            return CGAlgorithms.distancePointLine(D, A, B)

        # AB and CD are line segments
        """
            from comp.graphics.algo

            Solving the above for r and s yields
                        (Ay-Cy)(Dx-Cx)-(Ax-Cx)(Dy-Cy)
                       r = ----------------------------- (eqn 1)
                        (Bx-Ax)(Dy-Cy)-(By-Ay)(Dx-Cx)

                    (Ay-Cy)(Bx-Ax)-(Ax-Cx)(By-Ay)
                s = ----------------------------- (eqn 2)
                    (Bx-Ax)(Dy-Cy)-(By-Ay)(Dx-Cx)
            Let P be the position vector of the intersection point, then
                P=A+r(B-A) or
                Px=Ax+r(Bx-Ax)
                Py=Ay+r(By-Ay)
            By examining the values of r & s, you can also determine some other
            limiting conditions:
                If 0<=r<=1 & 0<=s<=1, intersection exists
                r<0 or r>1 or s<0 or s>1 line segments do not intersect
                If the denominator in eqn 1 is zero, AB & CD are parallel
                If the numerator in eqn 1 is also zero, AB & CD are collinear.
        """
        r_top = (A.y - C.y) * (D.x - C.x) - (A.x - C.x) * (D.y - C.y)
        r_bot = (B.x - A.x) * (D.y - C.y) - (B.y - A.y) * (D.x - C.x)
        s_top = (A.y - C.y) * (B.x - A.x) - (A.x - C.x) * (B.y - A.y)
        s_bot = (B.x - A.x) * (D.y - C.y) - (B.y - A.y) * (D.x - C.x)
        if r_bot == 0 or s_bot == 0:
            return min([CGAlgorithms.distancePointLine(A, C, D),
                CGAlgorithms.distancePointLine(B, C, D),
                CGAlgorithms.distancePointLine(C, A, B),
                CGAlgorithms.distancePointLine(D, A, B)])

        s = s_top / s_bot
        r = r_top / r_bot
        if r < 0 or r > 1 or s < 0 or s > 1:
            # no intersection
            return min([CGAlgorithms.distancePointLine(A, C, D),
                CGAlgorithms.distancePointLine(B, C, D),
                CGAlgorithms.distancePointLine(C, A, B),
                CGAlgorithms.distancePointLine(D, A, B)])
        else:
            # intersection exists
            return 0.0


class Mod2BoundaryNodeRule():
    """
     * A {BoundaryNodeRule} specifies that points are in the
     * boundary of a lineal geometry iff
     * the point lies on the boundary of an odd number
     * of components.
     * Under this rule {LinearRing}s and closed
     * {LineString}s have an empty boundary.
     * <p>
     * This is the rule specified by the <i>OGC SFS</i>,
     * and is the default rule used in JTS.
     *
     * @author Martin Davis
     * @version 1.7
    """
    def isInBoundary(self, boundaryCount):
        return boundaryCount % 2 == 1


class EndPointBoundaryNodeRule():
    """
     * A {@link BoundaryNodeRule} which specifies that any points
     * which are endpoints
     * of lineal components are in the boundary of the
     * parent geometry.
     * This corresponds to the "intuitive" topological definition
     * of boundary.
     * Under this rule {@link LinearRing}s have a non-empty boundary
     * (the common endpoint of the underlying LineString).
     * <p>
     * This rule is useful when dealing with linear networks.
     * For example, it can be used to check
     * whether linear networks are correctly noded.
     * The usual network topology constraint is that linear segments may
     * touch only at endpoints.
     * In the case of a segment touching a closed segment (ring) at one
     * point,
     * the Mod2 rule cannot distinguish between the permitted case of
     * touching at the
     * node point and the invalid case of touching at some other interiors
     * (non-node) point.
     * The EndPoint rule does distinguish between these cases,
     * so is more appropriate for use.
     *
     * @author Martin Davis
     * @version 1.7
    """
    def isInBoundary(self, boundaryCount):
        return boundaryCount > 0


class BoundaryNodeRule():

    mod2Rule = Mod2BoundaryNodeRule()
    endPointRule = EndPointBoundaryNodeRule()

    @staticmethod
    def getBoundaryEndPoint():
        return BoundaryNodeRule.endPointRule

    @staticmethod
    def getBoundaryRuleMod2():
        return BoundaryNodeRule.mod2Rule

    @staticmethod
    def getBoundaryOGCSFS():
        return BoundaryNodeRule.getBoundaryRuleMod2()


class PointInRing():
    """
    """


class MCPointInRing(PointInRing):
    """
    """
    def __init__(self, newRing):
        self._ring = newRing
        self._interval = Interval()
        self.coords = None
        self._tree = None
        self._crossings = 0

    def isInside(self, coord):
        """
        """
        self._crossings = 0
        # test all segments intersected by ray from pt in positive x direction
        rayEnv = Envelope(-1e64, 1e64, coord.y, coord.y)
        self._interval.mini = coord.y
        self._interval.maxi = coord.y
        segs = self._tree.query(self._interval)
        mcSelecter = MCSelecter(coord, self)
        for mc in segs:
            # MonotoneChain
            self.testMonotoneChain(rayEnv, mcSelecter, mc)

        return (self._crossings % 2) == 1

    def testLineSegment(self, coord, seg):
        """
        """
        p1 = seg.p0
        p2 = seg.p1
        x1 = p1.x - coord.x
        y1 = p1.y - coord.y
        x2 = p2.x - coord.x
        y2 = p2.y - coord.y

        if (y1 > 0 and y2 <= 0) or (y2 > 0 and y1 <= 0):
            # segment straddles x axis, so compute intersection.
            xInt = RobustDeterminant.signOfDet2x2(x1, y1, x2, y2) / (y2 - y1)
            # crosses ray if strictly positive intersection.
            if 0.0 < xInt:
                self._crossings += 1

    def buildIndex(self):
        self._tree = Bintree()
        coords = CoordinateSequence.removeRepeatedPoints(self._ring.coords)
        mcList = MonotoneChainBuilder.getChains(coords)
        for mc in mcList:
            mcEnv = mc.envelope
            self._interval.mini = mcEnv.miny
            self._interval.maxi = mcEnv.maxy
            self._tree.insert(self._interval, mc)

    def testMonotoneChain(self, rayEnv, mcSelecter, mc):
        """
        """
        mc.select(rayEnv, mcSelecter)


class LineIntersector():
    """
     * A LineIntersector is an algorithm that can both test whether
     * two line segments intersect and compute the intersection point
     * if they do.
     *
     * The intersection point may be computed in a precise or non-precise manner.
     * Computing it precisely involves rounding it to an integer.  (This assumes
     * that the input coordinates have been made precise by scaling them to
     * an integer grid.)
    """
    # Indicates that line segments do not intersect
    NO_INTERSECTION = 0
    # Indicates that line segments intersect in a single point
    POINT_INTERSECTION = 1
    # Indicates that line segments intersect in a line segment
    COLLINEAR_INTERSECTION = 2

    def __init__(self, initialPrecisionModel=None):
        # PrecisionModel
        self._precisionModel = initialPrecisionModel
        self.intersections = 0
        # Coordinate
        self._inputLines = []
        self._isProper = False
        # Coordinate
        self.intersectionPts = [None, None]
        # int The indexes of the endpoints of the intersection lines, in order along
        # the corresponding line
        self._intLineIndex = [[0, 0], [0, 0]]

    @property
    def isInteriorIntersection(self):
        """
         * Tests whether either intersection point is an interiors point of
         * one of the input segments.
         *
         * @return true if either intersection point is in
         * the interiors of one of the input segments
        """
        if self._isInteriorIntersection(0):
            return True
        if self._isInteriorIntersection(1):
            return True
        return False

    def _isInteriorIntersection(self, inputLineIndex):
        """
         * Tests whether either intersection point is an interiors point
         * of the specified input segment.
         *
         * @return true if either intersection point is in
         * the interiors of the input segment
        """
        for i in range(self.intersections):
            if not ((self.intersectionPts[i] == self._inputLines[inputLineIndex][0]) or
                    (self.intersectionPts[i] == self._inputLines[inputLineIndex][1])):
                return True
        return False

    def nearestEndpoint(self, p1, p2, q1, q2):
        """
         * Finds the endpoint of the segments P and Q which
         * is closest to the other segment.
         * This is a reasonable surrogate for the true
         * intersection points in ill-conditioned cases
         * (e.g. where two segments are nearly coincident,
         * or where the endpoint of one segment lies almost on the other segment).
         * <p>
         * This replaces the older CentralEndpoint heuristic,
         * which chose the wrong endpoint in some cases
         * where the segments had very distinct slopes
         * and one endpoint lay almost on the other segment.
         *
         * @param p1 an endpoint of segment P
         * @param p2 an endpoint of segment P
         * @param q1 an endpoint of segment Q
         * @param q2 an endpoint of segment Q
         * @return the nearest endpoint to the other segment
        """
        nearestPt = p1
        minDist = CGAlgorithms.distancePointLine(p1, q1, q2)

        dist = CGAlgorithms.distancePointLine(p2, q1, q2)
        if dist < minDist:
            minDist = dist
            nearestPt = p2

        dist = CGAlgorithms.distancePointLine(q1, p1, p2)
        if dist < minDist:
            minDist = dist
            nearestPt = q1

        dist = CGAlgorithms.distancePointLine(q2, p1, p2)
        if dist < minDist:
            nearestPt = q2

        return nearestPt

    @staticmethod
    def computeEdgeDistance(p, p0, p1):
        """
         * Computes the "edge distance" of an intersection point p in an edge.
         *
         * The edge distance is a metric of the point along the edge.
         * The metric used is a robust and easy to compute metric function.
         * It is <b>not</b> equivalent to the usual Euclidean metric.
         * It relies on the fact that either the x or the y ordinates of the
         * points in the edge are unique, depending on whether the edge is longer in
         * the horizontal or vertical direction.
         *
         * NOTE: This function may produce incorrect distances
         *  for inputs where p is not precisely on p1-p2
         * (E.g. p = (139,9) p1 = (139,10), p2 = (280,1) produces distanct
         * 0.0, which is incorrect.
         *
         * My hypothesis is that the function is safe to use for points which are the
         * result of <b>rounding</b> points which lie on the line,
         * but not safe to use for <b>truncated</b> points.
        """
        dx = abs(p1.x - p0.x)
        dy = abs(p1.y - p0.y)
        # sentinel value
        dist = -1.0

        if p == p0:
            dist = 0.0

        elif p == p1:
            if dx > dy:
                dist = dx
            else:
                dist = dy

        else:
            pdx = abs(p.x - p0.x)
            pdy = abs(p.y - p0.y)
            if dx > dy:
                dist = pdx
            else:
                dist = pdy

            # hack to ensure that non-endpoints always have a non-zero distance
            if dist == 0.0 and not (p == p0):
                dist = max(pdx, pdy)

        assert(dist > 0.0 or p == p0), "Bad distance calculation"

        return dist

    def computeIntersection(self, p, p1, p2):
        """
         * Compute the intersection of a point p and the line p1-p2.
         *
         * This function computes the boolean value of the hasIntersection test.
         * The actual value of the intersection (if there is one)
         * is equal to the value of p.
        """
        self._isProper = False
        if Envelope.static_intersects(p1, p2, p):
            if (CGAlgorithms.orientationIndex(p1, p2, p) == 0 and
                    CGAlgorithms.orientationIndex(p2, p1, p) == 0):
                self._isProper = True

                if (p == p1) or (p == p2):
                    self._isProper = False

                self.intersectionPts[0] = p

                self.intersections = LineIntersector.POINT_INTERSECTION
                return
        self.intersections = LineIntersector.NO_INTERSECTION

    @staticmethod
    def _hasIntersection(p, p1, p2):
        # Same as above but doent's compute intersection point. Faster.
        if Envelope.static_intersects(p1, p2, p):
            if (CGAlgorithms.orientationIndex(p1, p2, p) == 0 and
                    CGAlgorithms.orientationIndex(p2, p1, p) == 0):
                return True
        return False

    def computeLinesIntersection(self, p1, p2, q1, q2):
        # Computes the intersection of the lines p1-p2 and p3-p4
        self._inputLines = [[p1, p2], [q1, q2]]
        # logger.debug("computeLinesIntersection %s", self)
        if USE_FUZZY_LINE_INTERSECTOR:
            self.intersections = self._fuzzy_computeIntersect(p1, p2, q1, q2)
        else:
            self.intersections = self._computeIntersect(p1, p2, q1, q2)

    def _fuzzy_computeIntersect(self, p1, p2, q1, q2):
        """ point_sur_segment return
            p: point d'intersection
            u: param t de l'intersection sur le segment courant
            v: param t de l'intersection sur le segment segment
            d: perpendicular distance of segment.p
        """
        self._isProper = False

        if Envelope.static_intersects(p1, p2, q1, q2):

            # vector seg 1
            vx = p2.x - p1.x
            vy = p2.y - p1.y

            # cross seg 2
            qx = q2.y - q1.y
            qy = q1.x - q2.x

            # dot product
            d = qx * vx + qy * vy

            # vector delta p1 q1
            dx = q1.x - p1.x
            dy = q1.y - p1.y

            # logger.debug("d:%s", d)

            # almost parallel, check for real proximity of points
            if d != 0 and abs(d) < 0.001:
                dx2 = q2.x - p2.x
                dy2 = q2.y - p2.y
                l = sqrt(vx * vx + vy * vy)
                dist0 = abs(vx * dy - vy * dx) / l
                if dist0 < EPSILON_SINGLE:
                    dist1 = abs(vy * dx2 - vx * dy2) / l
                    if dist1 < EPSILON_SINGLE:
                        vqx = q2.x - q1.x
                        vqy = q2.y - q1.y
                        l = sqrt(vqx * vqx + vqy * vqy)
                        dist2 = abs(vqy * dx - vqx * dy) / l
                        dist3 = abs(vqx * dy2 - vqy * dx2) / l
                        logger.debug("Almost parallel d0:%s d1:%s d2:%s d3:%s", dist0, dist1, dist2, dist3)
                        if (dist2 < EPSILON_SINGLE and
                                dist3 < EPSILON_SINGLE):
                            d = 0

            if d == 0:
                return self._computeCollinearIntersection(p1, p2, q1, q2)
                """
                # check for distance between segments
                l = sqrt(vx * vx + vy * vy)
                d = abs(vx * dy - vy * dx) / l
                if d < EPSILON_SINGLE:
                    # logger.debug("Parallel")

                    if vx > vy:
                        if p1.x > p2.x:
                            p1, p2 = p2, p1
                        if q1.x > q2.x:
                            q1, q2 = q2, q1
                        pq1 = p1.x <= q1.x <= p2.x
                        pq2 = p1.x <= q2.x <= p2.x
                        qp1 = q1.x <= p1.x <= q2.x
                        qp2 = q1.x <= p2.x <= q2.x

                    else:
                        if p1.y > p2.y:
                            p1, p2 = p2, p1
                        if q1.y > q2.y:
                            q1, q2 = q2, q1
                        pq1 = p1.y <= q1.y <= p2.y
                        pq2 = p1.y <= q2.y <= p2.y
                        qp1 = q1.y <= p1.y <= q2.y
                        qp2 = q1.y <= p2.y <= q2.y

                    if pq1 and pq2:
                        # q1 and q2 between p1 and p2
                        self.intersectionPts[0] = q1
                        self.intersectionPts[1] = q2
                        logger.debug("COLLINEAR_INTERSECTION q1:%s q2:%s", q1, q2)
                        return LineIntersector.COLLINEAR_INTERSECTION

                    if qp1 and qp2:
                        self.intersectionPts[0] = p1
                        self.intersectionPts[1] = p2
                        logger.debug("COLLINEAR_INTERSECTION p1:%s p2:%s", p1, p2)
                        return LineIntersector.COLLINEAR_INTERSECTION

                    if pq1 and qp1:
                        self.intersectionPts[0] = q1
                        self.intersectionPts[1] = p1
                        if (q1 == p1) and not pq2 and not qp2:
                            logger.debug("POINT_INTERSECTION q1 == p1:%s", q1)
                            return LineIntersector.POINT_INTERSECTION
                        else:
                            logger.debug("COLLINEAR_INTERSECTION q1:%s p1:%s", q1, p1)
                            return LineIntersector.COLLINEAR_INTERSECTION

                    if pq1 and qp2:
                        self.intersectionPts[0] = q1
                        self.intersectionPts[1] = p2
                        if (q1 == p2) and not pq2 and not qp1:
                            logger.debug("POINT_INTERSECTION q1 == p2:%s", q1)
                            return LineIntersector.POINT_INTERSECTION
                        else:
                            logger.debug("COLLINEAR_INTERSECTION q1:%s p2:%s", q1, p2)
                            return LineIntersector.COLLINEAR_INTERSECTION

                    if pq2 and qp1:
                        self.intersectionPts[0] = q2
                        self.intersectionPts[1] = p1
                        if (q2 == p1) and not pq1 and not qp2:
                            logger.debug("POINT_INTERSECTION q2 == p1:%s", q2)
                            return LineIntersector.POINT_INTERSECTION
                        else:
                            logger.debug("COLLINEAR_INTERSECTION q2:%s p1:%s", q2, p1)
                            return LineIntersector.COLLINEAR_INTERSECTION

                    if pq2 and qp2:
                        self.intersectionPts[0] = q2
                        self.intersectionPts[1] = p2
                        if (q2 == p2) and not pq1 and not qp1:
                            logger.debug("POINT_INTERSECTION q2 == p2:%s", q2)
                            return LineIntersector.POINT_INTERSECTION
                        else:
                            logger.debug("COLLINEAR_INTERSECTION q2:%s p2:%s", q2, p2)
                            return LineIntersector.COLLINEAR_INTERSECTION
                else:
                    return LineIntersector.NO_INTERSECTION
                """
                
            # logger.debug("Point intersection")
            # cross seg 1
            py = p1.x - p2.x
            px = p2.y - p1.y

            u = (qx * dx + qy * dy) / d
            v = (px * dx + py * dy) / d
            if 0 <= u <= 1 and 0 <= v <= 1:

                # check for distance between end points and intersection
                pt = p1 + Coordinate(vx, vy) * u
                
                if u < 0.5:
                    # near segment 1 start point
                    # distance p1 intersection
                    if  p1.distance(pt) < EPSILON:
                        self.u = 0
                        self.intersectionPts[0] = p1
                        logger.debug("POINT_INTERSECTION p1:%s", p1)
                        return LineIntersector.POINT_INTERSECTION

                elif u > 0.5:
                    # near segment 1 end point
                    if p2.distance(pt) < EPSILON:
                        self.u = 1
                        self.intersectionPts[0] = p2
                        logger.debug("POINT_INTERSECTION p2:%s", p2)

                        return LineIntersector.POINT_INTERSECTION

                if v < 0.5:
                    # near segment 1 start point
                    # distance p1 intersection
                    if q1.distance(pt) < EPSILON:
                        self.v = 0
                        self.intersectionPts[0] = q1
                        logger.debug("POINT_INTERSECTION q1:%s", q1)
                        return LineIntersector.POINT_INTERSECTION

                elif v > 0.5:
                    # near segment 1 end point
                    if q2.distance(pt) < EPSILON:
                        self.v = 1
                        self.intersectionPts[0] = q2
                        logger.debug("POINT_INTERSECTION q2:%s", q2)
                        return LineIntersector.POINT_INTERSECTION

                self._isProper = True
                logger.debug("POINT_INTERSECTION pt:%s", pt)
                self.intersectionPts[0] = pt

                return LineIntersector.POINT_INTERSECTION

        logger.debug("NO_INTERSECTION")
        return LineIntersector.NO_INTERSECTION

    @property
    def hasIntersection(self):
        return self.intersections != LineIntersector.NO_INTERSECTION

    @property
    def isProper(self):
        """
         * Tests whether an intersection is proper.
         *
         * The intersection between two line segments is considered proper if
         * they intersect in a single point in the interiors of both segments
         * (e.g. the intersection is a single point and is not equal to any of the
         * endpoints).
         *
         * The intersection between a point and a line segment is considered proper
         * if the point lies in the interiors of the segment (e.g. is not equal to
         * either of the endpoints).
         *
         * @return true if the intersection is proper
        """
        return self.hasIntersection and self._isProper

    @property
    def isCollinear(self):
        return self.intersections == LineIntersector.COLLINEAR_INTERSECTION

    @property
    def isEndpoint(self):
        return self.hasIntersection and not self._isProper

    def getIntersection(self, intIndex):
        return self.intersectionPts[intIndex]

    @staticmethod
    def isSameSignAndNonZero(a, b):
        if a == 0 or b == 0:
            return False
        return (a < 0 and b < 0) or (a > 0 and b > 0)

    def computeIntLineIndex(self):
        self._computeIntLineIndex(0)
        self._computeIntLineIndex(1)

    def isIntersection(self, pt):
        """
         * Test whether a point is a intersection point of two line segments.
         *
         * Note that if the intersection is a line segment, this method only tests for
         * equality with the endpoints of the intersection segment.
         * It does <b>not</b> return true if
         * the input point is internal to the intersection segment.
         *
         * @return true if the input point is one of the intersection points.
        """
        for i in range(self.intersections):
            if self.intersectionPts[i] == pt:
                return True
        return False

    def getIntersectionAlongSegment(self, segmentIndex, intIndex):
        """
         * Computes the intIndex'th intersection point in the direction of
         * a specified input line segment
         *
         * @param segmentIndex is 0 or 1
         * @param intIndex is 0 or 1
         *
         * @return the intIndex'th intersection point in the direction of the
         *         specified input line segment
        """
        self.computeIntLineIndex()
        return self.intersectionPts[self._intLineIndex[segmentIndex][intIndex]]

    def getIndexAlongSegment(self, segmentIndex, intIndex):
        """
         * Computes the index of the intIndex'th intersection point in the direction of
         * a specified input line segment
         *
         * @param segmentIndex is 0 or 1
         * @param intIndex is 0 or 1
         *
         * @return the index of the intersection point along the segment (0 or 1)
        """
        self.computeIntLineIndex()
        return self._intLineIndex[segmentIndex][intIndex]

    def getEdgeDistance(self, segmentIndex, intIndex):
        """
         * Computes the "edge distance" of an intersection point along the specified
         * input line segment.
         *
         * @param segmentIndex is 0 or 1
         * @param intIndex is 0 or 1
         *
         * @return the edge distance of the intersection point
        """
        # logger.debug("getEdgeDistance %s", self)
        return self.computeEdgeDistance(
            self.intersectionPts[intIndex],
            self._inputLines[segmentIndex][0],
            self._inputLines[segmentIndex][1]
            )

    def _intersectionWithNormalization(self, p1, p2, q1, q2, ret):
        # Make new Coordinates
        n1 = Coordinate(p1.x, p1.y)
        n2 = Coordinate(p2.x, p2.y)
        n3 = Coordinate(q1.x, q1.y)
        n4 = Coordinate(q2.x, q2.y)
        normPt = Coordinate(0, 0)
        self._normalizeToEnvCentre(n1, n2, n3, n4, normPt)
        self._safeHCoordinateIntersection(n1, n2, n3, n4, ret)
        ret.x += normPt.x
        ret.y += normPt.y

    def _computeIntLineIndex(self, segmentIndex):
        dist0 = self.getEdgeDistance(segmentIndex, 0)
        dist1 = self.getEdgeDistance(segmentIndex, 1)
        if dist0 > dist1:
            self._intLineIndex[segmentIndex][0] = 0
            self._intLineIndex[segmentIndex][1] = 1
        else:
            self._intLineIndex[segmentIndex][0] = 1
            self._intLineIndex[segmentIndex][1] = 0

    def _computeIntersect(self, p1, p2, q1, q2):

        self._isProper = False
        # logger.debug("LineIntersector.computeIntersect(p1:%s, p2:%s, q1:%s, q2:%s)", p1, p2, q1, q2)
        # logger.debug("LineIntersector.computeIntersect(%s)", self)
        if not Envelope.static_intersects(p1, p2, q1, q2):
            logger.debug("NO_INTERSECTION env p1:%s p2:%s q1:%s q2:%s", p1, p2, q1, q2)
            return LineIntersector.NO_INTERSECTION

        # for each endpoint, compute which side of the other segment it lies
        # if both endpoints lie on the same side of the other segment,
        # the segments do not intersect

        pq1 = CGAlgorithms.orientationIndex(p1, p2, q1)
        pq2 = CGAlgorithms.orientationIndex(p1, p2, q2)

        if (pq1 > 0 and pq2 > 0) or (pq1 < 0 and pq2 < 0):
            logger.debug("NO_INTERSECTION p side p1:%s p2:%s q1:%s q2:%s", p1, p2, q1, q2)
            return LineIntersector.NO_INTERSECTION

        qp1 = CGAlgorithms.orientationIndex(q1, q2, p1)
        qp2 = CGAlgorithms.orientationIndex(q1, q2, p2)

        if (qp1 > 0 and qp2 > 0) or (qp1 < 0 and qp2 < 0):
            logger.debug("NO_INTERSECTION q side p1:%s p2:%s q1:%s q2:%s", p1, p2, q1, q2)
            return LineIntersector.NO_INTERSECTION

        collinear = pq1 == 0 and pq2 == 0 and qp1 == 0 and qp2 == 0
        if collinear:
            return self._computeCollinearIntersection(p1, p2, q1, q2)

        """
         * At this point we know that there is a single intersection point
         * (since the lines are not collinear).

         * Check if the intersection is an endpoint.
         * If it is, copy the endpoint as
         * the intersection point. Copying the point rather than
         * computing it ensures the point has the exact value,
         * which is important for robustness. It is sufficient to
         * simply check for an endpoint which is on the other line,
         * since at this point we know that the inputLines must
         *  intersect.
        """
        if pq1 == 0 or pq2 == 0 or qp1 == 0 or qp2 == 0:
            """
             * Check for two equal endpoints.
             * This is done explicitly rather than by the orientation tests
             * below in order to improve robustness.
             *
             * (A example where the orientation tests fail
             *  to be consistent is:
             *
             * LINESTRING ( 19.850257749638203 46.29709338043669,
             *          20.31970698357233 46.76654261437082 )
             * and
             * LINESTRING ( -48.51001596420236 -22.063180333403878,
             *          19.850257749638203 46.29709338043669 )
             *
             * which used to produce the result:
             * (20.31970698357233, 46.76654261437082, NaN)
            """
            if p1 == q1 or p1 == q2:
                self.intersectionPts[0] = p1
                logger.debug("POINT_INTERSECTION p1:%s", p1)
            elif p2 == q1 or p2 == q2:
                self.intersectionPts[0] = p2
                logger.debug("POINT_INTERSECTION p2:%s", p2)
            elif pq1 == 0:
                # Now check to see if any endpoint lies on the interiors of the other segment
                self.intersectionPts[0] = q1
                logger.debug("POINT_INTERSECTION q1:%s", q1)
            elif pq2 == 0:
                self.intersectionPts[0] = q2
                logger.debug("POINT_INTERSECTION q2:%s", q2)
            elif qp1 == 0:
                self.intersectionPts[0] = p1
                logger.debug("POINT_INTERSECTION p1:%s", p1)
            elif qp2 == 0:
                self.intersectionPts[0] = p2
                logger.debug("POINT_INTERSECTION p2:%s", p2)
                        
        else:
            self._isProper = True
            self.intersectionPts[0] = Coordinate(0, 0)
            self._intersection(p1, p2, q1, q2, self.intersectionPts[0])

            logger.debug("POINT_INTERSECTION pt:%s", self.intersectionPts[0])

        return LineIntersector.POINT_INTERSECTION

    def _computeCollinearIntersection(self, p1, p2, q1, q2):

        # logger.debug("LineIntersector._computeCollinearIntersection()")

        pq1 = Envelope.static_intersects(p1, p2, q1)
        pq2 = Envelope.static_intersects(p1, p2, q2)
        qp1 = Envelope.static_intersects(q1, q2, p1)
        qp2 = Envelope.static_intersects(q1, q2, p2)

        if pq1 and pq2:
            # q1 and q2 between p1 and p2
            self.intersectionPts[0] = q1.clone()
            self.intersectionPts[1] = q2.clone()
            logger.debug("COLLINEAR_INTERSECTION q1:%s q2:%s between p1:%s p2:%s", q1, q2, p1, p2)
            return LineIntersector.COLLINEAR_INTERSECTION

        if qp1 and qp2:
            self.intersectionPts[0] = p1.clone()
            self.intersectionPts[1] = p2.clone()
            logger.debug("COLLINEAR_INTERSECTION p1:%s p2:%s between q1:%s q2:%s", p1, p2, q1, q2)
            return LineIntersector.COLLINEAR_INTERSECTION

        if pq1 and qp1:
            self.intersectionPts[0] = q1.clone()
            self.intersectionPts[1] = p1.clone()
            if (q1 == p1) and not pq2 and not qp2:
                logger.debug("POINT_INTERSECTION q1 == p1:%s", q1)
                return LineIntersector.POINT_INTERSECTION
            else:
                logger.debug("COLLINEAR_INTERSECTION q1:%s p1:%s", q1, p1)
                return LineIntersector.COLLINEAR_INTERSECTION

        if pq1 and qp2:
            self.intersectionPts[0] = q1.clone()
            self.intersectionPts[1] = p2.clone()
            if (q1 == p2) and not pq2 and not qp1:
                logger.debug("POINT_INTERSECTION q1 == p2:%s", q1)
                return LineIntersector.POINT_INTERSECTION
            else:
                logger.debug("COLLINEAR_INTERSECTION q1:%s p2:%s", q1, p2)
                return LineIntersector.COLLINEAR_INTERSECTION

        if pq2 and qp1:
            self.intersectionPts[0] = q2.clone()
            self.intersectionPts[1] = p1.clone()
            if (q2 == p1) and not pq1 and not qp2:
                logger.debug("POINT_INTERSECTION q2 == p1:%s", q2)
                return LineIntersector.POINT_INTERSECTION
            else:
                logger.debug("COLLINEAR_INTERSECTION q2:%s p1:%s", q2, p1)
                return LineIntersector.COLLINEAR_INTERSECTION

        if pq2 and qp2:
            self.intersectionPts[0] = q2.clone()
            self.intersectionPts[1] = p2.clone()
            if (q2 == p2) and not pq1 and not qp1:
                logger.debug("POINT_INTERSECTION q2 == p2:%s", q2)
                return LineIntersector.POINT_INTERSECTION
            else:
                logger.debug("COLLINEAR_INTERSECTION q2:%s p2:%s", q2, p2)
                return LineIntersector.COLLINEAR_INTERSECTION

        logger.debug("NO_INTERSECTION collinear p1:%s p2:%s q1:%s q2:%s", p1, p2, q1, q2)
        return LineIntersector.NO_INTERSECTION

    def _intersection(self, p1, p2, q1, q2, intPtOut):
        """
         * This method computes the actual value of the intersection point.
         *
         * To obtain the maximum precision from the intersection calculation,
         * the coordinates are normalized by subtracting the minimum
         * ordinate values (in absolute value).  This has the effect of
         * removing common significant digits from the calculation to
         * maintain more bits of precision.
        """
        self._intersectionWithNormalization(p1, p2, q1, q2, intPtOut)
        """
         * Due to rounding it can happen that the computed intersection is
         * outside the envelopes of the input segments.  Clearly this
         * is inconsistent.
         * This code checks this condition and forces a more reasonable answer
         *
         * MD - May 4 2005 - This is still a problem.  Here is a failure case:
         *
         * LINESTRING (2089426.5233462777 1180182.3877339689,
         *             2085646.6891757075 1195618.7333999649)
         * LINESTRING (1889281.8148903656 1997547.0560044837,
         *             2259977.3672235999 483675.17050843034)
         * int point = (2097408.2633752143,1144595.8008114607)
        """
        if not self._isInSegmentEnvelopes(intPtOut):
            intPtOut = self.nearestEndpoint(p1, p2, q1, q2)

    def _smallestInAbsValue(self, x1, x2, x3, x4):
        """
        """
        x = x1
        xabs = abs(x)
        if abs(x2) < xabs:
            x = x2
            xabs = abs(x2)
        if abs(x3) < xabs:
            x = x3
            xabs = abs(x3)
        if abs(x4) < xabs:
            x = x4
        return x

    def _isInSegmentEnvelopes(self, intPt):
        """
         * Test whether a point lies in the envelopes of both input segments.
         * A correctly computed intersection point should return true
         * for this test.
         * Since this test is for debugging purposes only, no attempt is
         * made to optimize the envelope test.
         *
         * @return true if the input point lies within both
         *         input segment envelopes
        """
        p0, p1 = self._inputLines[0]
        env0 = Envelope(p0, p1)
        p0, p1 = self._inputLines[0]
        env1 = Envelope(p0, p1)
        return env0.contains(intPt) and env1.contains(intPt)

    def _normalizeToEnvCentre(self, p1, p2, q1, q2, normPt):
        """
         * Normalize the supplied coordinates to
         * so that the midpoint of their intersection envelope
         * lies at the origin.
         *
         * @param n00
         * @param n01
         * @param n10
         * @param n11
         * @param normPt
        """
        minx0, maxx0 = p1.x, p2.x
        if minx0 > maxx0:
            minx0, maxx0 = maxx0, minx0

        miny0, maxy0 = p1.y, p2.y
        if miny0 > maxy0:
            miny0, maxy0 = maxy0, miny0

        minx1, maxx1 = q1.x, q2.x
        if minx1 > maxx1:
            minx1, maxx1 = maxx1, minx1

        miny1, maxy1 = q1.y, q2.y
        if miny1 > maxy1:
            miny1, maxy1 = maxy1, miny1

        if minx0 > minx1:
            minx = minx0
        else:
            minx = minx1

        if miny0 > miny1:
            miny = miny0
        else:
            miny = miny1

        if maxx0 < maxx1:
            maxx = maxx0
        else:
            maxx = maxx1

        if maxy0 < maxy1:
            maxy = maxy0
        else:
            maxy = maxy1

        midx = (minx + maxx) / 2.0
        midy = (miny + maxy) / 2.0

        normPt.x = midx
        normPt.y = midy

        p1.x -= midx
        p1.y -= midy
        p2.x -= midx
        p2.y -= midy
        q1.x -= midx
        q1.y -= midy
        q2.x -= midx
        q2.y -= midy

    def _safeHCoordinateIntersection(self, p1, p2, q1, q2, intPt):
        """
         * Computes a segment intersection using homogeneous coordinates.
         * Round-off error can cause the raw computation to fail,
         * (usually due to the segments being approximately parallel).
         * If this happens, a reasonable approximation is computed instead.
         *
         * @param p1 a segment endpoint
         * @param p2 a segment endpoint
         * @param q1 a segment endpoint
         * @param q2 a segment endpoint
         * @param intPt the computed intersection point is stored there
        """
        try:
            HCoordinate.intersection(p1, p2, q1, q2, intPt)
        except:
            coord = self.nearestEndpoint(p1, p2, q1, q2)
            intPt.x = coord.x
            intPt.y = coord.y
            pass

    def __str__(self):
        return "p1:{}_p2:{} q1:{}_q2:{} : isEndpoint:{} isProper:{} isCollinear:{}".format(
            self._inputLines[0][0],
            self._inputLines[0][1],
            self._inputLines[1][0],
            self._inputLines[1][1],
            self.isEndpoint,
            self.isProper,
            self.isCollinear
            )


# algorithms/locate


class IntervalIndexedGeometry():
    def __init__(self, geom):

        # index.intervalrtree.SortedPackedIntervalRTree
        self.index = SortedPackedIntervalRTree()
        self.init(geom)

    def init(self, geom) -> None:
        # LineString
        lines = []
        LinearComponentExtracter.getLines(geom, lines)
        for line in lines:
            self.addLine(line.coords)

    def addLine(self, coords) -> None:
        for i in range(1, len(coords)):
            seg = LineSegment(coords[i - 1], coords[i])
            min = seg.p0.y
            max = seg.p1.y
            if min > max:
                max, min = min, max
            self.index.insert(min, max, seg)

    def query(self, min: float, max: float, visitor) -> None:
        self.index.query(min, max, visitor)


class SegmentVisitor(ItemVisitor):
    def __init__(self, counter):
        ItemVisitor.__init__(self)
        # algorithm.RayCrossingCounter
        self.counter = counter

    def visitItem(self, item) -> None:
        self.counter.countSegment(item.p0, item.p1)


class IndexedPointInAreaLocator():
    """
     * Determines the location of {@link Coordinate}s relative to
     * a {@link Polygon} or {@link MultiPolygon} geometry, using indexing for efficiency.
     *
     * This algorithm is suitable for use in cases where
     * many points will be tested against a given area.
     *
     * @author Martin Davis
     *
    """
    def __init__(self, geom):
        """
         * Creates a new locator for a given {@link Geometry}
         * @param g the Geometry to locate in
        """
        # Geometry
        self.geom = geom
        if geom.type_id not in [
                GeomTypeId.GEOS_POLYGON,
                GeomTypeId.GEOS_MULTIPOLYGON
                ]:
            raise ValueError("Argument must be Polygonal")

        # IntervalIndexedGeometry
        self.index = None
        self.buildIndex(geom)

    def buildIndex(self, geom) -> None:
        self.index = IntervalIndexedGeometry(geom)

    def locate(self, coord) -> int:
        """
         * Determines the {@link Location} of a point in an areal {@link Geometry}.
         *
         * @param p the point to test
         * @return the location of the point in the geometry
        """
        rcc = RayCrossingCounter(coord)
        visitor = SegmentVisitor(rcc)
        self.index.query(coord.y, coord.y, visitor)
        return rcc.location


class SimplePointInAreaLocator():
    @staticmethod
    def locate(p, geom):
        if geom.is_empty:
            return Location.EXTERIOR
        if SimplePointInAreaLocator.containsPoint(p, geom):
            return Location.INTERIOR
        return Location.EXTERIOR

    @staticmethod
    def containsPoint(p, geom):
        type_id = geom.type_id
        if type_id == GeomTypeId.GEOS_POLYGON:
            return SimplePointInAreaLocator.containsPointInPolygon(p, geom)

        elif (type_id == GeomTypeId.GEOS_GEOMETRYCOLLECTION or
                type_id == GeomTypeId.GEOS_MULTIPOLYGON):
            for g in geom.geoms:
                if SimplePointInAreaLocator.containsPoint(p, g):
                    return True
        return False

    @staticmethod
    def containsPointInPolygon(p, geom):
        if geom.is_empty:
            return False

        exterior = geom.exterior
        coords = exterior.coords

        if not CGAlgorithms.isPointInRing(p, coords):
            return False

        for hole in geom.interiors:
            coords = hole.coords
            if CGAlgorithms.isPointInRing(p, coords):
                return False

        return True


# geomgraph/index


class EdgeSetIntersector():
    """
     * Interface
    """
    def __init__(self):
        pass

    def computeSelfIntersections(self, edges, si, testAllSegments):
        """
         * Computes all self-intersections between edges in a set of edges,
         * allowing client to choose whether self-intersections are computed.
         *
         * @param edges a list of edges to test for intersections
         * @param si the SegmentIntersector to use
         * @param testAllSegments true if self-intersections are to be tested as well
        """
        raise NotImplementedError()

    def computeIntersections(self, edges0, edges1, si):
        """
         * Computes all mutual intersections between two sets of edges
        """
        raise NotImplementedError()


class SegmentIntersector():
    """
    """
    def __init__(self, newLi, newIncludeProper, newRecordIsolated):
        """
         * These variables keep track of what types of intersections were
         * found during ALL edges that have been intersected.
        """
        self.hasIntersection = False
        self.hasProper = False
        self.hasProperInterior = False
        self.isDone = False
        self.isDoneWhenProperInt = False

        # the proper intersection point found
        # Coordinate
        self.properIntersectionPoint = None

        # LineIntersector
        self._li = newLi

        # bool
        self._includeProper = newIncludeProper
        self._recordIsolated = newRecordIsolated

        # int
        self.numIntersections = 0

        # Elements are externally owned
        # Node
        self._bdyNodes = [None, None]

        # testing only
        self.numTests = 0

    def setBoundaryNodes(self, bdyNodes0, bdyNodes1):
        self._bdyNodes[0] = bdyNodes0
        self._bdyNodes[1] = bdyNodes1

    def addIntersections(self, e0, segIndex0, e1, segIndex1):
        """
         * This method is called by clients of the EdgeIntersector class to test
         * for and add intersections for two segments of the edges being intersected.
         * Note that clients (such as MonotoneChainEdges) may choose not to intersect
         * certain pairs of segments for efficiency reasons.
         * @param e0: Edge 1 first edge
         * @param e1: Edge 2 other edge
         * @param segIndex0 index of first vertex of segment of Edge 1
         * @param segIndex1 index of first vertex of segment of Edge 2
        """
        if e0 == e1 and segIndex0 == segIndex1:
            return

        self.numTests += 1

        # CoordinateSequence
        c0 = e0.coords
        # Coordinate
        p1 = c0[segIndex0]
        p2 = c0[segIndex0 + 1]

        # CoordinateSequence
        c1 = e1.coords
        # Coordinate
        q1 = c1[segIndex1]
        q2 = c1[segIndex1 + 1]

        # LineIntersector
        _li = self._li
        _li.computeLinesIntersection(p1, p2, q1, q2)
        """
         * Always record any non-proper intersections.
         * If includeProper is true, record any proper intersections as well.
        """
        if _li.hasIntersection:

            if self._recordIsolated:
                e0.isIsolated = False
                e1.isIsolated = False

            self.numIntersections += 1

            # If the segments are adjacent they have at least one trivial
            # intersection, the shared endpoint.
            # Don't bother adding it if it is the
            # only intersection.
            if not self.isTrivialIntersection(e0, segIndex0, e1, segIndex1):

                self.hasIntersection = True

                if self._includeProper or not _li.isProper:

                    logger.debug("SegmentIntersector.addIntersections(): (icludeProper: %s || !li.isProper: %s)",
                        self._includeProper,
                        not _li.isProper)

                    e0.addIntersections(_li, segIndex0, 0)
                    e1.addIntersections(_li, segIndex1, 1)

                if _li.isProper:

                    self.properIntersectionPoint = _li.getIntersection(0)

                    self.hasProper = True
                    logger.debug("SegmentIntersector.addIntersections(): properIntersectionPoint: %s",
                        self.properIntersectionPoint)

                    if self.isDoneWhenProperInt:
                        self.isDone = True

                    if not self.isBoundaryPoint(_li, self._bdyNodes):
                        self.hasProperInterior = True

    def isAdjacentSegments(self, i1, i2):
        return abs(i1 - i2) == 1

    def isTrivialIntersection(self, e0, segIndex0, e1, segIndex1):
        """
         * A trivial intersection is an apparent self-intersection which in fact
         * is simply the point shared by adjacent line segments.
         * Note that closed edges require a special check for the point
         * shared by the beginning and end segments.
         * @param e0: Edge 1 first edge
         * @param e1: Edge 2 other edge
         * @param segIndex0 index of first vertex of segment of Edge 1
         * @param segIndex1 index of first vertex of segment of Edge 2
        """
        if e0 == e1:
            # only one intersection detected
            if self._li.intersections == 1:

                # is between consecutive segments
                if self.isAdjacentSegments(segIndex0, segIndex1):
                    return True

                # is between last and first segment of closed shape
                if e0.isClosed:
                    maxSegIndex = len(e0.coords) - 1
                    if ((segIndex0 == 0 and segIndex1 == maxSegIndex) or
                            (segIndex1 == 0 and segIndex0 == maxSegIndex)):
                        return True

        return False

    def _isBoundaryPoint(self, li, tstBdyNodes):

        if tstBdyNodes is None:
            return False

        for node in tstBdyNodes:
            if li.isIntersection(node.coord):
                return True

        return False

    def isBoundaryPoint(self, li, tstBdyNodes):

        if self._isBoundaryPoint(li, tstBdyNodes[0]):
            return True

        if self._isBoundaryPoint(li, tstBdyNodes[1]):
            return True

        return False


class SweepLineEvent():

    INSERT_EVENT = 1
    DELETE_EVENT = 2

    def __init__(self, edgeSet, x, insertEvent, newObj):

        self.edgeSet = edgeSet
        # SweepLineSegment
        self._obj = newObj
        self.x = x
        # SweepLineEvent INSERT_EVENT
        self._insertEvent = insertEvent
        self._deleteEventIndex = 0
        if insertEvent is None:
            self.eventType = SweepLineEvent.INSERT_EVENT
        else:
            self.eventType = SweepLineEvent.DELETE_EVENT

    def compareTo(self, other):
        """
         * ProjectionEvents are ordered first by their x-value, and then by their
         * eventType.
         * It is important that Insert events are sorted before Delete events, so that
         * items whose Insert and Delete events occur at the same x-value will be
         * correctly handled.
        """
        if self.x < other.x:
            return -1
        elif self.x > other.x:
            return 1
        elif self.eventType < other.eventType:
            return -1
        elif self.eventType > other.eventType:
            return 1
        else:
            return 0

    @property
    def isInsert(self):
        return self.eventType == SweepLineEvent.INSERT_EVENT

    @property
    def isDelete(self):
        return self.eventType == SweepLineEvent.DELETE_EVENT


class SweepLineSegment():

    def __init__(self, newEdge, newPtIndex):
        self.edge = newEdge
        self.coords = newEdge.coords
        self.ptIndex = newPtIndex

    @property
    def minx(self):
        x1 = self.coords[self.ptIndex].x
        x2 = self.coords[self.ptIndex + 1].x
        if x1 < x2:
            return x1
        else:
            return x2

    @property
    def maxx(self):
        x1 = self.coords[self.ptIndex].x
        x2 = self.coords[self.ptIndex + 1].x
        if x1 > x2:
            return x1
        else:
            return x2

    def computeIntersections(self, ss, si):
        si.addIntersections(self.edge, self.ptIndex, ss.edge, ss.ptIndex)


class MonotoneChainEdge():
    """
     * MonotoneChains are a way of partitioning the segments of an edge to
     * allow for fast searching of intersections.
     * They have the following properties:
     *
     *  -  the segments within a monotone chain will never intersect each other
     *  -  the envelope of any contiguous subset of the segments in a monotone
     *     chain is simply the envelope of the endpoints of the subset.
     *
     * Property 1 means that there is no need to test pairs of segments from
     * within the same monotone chain for intersection.
     * Property 2 allows binary search to be used to find the intersection
     * points of two monotone chains.
     * For many types of real-world data, these properties eliminate a large
     * number of segment comparisons, producing substantial speed gains.
     * @version 1.1
    """
    def __init__(self, newEdge):
        # Edge
        self._edge = newEdge
        # CoordinateSequence
        self.coords = newEdge.coords
        # int
        # the lists of start/end indexes of the monotone chains.
        # Includes the end point of the edge as a sentinel
        self._startIndex = []
        # Envelope
        # these envelopes are created once and reused
        self._env1 = Envelope()
        self._env2 = Envelope()
        mcb = MonotoneChainIndexer()
        mcb.getChainStartIndices(self.coords, self._startIndex)
        logger.debug("MonotoneChainEdge.indices %s", ", ".join([str(i) for i in self._startIndex]))

    def getMinX(self, chainIndex):
        x1 = self.coords[self._startIndex[chainIndex]].x
        x2 = self.coords[self._startIndex[chainIndex + 1]].x
        if x1 < x2:
            return x1
        else:
            return x2

    def getMaxX(self, chainIndex):
        x1 = self.coords[self._startIndex[chainIndex]].x
        x2 = self.coords[self._startIndex[chainIndex + 1]].x
        if x1 > x2:
            return x1
        else:
            return x2

    def computeIntersects(self, mce, si):
        for i in range(len(self._startIndex) - 1):
            for j in range(len(mce._startIndex) - 1):
                self.computeIntersectsForChain(i, mce, j, si)

    def computeIntersectsForChain(self, chainIndex0: int, mce, chainIndex1: int, si):
        self._computeIntersectsForChain(
            self._startIndex[chainIndex0],
            self._startIndex[chainIndex0 + 1],
            mce,
            mce._startIndex[chainIndex1],
            mce._startIndex[chainIndex1 + 1],
            si)

    def _computeIntersectsForChain(self,
            start0: int, end0: int,
            mce,
            start1: int, end1: int,
            si) -> None:

        # terminating condition for the recursion
        if end0 - start0 == 1 and end1 - start1 == 1:
            # SegmentIntersector
            logger.debug("MonotoneChainEdge._computeIntersectsForChain indexes: %s  / %s", start0, start1)
            si.addIntersections(self._edge, start0, mce._edge, start1)
            return


        p1 = self.coords[start0]
        p2 = self.coords[end0]
        q1 = mce.coords[start1]
        q2 = mce.coords[end1]
        self._env1.__init__(p1, p2)
        self._env2.__init__(q1, q2)

        if not self._env1.intersects(self._env2):
            logger.debug("MonotoneChainEdge._computeIntersectsForChain env dosent intersect")
            return

        mid0 = int((start0 + end0) / 2)
        mid1 = int((start1 + end1) / 2)
        logger.debug("MonotoneChainEdge._computeIntersectsForChain indexes: %s - %s - %s  / %s - %s - %s", start0, mid0, end0, start1, mid1, end1)

        if start0 < mid0:
            if start1 < mid1:
                self._computeIntersectsForChain(start0, mid0, mce, start1, mid1, si)
            if mid1 < end1:
                self._computeIntersectsForChain(start0, mid0, mce, mid1, end1, si)

        if mid0 < end0:
            if start1 < mid1:
                self._computeIntersectsForChain(mid0, end0, mce, start1, mid1, si)
            if mid1 < end1:
                self._computeIntersectsForChain(mid0, end0, mce, mid1, end1, si)


class SimpleMonotoneChain():
    def __init__(self, mce, chainIndex):
        self.mce = mce
        self.chainIndex = chainIndex

    def computeIntersections(self, mc, si):
        self.mce.computeIntersectsForChain(self.chainIndex, mc.mce, mc.chainIndex, si)


class SimpleMCSweepLineIntersector(EdgeSetIntersector):
    """
     * Finds all intersections in one or two sets of edges,
     * using an x-axis sweepline algorithm in conjunction with Monotone Chains.
     *
     * While still O(n^2) in the worst case, this algorithm
     * drastically improves the average-case time.
     * The use of MonotoneChains as the items in the index
     * seems to offer an improvement in performance over a sweep-line alone.
    """
    def __init__(self):
        # SweepLineEvent
        self.events = []
        self.nOverlaps = 0

    def computeSelfIntersections(self, edges, si, testAllSegments=False):
        """
         * Computes all self-intersections between edges in a set of edges,
         * allowing client to choose whether self-intersections are computed.
         *
         * @param edges a list of edges to test for intersections
         * @param si the SegmentIntersector to use
         * @param testAllSegments true if self-intersections are to be tested as well
        """
        if testAllSegments:
            self.addEdges(edges, None)
        else:
            for edge in edges:
                self._add(edge, edge)
        self._computeIntersections(si)

    def computeIntersections(self, edges0, edges1, si):
        """
         * Computes all mutual intersections between two sets of edges
        """
        self.addEdges(edges0, edges0)
        self.addEdges(edges1, edges1)
        self._computeIntersections(si)

    def _computeIntersections(self, si):
        self.nOverlaps = 0
        self._prepareEvents()
        for i, ev in enumerate(self.events):
            if ev.isInsert:
                self._processOverlaps(i, ev._deleteEventIndex, ev, si)
            if si.isDone:
                break

    def _prepareEvents(self):
        """
         * Because Delete Events have a link to their corresponding Insert event,
         * it is possible to compute exactly the range of events which must be
         * compared to a given Insert event object.
        """
        self.events = sorted(self.events, key=lambda ev: (ev.x, ev.eventType))
        for i, ev in enumerate(self.events):
            if ev.isDelete:
                ev._insertEvent._deleteEventIndex = i

    def addEdges(self, edges, edgeSet=None):
        for edge in edges:
            self._add(edge, edgeSet)
    """
    def _add(self, edge, edgeSet):
        mce = edge.coords
        for i in range(len(mce) - 1):
            mc = SweepLineSegment(edge, i)
            insertEvent = SweepLineEvent(edgeSet, mc.minx, None, mc)
            self.events.append(insertEvent)
            self.events.append(SweepLineEvent(edgeSet, mc.maxx, insertEvent, mc))
    """

    def _add(self, edge, edgeSet):
        mce = edge.monotoneChainEdge
        for i in range(len(mce._startIndex) - 1):
            mc = SimpleMonotoneChain(mce, i)
            insertEvent = SweepLineEvent(edgeSet, mce.getMinX(i), None, mc)
            self.events.append(insertEvent)
            self.events.append(SweepLineEvent(edgeSet, mce.getMaxX(i), insertEvent, mc))

    def _processOverlaps(self, start, end, ev0, si):
        # SimpleMonotoneChain
        mc0 = ev0._obj
        """
         * Since we might need to test for self-intersections,
         * include current insert event object in list of event objects to test.
         * Last index can be skipped, because it must be a Delete event.
        """
        for i in range(start, end):
            # SeeplineEvent
            ev1 = self.events[i]
            if ev1.isInsert:
                # SimpleMonotoneChain
                mc1 = ev1._obj
                if ev0.edgeSet is None or ev0.edgeSet != ev1.edgeSet:
                    mc0.computeIntersections(mc1, si)
                    self.nOverlaps += 1


# algorithm/MCPointInRing


class MCSelecter(MonotoneChainSelectAction):
    def __init__(self, newCoord, mcp):
        # Coordinate
        self.coord = newCoord
        # MCPointInRing
        self._parent = mcp

    def select(self, seg):
        self._parent.testLineSegment(self.coord, seg)
