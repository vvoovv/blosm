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


from .constants import (
    Envelope,
    Quadrant
)


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

        p0 = self.coords[start0]
        p1 = self.coords[end0]
        p2 = mc.coords[start1]
        p3 = mc.coords[end1]
        # nothing to do if the envelopes of these chains don't overlap
        mco.tempEnv1.__init__(p0, p1)
        mco.tempEnv2.__init__(p2, p3)
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
        n = len(startIndex) - 1
        for i in range(n):
            mc = MonotoneChain(coords, startIndex[i], startIndex[i + 1], context)
            mcList.append(mc)

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
        npts = len(coords)
        safeStart = start

        # skip any zero-length segments at the start of the sequence
        # (since they cannot be used to establish a quadrant)
        while (safeStart < npts - 1 and coords[safeStart] is coords[safeStart + 1]):
            safeStart += 1

        # check if there are NO non-zero-length segments
        if safeStart >= npts - 1:
            return npts - 1

        # determine overall quadrant for chain
        chainQuad = Quadrant.from_coords(coords[safeStart], coords[safeStart + 1])

        last = start + 1
        while last < npts:
            if coords[last - 1] is not coords[last]:
                quad = Quadrant.from_coords(coords[last - 1], coords[last])
                if quad != chainQuad:
                    break
            last += 1

        return last - 1


class MonotoneChainSelectAction():

    def __init__(self):
        # geom::LineSegment
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
        # geom::LineSegment
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
        mc1.getLineSegment(start1, self._overlapSeg1)
        mc2.getLineSegment(start2, self._overlapSeg2)
        self._overlap(self._overlapSeg1, self._overlapSeg2)

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
