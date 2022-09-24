# -*- coding:utf-8 -*-

# ##### BEGIN LGPL LICENSE BLOCK #####
# GEOS - Geometry Engine Open Source
# http:#geos.osgeo.org
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


from .shared import (
    logger,
    Envelope,
    GeomTypeId,
    GeometryComponentFilter,
    GeometryTransformer
    )
from .algorithms import (
    LineSegment,
    LineIntersector,
    ItemVisitor
    )
from .index_quadtree import Quadtree


class DouglasPeuckerLineSimplifier():
    """
     * Simplifies a linestring (sequence of points) using
     * the standard Douglas-Peucker algorithm.
    """
    def __init__(self, coords):
        self.coords = coords
        self.usePt = []
        self.tolerance = 0

    @staticmethod
    def simplify(coords, tolerance):
        dpls = DouglasPeuckerLineSimplifier(coords)
        dpls.tolerance = tolerance
        return dpls._simplify()

    def _simplify(self):

        nCoords = len(self.coords)
        if nCoords == 0:
            return self.coords
        self.usePt = [True for i in range(nCoords)]
        self.simplifySection(0, nCoords - 1)
        # S.L add : remove first/last point of closed curves when apply
        if self.coords[0] == self.coords[-1] and nCoords > 2:
            c = self.coords

            # find one valid points on both ends
            start = -2
            end = 1
            if self.usePt[end] and self.usePt[start]:
                seg = LineSegment(c[start], c[end])
                distance = seg.distance(c[0])
                # remove first and last point
                if distance < self.tolerance:
                    self.usePt[0] = False
                    self.usePt[-1] = False
                    newCoords = [coord for i, coord in enumerate(self.coords) if self.usePt[i]]
                    # close the ring
                    newCoords.append(newCoords[0])
                    return newCoords
        
        return [coord for i, coord in enumerate(self.coords) if self.usePt[i]]

    def simplifySection(self, i: int, j: int) -> None:

        c = self.coords

        if i + 1 == j:
            return

        seg = LineSegment(c[i], c[j])
        maxDistance = -1.0
        maxIndex = i

        for k in range(i + 1, j):
            distance = seg.distance(c[k])
            if distance > maxDistance:
                maxDistance = distance
                maxIndex = k

        if maxDistance <= self.tolerance:
            for k in range(i + 1, j):
                self.usePt[k] = False
        else:
            self.simplifySection(i, maxIndex)
            self.simplifySection(maxIndex, j)


class DPTransformer(GeometryTransformer):

    def __init__(self, tolerance: float):
        GeometryTransformer.__init__(self)
        self.tolerance = tolerance
        self.setSkipTransformedInvalidInteriorRings = True

    def transformCoordinates(self, coords, parent):
        newPts = DouglasPeuckerLineSimplifier.simplify(coords, self.tolerance)
        return self._factory.coordinateSequenceFactory.create(newPts)

    def transformPolygon(self, geom, parent):
        roughGeom = GeometryTransformer.transformPolygon(self, geom, parent)

        if parent.type_id == GeomTypeId.GEOS_MULTIPOLYGON:
            return roughGeom

        return self.createValidArea(roughGeom)

    def transformMultiPolygon(self, geom, parent):
        roughGeom = GeometryTransformer.transformMultiPolygon(self, geom, parent)
        return self.createValidArea(roughGeom)

    def createValidArea(self, roughAreaGeom):
        """
         * Creates a valid area geometry from one that possibly has
         * bad topology (i.e. self-intersections).
         * Since buffer can handle invalid topology, but always returns
         * valid geometry, constructing a 0-width buffer "corrects" the
         * topology.
         * Note this only works for area geometries, since buffer always returns
         * areas.  This also may return empty geometries, if the input
         * has no actual area.
         *
         * @param roughAreaGeom an area geometry possibly containing
         *        self-intersections
         * @return a valid area geometry
        """
        return roughAreaGeom.buffer(0)


class DouglasPeukerSimplifier():
    """
     * Simplifies a Geometry using the standard Douglas-Peucker algorithm.
     *
     * Ensures that any polygonal geometries returned are valid.
     * Simple lines are not guaranteed to remain simple after simplification.
     *
     * Note that in general D-P does not preserve topology -
     * e.g. polygons can be split, collapse to lines or disappear
     * interiors can be created or disappear,
     * and lines can cross.
     * To simplify geometry while preserving topology use TopologyPreservingSimplifier.
     * (However, using D-P is significantly faster).
    """

    def __init__(self, geom):
        self.geom = geom
        """
         * Sets the distance tolerance for the simplification.
         *
         * All vertices in the simplified geometry will be within this
         * distance of the original geometry.
         * The tolerance value must be non-negative.  A tolerance value
         * of zero is effectively a no-op.
         *
         * @param distanceTolerance the approximation tolerance to use
        """
        self.tolerance = 0

    @staticmethod
    def simplify(geom, tolerance: float):
        dps = DouglasPeukerSimplifier(geom)
        logger.debug("******************************\n")
        logger.debug("DouglasPeukerSimplifier.simplify()\n")
        logger.debug("******************************")
        dps.tolerance = tolerance
        return dps.getResultGeometry()

    def getResultGeometry(self):
        dpt = DPTransformer(self.tolerance)
        return dpt.transform(self.geom)


class LineStringTransformer(GeometryTransformer):

    def __init__(self, linestringMap):
        """
         * @param nMap - reference to LinesMap instance.
        """
        # LinesMap
        self.linestringMap = linestringMap

    def transformCoordinates(self, coords, parent):

        if parent.type_id == GeomTypeId.GEOS_LINESTRING:
            taggedLine = self.linestringMap.find(parent)
            newCoords = taggedLine.resultCoordinates
            logger.debug("LineStringTransformer.transformCoordinates(%s)", len(newCoords))
            return newCoords

        else:
            # for anything else (e.g. points) just copy the coordinates
            return GeometryTransformer.transformCoordinates(self, coords, parent)


class LineStringMapBuilderFilter(GeometryComponentFilter):
    """
     * A filter to add linear geometries to the linestring map
     * with the appropriate minimum size constraint.
     * Closed {@link LineString}s (including {@link LinearRing}s
     * have a minimum output size constraint of 4,
     * to ensure the output is valid.
     * For all other linestrings, the minimum size is 2 points.
     *
     * This class populates the given LineString=>TaggedLineString map
     * with newly created TaggedLineString objects.
     * Users must take care of deleting the map's values (elem.second).
    """
    def __init__(self, linestringMap):
        # LinesMap
        self.linestringMap = linestringMap

    def filter_ro(self, geom):
        if geom.type_id == GeomTypeId.GEOS_LINESTRING:
            if geom.isClosed:
                minSize = 4
            else:
                minSize = 2
            taggedLine = TaggedLineString(geom, minSize)
            self.linestringMap.insert(geom, taggedLine)
        else:
            return


class LineSegmentVisitor(ItemVisitor):

    def __init__(self, seg):
        ItemVisitor.__init__(self),
        self.seg = seg
        self.items = []

    def visitItem(self, seg):
        if Envelope.static_intersects(seg.p0, seg.p1, self.seg.p0, self.seg.p1):
            self.items.append(seg)


class LineSegmentIndex():
    """
    """
    def __init__(self):
        self.index = Quadtree()

    def add(self, line):
        for seg in line.segs:
            self.addSegment(seg)

    def addSegment(self, seg):
        env = Envelope(seg.p0, seg.p1)
        self.index.insert(env, seg)

    def remove(self, seg):
        env = Envelope(seg.p0, seg.p1)
        self.index.remove(env, seg)

    def query(self, seg):
        env = Envelope(seg.p0, seg.p1)
        visitor = LineSegmentVisitor(seg)
        self.index.visit(env, visitor)
        # LineSegment
        return visitor.items


class TaggedLineSegment(LineSegment):
    """
     * A geom.LineSegment which is tagged with its location in a geom.Geometry.
     *
     * Used to index the segments in a geometry and recover the segment locations
     * from the index.
    """
    def __init__(self, p0, p1=None, parent=None, index: int=0):

        if p1 is None:
            # using another TaggedLineSegment
            p0, p1, parent, index = p0.p0, p0.p1, p0.parent, p0.index

        LineSegment.__init__(self, p0, p1)
        self.parent = parent
        self.index = index


class TaggedLineString():
    """
     * Contains and owns a list of TaggedLineSegments
    """
    def __init__(self, parent, minimumSize: int=2) -> None:
        # Linestring
        self.parent = parent
        self.minimumSize = minimumSize
        # TaggedLineSegments
        self.segs = []
        self.result = []
        self.init()

    def init(self) -> None:
        coords = self.parent.coords
        if len(coords) > 0:
            for i in range(len(coords) - 1):
                seg = TaggedLineSegment(coords[i], coords[i + 1], self.parent, i)
                self.segs.append(seg)

    @property
    def resultCoordinates(self):
        coords = self.extractCoordinates(self.result)
        return self.parent._factory.coordinateSequenceFactory.create(coords)

    def asLineString(self):
        return self.parent._factory.createLineString(self.resultCoordinates)

    def asLinearRing(self):
        return self.parent._factory.createLinearRing(self.resultCoordinates)

    @property
    def resultSize(self) -> int:
        res = len(self.result)
        if res > 0:
            res += 1
        return res

    def addToResult(self, seg):
        self.result.append(seg)

    def extractCoordinates(self, segs):
        coords = [seg.p0 for seg in segs]
        coords.append(segs[-1].p1)
        return coords


class TaggedLineStringSimplifier():
    """
     * Simplifies a TaggedLineString, preserving topology
     * (in the sense that no new intersections are introduced).
     * Uses the recursive Douglas-Peucker algorithm.
    """
    def __init__(self, inputIndex, outputIndex) -> None:
        self.inputIndex = inputIndex
        self.outputIndex = outputIndex
        self.li = LineIntersector()
        # TaggedLineString
        self.line = None
        self.coords = None
        self.tolerance = 0

    def simplify(self, line) -> None:
        """
         * Simplifies the given {@link TaggedLineString}
         * using the distance tolerance specified.
         *
         * @param line the linestring to simplify
        """
        self.line = line
        self.coords = line.parent.coords
        if len(self.coords) == 0:
            logger.warning("TaggedLineStringSimplifier.simplify parent.coords == 0")
            return
        self.simplifySection(0, len(self.coords) - 1, 0)
        logger.debug("TaggedLineStringSimplifier.simplify segs:%s result:%s", len(self.line.segs), self.line.resultSize)
            
    def simplifySection(self, i: int, j: int, depth: int) -> None:
        depth += 1
        sectionIndex = [0, 0]
        if i + 1 == j:
            self.line.addToResult(self.line.segs[i])
            # leave this segment in the input index, for efficiency
            return

        isValidToSimplify = True
        """
         * Following logic ensures that there is enough points in the
         * output line.
         * If there is already more points than the minimum, there's
         * nothing to check.
         * Otherwise, if in the worst case there wouldn't be enough points,
         * don't flatten this segment (which avoids the worst case scenario)
        """
        if self.line.resultSize < self.line.minimumSize:
            worstCaseSize = depth + 1
            if worstCaseSize < self.line.minimumSize:
                isValidToSimplify = False

        furthestPtIndex, distance = self.findFurthestPoint(self.coords, i, j)

        # flattening must be less than distanceTolerance
        if distance > self.tolerance:
            isValidToSimplify = False

        candidateSeg = LineSegment(self.coords[i], self.coords[j])
        sectionIndex[0] = i
        sectionIndex[1] = j

        if self.hasBadIntersection(self.line, sectionIndex, candidateSeg):
            isValidToSimplify = False

        if isValidToSimplify:
            # TaggedLineSegment
            newSeg = self.flatten(i, j)
            self.line.addToResult(newSeg)
            return

        self.simplifySection(i, furthestPtIndex, depth)
        self.simplifySection(furthestPtIndex, j, depth)

    def findFurthestPoint(self, coords, i: int, j: int):
        seg = LineSegment(coords[i], coords[j])
        maxDist = -1.0
        maxIndex = i
        for k in range(i + 1, j):
            midPt = coords[k]
            distance = seg.distance(midPt)
            if distance > maxDist:
                maxDist = distance
                maxIndex = k

        return maxIndex, maxDist

    def hasBadIntersection(self, parentLine, sectionIndex: list, candidateSeg) -> bool:

        if self.hasBadOutputIntersection(candidateSeg):
            return True

        if self.hasBadInputIntersection(parentLine, sectionIndex, candidateSeg):
            return True

        return False

    def hasBadInputIntersection(self, parentLine, sectionIndex: list, candidateSeg) -> bool:
        querySegs = self.inputIndex.query(candidateSeg)
        for seg in querySegs:
            if self.hasInteriorIntersection(seg, candidateSeg):
                if self.isInLineSection(parentLine, sectionIndex, seg):
                    continue
                return True
        return False

    def hasBadOutputIntersection(self, candidateSeg) -> bool:
        querySegs = self.outputIndex.query(candidateSeg)
        for seg in querySegs:
            if self.hasInteriorIntersection(seg, candidateSeg):
                return True
        return False

    def hasInteriorIntersection(self, seg0, seg1) -> bool:
        self.li.computeLinesIntersection(seg0.p0, seg0.p1, seg1.p0, seg1.p1)
        return self.li.isInteriorIntersection

    def flatten(self, start: int, end: int):
        p0 = self.coords[start]
        p1 = self.coords[end]
        newSeg = TaggedLineSegment(p0, p1)
        self.remove(self.line, start, end)
        self.outputIndex.addSegment(newSeg)
        return newSeg

    def isInLineSection(self, parentLine, sectionIndex: list, seg) -> bool:
        """
         * Tests whether a segment is in a section of a TaggedLineString
         *
         * @param line
         * @param sectionIndex
         * @param seg
         * @return
        """
        if seg.parent is not self.line.parent:
            return False

        segIndex = seg.index
        if segIndex >= sectionIndex[0] and segIndex < sectionIndex[1]:
            return True

        return False

    def remove(self, line, start: int, end: int) -> None:
        """
         * Remove the segs in the section of the line
         *
         * @param line
         * @param pts
         * @param sectionStartIndex
         * @param sectionEndIndex
        """
        for i in range(start, end):
            seg = line.segs[i]
            self.inputIndex.remove(seg)


class TaggedLinesSimplifier():
    
    def __init__(self):

        # LineSegmentIndex
        self.inputIndex = LineSegmentIndex()
        self.outputIndex = LineSegmentIndex()
        self.taggedlineSimplifier = TaggedLineStringSimplifier(self.inputIndex, self.outputIndex)

    @property
    def tolerance(self):
        return self.taggedlineSimplifier.tolerance

    @tolerance.setter
    def tolerance(self, tolerance):
        self.taggedlineSimplifier.tolerance = tolerance

    def simplify(self, linestrings, start, end) -> None:
        """
         * Simplify a set of {@link TaggedLineString}s
         * @param linestrings set of TaggedLineString(s)
         * @param start: start index
         * @param end: end index
        """
        for i in range(start, end):
            self.inputIndex.add(linestrings[i])

        for i in range(start, end):
            self.taggedlineSimplifier.simplify(linestrings[i])


class LinesMap(dict):

    def __init__(self):
        dict.__init__(self)

    def insert(self, geom, taggedLine):
        tl = self.find(id(geom))
        if tl is None:
            self[id(geom)] = taggedLine
            
    def find(self, geom):
        return self.get(id(geom))


class TopologyPreservingSimplifier():
    """
     * Simplifies a geometry, ensuring that
     * the result is a valid geometry having the
     * same dimension and number of components as the input.
     *
     * The simplification uses a maximum distance difference algorithm
     * similar to the one used in the Douglas-Peucker algorithm.
     *
     * In particular, if the input is an areal geometry
     * ( Polygon or MultiPolygon )
     *
     *  -  The result has the same number of exteriors and interiors (rings) as the input,
     *     in the same order
     *  -  The result rings touch at <b>no more</b> than the number of touching point in the input
     *     (although they may touch at fewer points)
     *
    """
    def __init__(self, geom):
        self.geom = geom
        """
         * Sets the distance tolerance for the simplification.
         *
         * All vertices in the simplified geometry will be within this
         * distance of the original geometry.
         * The tolerance value must be non-negative.  A tolerance value
         * of zero is effectively a no-op.
         *
         * @param distanceTolerance the approximation tolerance to use
        """
        self.tolerance = 0
        self.lineSimplifier = TaggedLinesSimplifier()

    @staticmethod
    def simplify(geom, tolerance):
        tps = TopologyPreservingSimplifier(geom)
        tps.lineSimplifier.tolerance = tolerance
        return tps.getResultGeometry()

    def getResultGeometry(self):

        if self.geom.is_empty:
            return self.geom.clone()

        linestringMap = LinesMap()

        lsmbf = LineStringMapBuilderFilter(linestringMap)
        self.geom.apply_ro(lsmbf)

        linestrings = list(linestringMap.values())
        logger.debug("TopologyPreservingSimplifier.getResultGeometry linestrings:%s", len(linestrings))
        self.lineSimplifier.simplify(linestrings, 0, len(linestrings))

        trans = LineStringTransformer(linestringMap)
        return trans.transform(self.geom)
