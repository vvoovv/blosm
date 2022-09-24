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


from .algorithms import (
    PointLocator,
    LineIntersector,
    SimplePointInAreaLocator,
    IndexedPointInAreaLocator
    )
from .shared import (
    logger,
    Envelope,
    LinearComponentExtracter,
    ComponentCoordinateExtracter,
    ShortCircuitedGeometryVisitor,
    GeomTypeId,
    Location
    )
from .noding import (
    SegmentStringUtil,
    SegmentIntersectionDetector,
    FastSegmentSetIntersectionFinder
    )


# operation/predicate


class SegmentIntersectionTester():
    def __init__(self):
        self.li = LineIntersector()
        self.hasIntersection = False
    
    def hasIntersectionWithLineStrings(self, line, lines) -> bool:
        self.hasIntersection = False
        for testline in lines:
            self.findIntersection(line, testline)
            if self.hasIntersection:
                break
        return self.hasIntersection
            
    def findIntersection(self, line, testLine):
        seq0 = line.coords
        seq1 = testLine.coords
        for i in range(1, len(seq0)):
            p0 = seq0[i - 1]
            p1 = seq0[i]
            for j in range(1, len(seq1)):
                q0 = seq1[j - 1]
                q1 = seq1[j]    
                self.li.computeLinesIntersection(p0, p1, q0, q1)
                if self.li.hasIntersection:
                    self.hasIntersection = True
                    return self.hasIntersection            
        return self.hasIntersection
            
    def hasIntersectionWithEnvelopeFilter(self, line, testLine):
        """
         * Tests the segments of a LineString against the segs in
         * another LineString for intersection.
         * Uses the envelope of the query LineString
         * to filter before testing segments directly.
         * This is optimized for the case when the query
         * LineString is a rectangle.
         *
         * Testing shows this is somewhat faster than not checking the envelope.
         *
         * @param line
         * @param testLine
         * @return
        """
        seq0 = line.coords
        seq1 = testLine.coords
        env = line.envelope
        for i in range(1, len(seq0)):
            p0 = seq0[i - 1]
            p1 = seq0[i]
            if not env.intersects(Envelope(p0, p1)):
                continue
            for j in range(1, len(seq1)):
                q0 = seq1[j - 1]
                q1 = seq1[j]    
                self.li.computeLinesIntersection(p0, p1, q0, q1)
                if self.li.hasIntersection:
                    self.hasIntersection = True
                    return self.hasIntersection            
        return self.hasIntersection
        

class EnvelopeIntersectsVisitor(ShortCircuitedGeometryVisitor):

    def __init__(self, envelope):
        ShortCircuitedGeometryVisitor.__init__(self)
        self.env = envelope
        self.intersects = False
        
    def visit(self, element) -> None:
        """*
         * Reports whether it can be concluded that an intersection occurs,
         * or whether further testing is required.
         *
         * @return <code>true</code> if an intersection must occur
         * <code>false</code> if no conclusion can be made
        """
        env = element.envelope
        
        # skip if envelopes do not intersect
        if not self.env.intersects(env):
            return

        # fully contained - must intersect
        if self.env.contains(env):
            self.intersects = True
            return
        

        """
         * Since the envelopes intersect and the test element is
         * connected, if the test envelope is completely bisected by
         * an edge of the rectangle the element and the rectangle
         * must touch (This is basically an application of the
         * Jordan Curve Theorem).  The alternative situation
         * is that the test envelope is "on a corner" of the
         * rectangle envelope, i.e. is not completely bisected.
         * In this case it is not possible to make a conclusion
         * about the presence of an intersection.
        """
        if env.minx >= self.env.minx and env.maxx <= self.env.maxx:
            self.intersects = True
            return
        
        if env.miny >= self.env.miny and env.maxy <= self.env.maxy:
            self.intersects = True
            return
        
    def isDone(self) -> bool:
        return self.intersects
        

class ContainsPointVisitor(ShortCircuitedGeometryVisitor):
    """*
     * Tests whether it can be concluded
     * that a geometry contains a corner point of a rectangle.
    """
    def __init__(self, geom):
        ShortCircuitedGeometryVisitor.__init__(self)
        self.env = geom.envelope
        self.coords = geom.exterior.coords
        self.contains = False
    
    def visit(self, geom) -> None:
    
        # if test geometry is not polygonal this check is not needed
        if not geom.type_id == GeomTypeId.GEOS_POLYGON: 
            return
        
        env = geom.envelope
        
        if not self.env.intersects(env):
            return
        
        # test each corner of rectangle for inclusion
        for coord in self.coords:
            
            if not env.contains(coord):
                continue

            # check rect point in poly (rect is known not to
            # touch polygon at this point)
            if SimplePointInAreaLocator.containsPointInPolygon(coord, geom):
            
                self.contains = True
                return
            
    def isDone(self) -> bool:
        return self.contains


class LineIntersectsVisitor(ShortCircuitedGeometryVisitor):

    def __init__(self, geom):
        ShortCircuitedGeometryVisitor.__init__(self)
        self.env = geom.envelope
        self.line = geom.exterior
        self.intersects = False
  
    def computeSegmentIntersection(self, geom):
    
        # check segment intersection
        # get all lines from geom (e.g. if it's a multi-ring polygon)
        lines = []
        LinearComponentExtracter.getLines(geom, lines)
         
        si = SegmentIntersectionTester()
        if si.hasIntersectionWithLineStrings(self.line, lines):
            self.intersects = True
            return
    
    def visit(self, geom) -> None:

        env = geom.envelope

        # check for envelope intersection
        if not self.env.intersects(env):
            return

        self.computeSegmentIntersection(geom)
    
    def isDone(self) -> bool:
        return self.intersects
    

class RectangleIntersects():
    def __init__(self, geom):
        self.rectangle = geom
        self.env = geom.envelope
    
    @staticmethod
    def intersects(rect, geom) -> bool:
        rp = RectangleIntersects(rect)
        return rp._intersects(geom)
    
    def _intersects(self, geom) -> bool:
    
        if not self.env.intersects(geom.envelope):
            return False

        # test envelope relationships
        visitor = EnvelopeIntersectsVisitor(self.env)
        visitor.applyTo(geom)
        if visitor.intersects:
            return True

        # test if any rectangle corner is contained in the target
        ecpVisitor = ContainsPointVisitor(self.rectangle)
        ecpVisitor.applyTo(geom)
        if ecpVisitor.contains:
            return True

        # test if any lines intersect
        liVisitor = LineIntersectsVisitor(self.rectangle)
        liVisitor.applyTo(geom);
        if liVisitor.intersects:
            return True

        return False
    

class RectangleContains():
    """
     * Optimized implementation of spatial predicate "contains"
     * for cases where the first Geometry is a rectangle.
     *
     * As a further optimization,
     * this class can be used directly to test many geometries against a single
     * rectangle.
     *
    """
    def __init__(self, rect):
        """
         * Create a new contains computer for two geometries.
         *
         * @param rect a rectangular geometry
        """
        self.rectangle = rect
        self.rectEnv = rect.envelope

    def isContainedInBoundary(self, geom) -> bool:

        # polygons can never be wholely contained in the boundary
        if geom.type_id == GeomTypeId.GEOS_POLYGON:
            return False

        if geom.type_id == GeomTypeId.GEOS_POINT:
            return self.isPointContainedInBoundary(geom)

        if geom.type_id == GeomTypeId.GEOS_LINESTRING:
            return self.isLineStringContainedInBoundary(geom)

        for g in geom.geoms:
            if not self.isContainedInBoundary(g):
                return False

        return True

    def isCoordContainedInBoundary(self, coord) -> bool:
        """
         * contains = false if the point is properly contained
         * in the rectangle.
         *
         * This code assumes that the point lies in the rectangle envelope
        """
        return (coord.x == self.rectEnv.minx or
            coord.x == self.rectEnv.maxx or
            coord.y == self.rectEnv.miny or
            coord.y == self.rectEnv.maxy)

    def isPointContainedInBoundary(self, geom) -> bool:
        """
         * Tests if a point is contained in the boundary of the target
         * rectangle.
         *
         * @param pt the point to test
         * @return true if the point is contained in the boundary
        """
        return self.isCoordContainedInBoundary(geom.coord)

    def isLineStringContainedInBoundary(self, geom) -> bool:
        """
         * Tests if a linestring is completely contained in the boundary
         * of the target rectangle.
         *
         * @param line the linestring to test
         * @return true if the linestring is contained in the boundary
        """
        coords = geom.coords
        for i in range(1, len(coords)):
            if not self.isLineSegmentContainedInBoundary(coords[i - 1], coords[i]):
                return False
        return True

    def isLineSegmentContainedInBoundary(self, p0, p1) -> bool:
        """
         * Tests if a line segment is contained in the boundary of the
         * target rectangle.
         *
         * @param p0 an endpoint of the segment
         * @param p1 an endpoint of the segment
         * @return true if the line segment is contained in the boundary
        """
        if p0 == p1:
            return self.isCoordContainedInBoundary(p0)

        if p0.x == p1.x:
            if (p0.x == self.rectEnv.minx or
                    p0.x == self.rectEnv.maxx):
                return True
        elif p0.y == p1.y:
            if (p0.y == self.rectEnv.miny or
                    p0.y == self.rectEnv.maxy):
                return True
        return False

    @staticmethod
    def contains(rect, geom) -> bool:
        rc = RectangleContains(rect)
        return rc._contains(geom)

    def _contains(self, geom) -> bool:

        if not self.rectEnv.contains(geom.envelope):
            return False

        # check that geom is not contained entirely in the rectangle boundary
        if self.isContainedInBoundary(geom):
            return False

        return True


# prepared


class PreparedGeometry():
    """
     * A base class for {@link PreparedGeometry} subclasses.
     *
     * Contains default implementations for methods, which simply delegate
     * to the equivalent {@link Geometry} methods.
     * This class may be used as a "no-op" class for Geometry types
     * which do not have a corresponding {@link PreparedGeometry} implementation.
     *
     * @author Martin Davis
    """
    def __init__(self, geom):
        self.setGeometry(geom)

    def setGeometry(self, geom):
        self.geom = geom
        """
         * List of representative points for this geometry.
         * One vertex is included for every component of the geometry
         * (i.e. including one for every ring of polygonal geometries)
        """
        self.representativePts = []
        ComponentCoordinateExtracter.getCoordinates(geom, self.representativePts)
        logger.debug("PreparedGeometry.setGeometry() pts:%s", len(self.representativePts))
        
    def envelopeIntersects(self, geom) -> bool:
        """
         * Determines whether a Geometry g interacts with
         * this geometry by testing the geometry envelopes.
         *
         * @param g a Geometry
         * @return true if the envelopes intersect
        """
        return self.geom.envelope.intersects(geom.envelope)

    def envelopeCovers(self, geom) -> bool:
        return self.geom.envelope.covers(geom.envelope)

    def isAnyTargetComponentInTest(self, geom) -> bool:
        """
         * Tests whether any representative of the target geometry
         * intersects the test geometry.
         * This is useful in A/A, A/L, A/P, L/P, and P/P cases.
         *
         * @param geom the test geometry
         * @param repPts the representative points of the target geometry
         * @return true if any component intersects the areal test geometry
        """
        locator = PointLocator()
        for coord in self.representativePts:
            if locator.intersects(coord, geom):
                return True
        return False

    def contains(self, geom) -> bool:
        return self.geom.contains(geom)

    def containsProperly(self, geom) -> bool:
        # since raw relate is used, provide some optimizations
        if not self.geom.envelope.contains(geom.envelope):
            return False
        # otherwise, compute using relate mask
        return self.geom.relate(geom, "T**FF*FF*")

    def coveredBy(self, geom) -> bool:
        return self.geom.coveredBy(geom)

    def covers(self, geom) -> bool:
        return self.geom.covers(geom)

    def crosses(self, geom) -> bool:
        return self.geom.crosses(geom)

    def disjoint(self, geom) -> bool:
        """
         * Standard implementation for all geometries.
        """
        return not self.intersects(geom)

    def intersects(self, geom) -> bool:
        return self.geom.intersects(geom)

    def overlaps(self, geom) -> bool:
        return self.geom.overlaps(geom)

    def touches(self, geom) -> bool:
        return self.geom.touches(geom)

    def within(self, geom) -> bool:
        return self.geom.within(geom)


class PreparedPoint(PreparedGeometry):
    """
     * A prepared version of {@link Point} or {@link MultiPoint} geometries.
     *
     * @author Martin Davis
    """
    def __init__(self, geom):
        PreparedGeometry.__init__(self, geom)

    def intersects(self, geom) -> bool:
        """
         * Tests whether this point intersects a {@link Geometry}.
         *
         * The optimization here is that computing topology for the test
         * geometry is avoided. This can be significant for large geometries.
        """
        if not self.envelopeIntersects(geom):
            return False

        # This avoids computing topology for the test geometry
        return self.isAnyTargetComponentInTest(geom)


class PreparedLineString(PreparedGeometry):
    """
     * A prepared version of {@link LinearRing}, {@link LineString} or {@link MultiLineString} geometries.
     *
     * @author mbdavis
     *
    """
    def __init__(self, geom):
        PreparedGeometry.__init__(self, geom)
        # noding.FastSegmentSetIntersectionFinder
        self._segIntFinder = None
    
    @property
    def intersectionFinder(self):
        if self._segIntFinder is None:
            segStrings = [] 
            SegmentStringUtil.extractSegmentStrings(self.geom, segStrings)
            self._segIntFinder = FastSegmentSetIntersectionFinder(segStrings)
        return self._segIntFinder

    def intersects(self, geom) -> bool:
        if not self.envelopeIntersects(geom):
            return False
        return PreparedLineStringIntersects.intersects(self, geom)


class PreparedLineStringIntersects():
    """
     * Computes the <tt>intersects</tt> spatial relationship predicate
     * for a target {@link PreparedLineString} relative to all other
     * {@link Geometry} classes.
     *
     * Uses short-circuit tests and indexing to improve performance.
     *
     * @author Martin Davis
     *
    """
    def __init__(self, prep):
        self.prepLine = prep

    @staticmethod
    def intersects(prep, geom) -> bool:
        """
         * Computes the intersects predicate between a {@link PreparedLineString}
         * and a {@link Geometry}.
         *
         * @param prep the prepared linestring
         * @param geom a test geometry
         * @return true if the linestring intersects the geometry
        """
        op = PreparedLineStringIntersects(prep)
        return op._intersects(geom)

    def _intersects(self, geom) -> bool:
        """
         * Tests whether this geometry intersects a given geometry.
         *
         * @param geom the test geometry
         * @return true if the test geometry intersects
        """
        lineSegStr = []
        SegmentStringUtil.extractSegmentStrings(geom, lineSegStr)
        # If any segments intersect, obviously intersects = true
        segsIntersect = self.prepLine.intersectionFinder.intersects(lineSegStr)
        if segsIntersect:
            return True

        # For L/L case we are done
        if geom.dimension == 1:
            return False

        # For L/A case, need to check for proper inclusion of the target in the test
        if geom.dimension == 2 and self.prepLine.isAnyTargetComponentInTest(geom):
            return True

        # for L/P case, need to check if any points lie on line(s)
        if geom.dimension == 0:
            return self.isAnyTestPointInTarget(geom)

        return False

    def isAnyTestPointInTarget(self, geom) -> bool:
        """
         * Tests whether any representative point of the test Geometry intersects
         * the target geometry.
         * Only handles test geometries which are Puntal (dimension 0)
         *
         * @param geom a Puntal geometry to test
         * @return true if any point of the argument intersects the prepared geometry
        """
        locator = PointLocator()
        coords = []
        ComponentCoordinateExtracter.getCoordinates(geom, coords)

        for coord in coords:

            if locator.intersects(coord, self.prepLine.geom):
                return True

        return False


class PreparedPolygonPredicate():
    """
     * A base class for predicate operations on {@link PreparedPolygon}s.
     *
     * @author mbdavis
    """
    def __init__(self, prep):
        # PreparedPolygon
        self.prep = prep

    def isAllTestComponentsInTarget(self, geom) -> bool:
        """
         * Tests whether all components of the test Geometry
         * are contained in the target geometry.
         *
         * Handles both linear and point components.
         *
         * @param geom a geometry to test
         * @return true if all components of the argument are contained
         *              in the target geometry
        """
        pts = []
        ComponentCoordinateExtracter.getCoordinates(geom, pts)
        for pt in pts:
            loc = self.prep.pointLocator.locate(pt)
            if loc == Location.EXTERIOR:
                return False
        return True

    def isAllTestComponentsInTargetInterior(self, geom) -> bool:
        """
         * Tests whether all components of the test Geometry
         * are contained in the interiors of the target geometry.
         *
         * Handles both linear and point components.
         *
         * @param geom a geometry to test
         * @return true if all componenta of the argument are contained in
         *              the target geometry interiors
        """
        pts = []
        ComponentCoordinateExtracter.getCoordinates(geom, pts)
        for pt in pts:
            loc = self.prep.pointLocator.locate(pt)
            if loc != Location.INTERIOR:
                return False
        return True

    def isAnyTestComponentInTarget(self, geom) -> bool:
        """
         * Tests whether any component of the test Geometry intersects
         * the area of the target geometry.
         *
         * Handles test geometries with both linear and point components.
         *
         * @param geom a geometry to test
         * @return true if any component of the argument intersects the
         *              prepared geometry
        """
        pts = []
        ComponentCoordinateExtracter.getCoordinates(geom, pts)
        for pt in pts:
            loc = self.prep.pointLocator.locate(pt)
            if loc != Location.EXTERIOR:
                return True
        return False

    def isAnyTestComponentInTargetInterior(self, geom) -> bool:
        """
         * Tests whether any component of the test Geometry intersects
         * the interiors of the target geometry.
         *
         * Handles test geometries with both linear and point components.
         *
         * @param geom a geometry to test
         * @return true if any component of the argument intersects the
         *              prepared area geometry interiors
        """
        pts = []
        ComponentCoordinateExtracter.getCoordinates(geom, pts)
        for pt in pts:
            loc = self.prep.pointLocator.locate(pt)
            if loc == Location.INTERIOR:
                return True
        return False

    def isAnyTargetComponentInAreaTest(self, geom, targetPts) -> bool:
        """
         * Tests whether any component of the target geometry
         * intersects the test geometry (which must be an areal geometry)
         *
         * @param geom the test geometry
         * @param repPts the representative points of the target geometry
         * @return true if any component intersects the areal test geometry
        """
        pts = []
        ComponentCoordinateExtracter.getCoordinates(geom, pts)
        for pt in pts:
            loc = SimplePointInAreaLocator.locate(pt, geom)
            if loc != Location.EXTERIOR:
                return True
        return False


class AbstractPreparedPolygonContains(PreparedPolygonPredicate):
    """
     * A base class containing the logic for computes the <tt>contains</tt>
     * and <tt>covers</tt> spatial relationship predicates
     * for a {@link PreparedPolygon} relative to all other {@link Geometry} classes.
     *
     * Uses short-circuit tests and indexing to improve performance.
     *
     * Contains and covers are very similar, and differ only in how certain
     * cases along the boundary are handled.  These cases require
     * full topological evaluation to handle, so all the code in
     * this class is common to both predicates.
     *
     * It is not possible to short-circuit in all cases, in particular
     * in the case where line segments of the test geometry touches the polygon
     * linework.
     * In this case full topology must be computed.
     * (However, if the test geometry consists of only points, this
     * <i>can</i> be evaluated in an optimized fashion.
     *
     * @author Martin Davis
    """
    def __init__(self, prep, requireSomePointInInterior: bool=False):
        PreparedPolygonPredicate.__init__(self, prep)
        self.hasSegmentIntersection = False
        self.hasProperIntersection = False
        self.hasNonProperIntersection = False
        """
         * This flag controls a difference between contains and covers.
         *
         * For contains the value is true.
         * For covers the value is false.
        """
        self.requireSomePointInInterior = requireSomePointInInterior

    def isProperIntersectionImpliesNotContainedSituation(self, geom) -> bool:

        # If the test geometry is polygonal we have the A/A situation.
        # In this case, a proper intersection indicates that
        # the Epsilon-Neighbourhood Exterior Intersection condition exists.
        # This condition means that in some small
        # area around the intersection point, there must exist a situation
        # where the interiors of the test intersects the exterior of the target.
        # This implies the test is NOT contained in the target.

        if (geom.type_id == GeomTypeId.GEOS_MULTIPOLYGON or
                geom.type_id == GeomTypeId.GEOS_POLYGON):
            return True

        # A single exterior with no interiors allows concluding that
        # a proper intersection implies not contained
        # (due to the Epsilon-Neighbourhood Exterior Intersection condition)

        if self.isSingleShell(self.prep.geom):
            return True

        return False

    def isSingleShell(self, geom) -> bool:
        """
         * Tests whether a geometry consists of a single polygon with no interiors.
         *
         * @return true if the geometry is a single polygon with no interiors
        """
        # handles single-element MultiPolygons, as well as Polygons
        if geom.numgeoms != 1:
            return False

        poly = geom.getGeometryN(0)
        return len(poly.interiors) == 0

    def findAndClassifyIntersections(self, geom) -> None:
        # noding.SegmentString
        lineSegStr = []
        SegmentStringUtil.extractSegmentStrings(geom, lineSegStr)

        li = LineIntersector()
        intDetector = SegmentIntersectionDetector(li)

        self.prep.intersectionFinder.intersects(lineSegStr, intDetector)

        self.hasSegmentIntersection = intDetector.hasIntersection
        self.hasProperIntersection = intDetector.hasProperIntersection
        self.hasNonProperIntersection = intDetector.hasNonProperIntersection

    def eval(self, geom) -> bool:
        """
         * Evaluate the <tt>contains</tt> or <tt>covers</tt> relationship
         * for the given geometry.
         *
         * @param geom the test geometry
         * @return true if the test geometry is contained
        """
        # Do point-in-poly tests first, since they are cheaper and may result
        # in a quick negative result.
        #
        # If a point of any test components does not lie in target,
        # result is false

        isAllInTargetArea = self.isAllTestComponentsInTarget(geom)
        if not isAllInTargetArea:
            return False

        # If the test geometry consists of only Points,
        # then it is now sufficient to test if any of those
        # points lie in the interiors of the target geometry.
        # If so, the test is contained.
        # If not, all points are on the boundary of the area,
        # which implies not contained.
        if self.requireSomePointInInterior and geom.dimension == 0:
            return self.isAnyTestComponentInTargetInterior(geom)

        # Check if there is any intersection between the line segments
        # in target and test.
        # In some important cases, finding a proper interesection implies that the
        # test geometry is NOT contained.
        # These cases are:
        # - If the test geometry is polygonal
        # - If the target geometry is a single polygon with no interiors
        # In both of these cases, a proper intersection implies that there
        # is some portion of the interiors of the test geometry lying outside
        # the target, which means that the test is not contained.
        properIntersectionImpliesNotContained = self.isProperIntersectionImpliesNotContainedSituation(geom)

        # find all intersection types which exist
        self.findAndClassifyIntersections(geom)

        if properIntersectionImpliesNotContained and self.hasProperIntersection:
            return False

        # If all intersections are proper
        # (i.e. no non-proper intersections occur)
        # we can conclude that the test geometry is not contained in the target area,
        # by the Epsilon-Neighbourhood Exterior Intersection condition.
        # In real-world data this is likely to be by far the most common situation,
        # since natural data is unlikely to have many exact vertex segment intersections.
        # Thus this check is very worthwhile, since it avoid having to perform
        # a full topological check.
        #
        # (If non-proper (vertex) intersections ARE found, this may indicate
        # a situation where two exteriors touch at a single vertex, which admits
        # the case where a line could cross between the exteriors and still be wholely contained in them.

        if self.hasSegmentIntersection and not self.hasNonProperIntersection:
            return False

        # If there is a segment intersection and the situation is not one
        # of the ones above, the only choice is to compute the full topological
        # relationship.  This is because contains/covers is very sensitive
        # to the situation along the boundary of the target.
        if self.hasSegmentIntersection:
            return self.fullTopologicalPredicate(geom)

        # This tests for the case where a ring of the target lies inside
        # a test polygon - which implies the exterior of the Target
        # intersects the interiors of the Test, and hence the result is false
        if (geom.type_id == GeomTypeId.GEOS_MULTIPOLYGON or
                geom.type_id == GeomTypeId.GEOS_POLYGON):
            isTargetInTestArea = self.isAnyTargetComponentInAreaTest(geom, self.prep.representativePts)
            if isTargetInTestArea:
                return False
        return True

    def fullTopologicalPredicate(self, geom) -> bool:
        """
         * Computes the full topological predicate.
         * Used when short-circuit tests are not conclusive.
         *
         * @param geom the test geometry
         * @return true if this prepared polygon has the relationship with the test geometry
        """
        raise NotImplementedError()


class PreparedPolygon(PreparedGeometry):
    """
     * A prepared version of {@link Polygon} or {@link MultiPolygon} geometries.
     *
     * @author mbdavis
     *
    """
    def __init__(self, geom):
        PreparedGeometry.__init__(self, geom)

        self.is_rectangle = geom.is_rectangle

        # noding.FastSegmentSetIntersectionFinder
        self._segIntFinder = None

        # algorithm.locate.PointOnGeometryLocator
        self._ptOnGeomLoc = None

    @property    
    def intersectionFinder(self):
        if self._segIntFinder is None:
            segStrings = []
            SegmentStringUtil.extractSegmentStrings(self.geom, segStrings)
            self._segIntFinder = FastSegmentSetIntersectionFinder(segStrings)
        return self._segIntFinder

    @property    
    def pointLocator(self):
        if self._ptOnGeomLoc is None:
            self._ptOnGeomLoc = IndexedPointInAreaLocator(self.geom)
        return self._ptOnGeomLoc

    def contains(self, geom) -> bool:

        if not self.envelopeCovers(geom):
            return False

        if self.is_rectangle:
            return RectangleContains.contains(self.geom, geom)

        return PreparedPolygonContains.contains(self, geom)

    def containsProperly(self, geom) -> bool:
        if not self.envelopeCovers(geom):
            return False

        return PreparedPolygonContainsProperly.containsProperly(self, geom)

    def covers(self, geom) -> bool:
        if not self.envelopeCovers(geom):
            return False

        return PreparedPolygonCovers.covers(self, geom)

    def intersects(self, geom) -> bool:
        if not self.envelopeIntersects(geom):
            return False

        if self.is_rectangle:
            return RectangleIntersects.intersects(self.geom, geom)

        return PreparedPolygonIntersects.intersects(self, geom)


class PreparedPolygonIntersects(PreparedPolygonPredicate):
    """
     * Computes the <tt>intersects</tt> spatial relationship predicate
     * for {@link PreparedPolygon}s relative to all other {@link Geometry} classes.
     *
     * Uses short-circuit tests and indexing to improve performance.
     *
     * @author Martin Davis
     *
    """
    def __init__(self, prep):
        """
         * Creates an instance of this operation.
         *
         * @param prep the PreparedPolygon to evaluate
        """
        PreparedPolygonPredicate.__init__(self, prep)

    @staticmethod
    def intersects(prep, geom) -> bool:
        """
         * Computes the intersects predicate between a {@link PreparedPolygon}
         * and a {@link Geometry}.
         *
         * @param prep the prepared polygon
         * @param geom a test geometry
         * @return true if the polygon intersects the geometry
        """
        polyInt = PreparedPolygonIntersects(prep)
        return polyInt._intersects(geom)

    def _intersects(self, geom) -> bool:
        """
         * Tests whether this PreparedPolygon intersects a given geometry.
         *
         * @param geom the test geometry
         * @return true if the test geometry intersects
        """
        # logger.debug("PreparedPolygonIntersects._intersects() type:%s", type(geom).__name__)
        
        isInPrepGeomArea = self.isAnyTestComponentInTarget(geom)

        if isInPrepGeomArea:
            return True
        
        if (geom.type_id == GeomTypeId.GEOS_POINT or
                geom.type_id == GeomTypeId.GEOS_MULTIPOINT):
            return False

        # if any segment intersect, result is true
        # noding.SegmentString.ConstVect lineSegStr;
        lineSegStr = []
        SegmentStringUtil.extractSegmentStrings(geom, lineSegStr)
        # logger.debug("PreparedPolygonIntersects._intersects() lines:%s", len(lineSegStr))
        segsIntersect = self.prep.intersectionFinder.intersects(lineSegStr)
        
        if segsIntersect:
            return True
        
        # If the test has dimension = 2 as well, it is necessary to
        # test for proper inclusion of the target.
        # Since no segments intersect, it is sufficient to test representative points.
        if geom.dimension == 2:

            isPrepGeomInArea = self.isAnyTargetComponentInAreaTest(geom, self.prep.representativePts)
            if isPrepGeomInArea:
                return True

        return False


class PreparedPolygonCovers(AbstractPreparedPolygonContains):
    """
     * Computes the <tt>covers</tt> spatial relationship predicate
     * for a {@link PreparedPolygon} relative to all other {@link Geometry} classes.
     *
     * Uses short-circuit tests and indexing to improve performance.
     *
     * It is not possible to short-circuit in all cases, in particular
     * in the case where the test geometry touches the polygon linework.
     * In this case full topology must be computed.
     *
     * @author Martin Davis
     *
    """
    def __init__(self, prep):
        PreparedPolygonPredicate.__init__(self, prep)

    def fullTopologicalPredicate(self, geom) -> bool:
        """
         * Computes the full topological <tt>covers</tt> predicate.
         * Used when short-circuit tests are not conclusive.
         *
         * @param geom the test geometry
         * @return true if this prepared polygon covers the test geometry
        """
        return self.prep.geom.covers(geom)

    @staticmethod
    def covers(prep, geom) -> bool:
        """
         * Computes the </tt>covers</tt> predicate between a {@link PreparedPolygon}
         * and a {@link Geometry}.
         *
         * @param prep the prepared polygon
         * @param geom a test geometry
         * @return true if the polygon covers the geometry
        """
        polyInt = PreparedPolygonCovers(prep)
        return polyInt._covers(geom)

    def _covers(self, geom) -> bool:
        """
         * Tests whether this PreparedPolygon <tt>covers</tt> a given geometry.
         *
         * @param geom the test geometry
         * @return true if the test geometry is covered
        """
        return self.eval(geom)


class PreparedPolygonContains(AbstractPreparedPolygonContains):
    """
     * Computes the <tt>contains</tt> spatial relationship predicate
     * for a {@link PreparedPolygon} relative to all other {@link Geometry} classes.
     *
     * Uses short-circuit tests and indexing to improve performance.
     *
     * It is not possible to short-circuit in all cases, in particular
     * in the case where the test geometry touches the polygon linework.
     * In this case full topology must be computed.
     *
     * @author Martin Davis
    """
    def __init__(self, prep):
        """
         * Creates an instance of this operation.
         *
         * @param prep the PreparedPolygon to evaluate
        """
        AbstractPreparedPolygonContains.__init__(self, prep, True)

    @staticmethod
    def contains(prep, geom) -> bool:
        """
         * Computes the </tt>contains</tt> predicate between a {@link PreparedPolygon}
         * and a {@link Geometry}.
         *
         * @param prep the prepared polygon
         * @param geom a test geometry
         * @return true if the polygon contains the geometry
        """
        polyInt = PreparedPolygonContains(prep)
        return polyInt._contains(geom)

    def _contains(self, geom) -> bool:
        """
         * Tests whether this PreparedPolygon <tt>contains</tt> a given geometry.
         *
         * @param geom the test geometry
         * @return true if the test geometry is contained
        """
        return self.eval(geom)

    def fullTopologicalPredicate(self, geom) -> bool:
        """
         * Computes the full topological <tt>contains</tt> predicate.
         * Used when short-circuit tests are not conclusive.
         *
         * @param geom the test geometry
         * @return true if this prepared polygon contains the test geometry
        """
        return self.prep.geom.contains(geom)


class PreparedPolygonContainsProperly(PreparedPolygonPredicate):
    """
     * Computes the <tt>containsProperly</tt> spatial relationship predicate
     * for {@link PreparedPolygon}s relative to all other {@link Geometry} classes.
     *
     * Uses short-circuit tests and indexing to improve performance.
     *
     * A Geometry A <tt>containsProperly</tt> another Geometry B iff
     * all points of B are contained in the Interior of A.
     * Equivalently, B is contained in A AND B does not intersect
     * the Boundary of A.
     *
     * The advantage to using this predicate is that it can be computed
     * efficiently, with no need to compute topology at individual points.
     * In a situation with many geometries intersecting the boundary
     * of the target geometry, this can make a performance difference.
     *
     * @author Martin Davis
    """
    def __init__(self, prep):
        """
         * Creates an instance of this operation.
         *
         * @param prep the PreparedPolygon to evaluate
        """
        AbstractPreparedPolygonContains.__init__(self, prep)

    @staticmethod
    def containsProperly(prep, geom) -> bool:
        """
         * Computes the </tt>containsProperly</tt> predicate between a {@link PreparedPolygon}
         * and a {@link Geometry}.
         *
         * @param prep the prepared polygon
         * @param geom a test geometry
         * @return true if the polygon properly contains the geometry
        """
        polyInt = PreparedPolygonContainsProperly(prep)
        return polyInt._containsProperly(geom)

    def _containsProperly(self, geom)-> bool:
        """
         * Tests whether this PreparedPolygon containsProperly a given geometry.
         *
         * @param geom the test geometry
         * @return true if the test geometry is contained properly
        """
        # Do point-in-poly tests first, since they are cheaper and may result
        # in a quick negative result.
        # If a point of any test components does not lie in target,
        # result is false
        isAllInPrepGeomArea = self.isAllTestComponentsInTargetInterior(geom)
        if not isAllInPrepGeomArea:
            return False

        # If any segments intersect, result is false
        # noding.SegmentString.ConstVect
        lineSegStr = []
        SegmentStringUtil.extractSegmentStrings(geom, lineSegStr)
        segsIntersect = self.prep.intersectionFinder.intersects(lineSegStr)

        if segsIntersect:
            return False

        """
         * Given that no segments intersect, if any vertex of the target
         * is contained in some test component.
         * the test is NOT properly contained.
         """
        if (geom.type_id == GeomTypeId.GEOS_MULTIPOLYGON or
                geom.type_id == GeomTypeId.GEOS_POLYGON):
            # TODO: generalize this to handle GeometryCollections

            isTargetGeomInTestArea = self.isAnyTargetComponentInAreaTest(geom, self.prep.representativePts)
            if isTargetGeomInTestArea:
                return False

        return True


class PreparedGeometryFactory():
    """
     * A factory for creating {@link PreparedGeometry}s.
     *
     * It chooses an appropriate implementation of PreparedGeometry
     * based on the geoemtric type of the input geometry.
     * In the future, the factory may accept hints that indicate
     * special optimizations which can be performed.
     *
     * @author Martin Davis
     *
    """

    @staticmethod
    def prepare(geom):
        """
         * Creates a new {@link PreparedGeometry} appropriate for the argument {@link Geometry}.
         *
         * @param geom the geometry to prepare
         * @return the prepared geometry
        """
        pf = PreparedGeometryFactory()
        return pf.create(geom)
    
    @staticmethod
    def prep(geom):    
        return PreparedGeometryFactory.prepare(geom)
        
    def create(self, geom):
        """
         * Creates a new {@link PreparedGeometry} appropriate for the argument {@link Geometry}.
         *
         * @param geom the geometry to prepare
         * @return the prepared geometry
        """
        tid = geom.type_id

        if tid in [
                GeomTypeId.GEOS_MULTIPOINT,
                GeomTypeId.GEOS_POINT
                ]:
            return PreparedPoint(geom)

        if tid in [
                GeomTypeId.GEOS_LINEARRING,
                GeomTypeId.GEOS_LINESTRING,
                GeomTypeId.GEOS_MULTILINESTRING
                ]:
            return PreparedLineString(geom)

        if tid in [
                GeomTypeId.GEOS_POLYGON,
                GeomTypeId.GEOS_MULTIPOLYGON
                ]:
            return PreparedPolygon(geom)

        return PreparedGeometry(geom)
