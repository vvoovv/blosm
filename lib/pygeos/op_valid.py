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


from .geomgraph import (
    PlanarGraph,
    GeometryGraph,
    EdgeRing,
    Node,
    DirectedEdgeStar
    )
from .algorithms import (
    CGAlgorithms,
    LineIntersector,
    MCPointInRing
    )
from .index_strtree import STRtree
from .shared import (
    logger,
    GeomTypeId,
    Position,
    Location
    )
from .op_relate import (
    RelateNodeGraph
    )
from .op_overlay import (
    MinimalEdgeRing,
    MaximalEdgeRing,
    OverlayNodeFactory
    )

    
class TopologyErrors():
    eError = 0
    eRepeatedPoint = 1
    eHoleOutsideShell = 2
    eNestedHoles = 3
    eDisconnectedInterior = 4
    eSelfIntersection = 5
    eRingSelfIntersection = 6
    eNestedShells = 7
    eDuplicatedRings = 8
    eTooFewPoints = 9
    eInvalidCoordinate = 10
    eRingNotClosed = 11
    msg = (
        "Topology Validation Error",
        "Repeated Point",
        "Hole lies outside exterior",
        "Holes are nested",
        "Interior is disconnected",
        "Self-intersection",
        "Ring Self-intersection",
        "Nested exteriors",
        "Duplicate Rings",
        "Too few points in geometry component",
        "Invalid Coordinate",
        "Ring is not closed")


class TopologyValidationError():

    def __init__(self, errorType: int, coord=None):
        self.errorType = errorType
        self.coord = coord

    @property
    def message(self):
        return TopologyErrors.msg[self.errorType]

    def __str__(self):
        return "{} at or near point:{}".format(self.message, self.coord)


class ConsistentAreaTester():
    """
     * Checks that a geomgraph.GeometryGraph representing an area
     * (a geom.Polygon or geom.MultiPolygon)
     * has consistent semantics for area geometries.
     * This check is required for any reasonable polygonal model
     * (including the OGC-SFS model, as well as models which allow ring
     * self-intersection at single points)
     *
     * Checks include:
     *
     *  - test for rings which properly intersect
     *    (but not for ring self-intersection, or intersections at vertices)
     *  - test for consistent labelling at all node points
     *    (this detects vertex intersections with invalid topology,
     *    i.e. where the exterior side of an edge lies in the interiors of the area)
     *  - test for duplicate rings
     *
     * If an inconsistency is found the location of the problem
     * is recorded and is available to the caller.
    """
    def __init__(self, graph):
        """
         * Creates a new tester for consistent areas.
         *
         * @param geomGraph the topology graph of the area geometry.
         *                  Caller keeps responsibility for its deletion
        """
        self._li = LineIntersector()
        # GeometryGraph
        self._graph = graph
        self._nodeGraph = RelateNodeGraph()
        # the intersection point found (if any)
        # Coordinate
        self.invalidPoint = None
    
    @property
    def isNodeConsistentArea(self):
        """
         * Check all nodes to see if their labels are consistent with
         * area topology.
         *
         * @return true if this area has a consistent node
         *         labelling
         * To fully check validity, it is necessary to
         * compute ALL intersections, including self-intersections within a single edge.
        """
        # SegmentIntersector
        intersector = self._graph.computeSelfNodes(self._li, True, True)
        if intersector.hasProper:
            logger.debug("ConsistentAreaTester SegmentIntersector.hasProper")
            self.invalidPoint = intersector.properIntersectionPoint
            return False

        self._nodeGraph.build(self._graph)
        return self.isNodeEdgeAreaLabelsConsistent

    @property
    def isNodeEdgeAreaLabelsConsistent(self):
        """
         * Check all nodes to see if their labels are consistent.
         * If any are not, return false
        """
        map = self._nodeGraph.nodes

        for node in map:
            if not node.star.isAreaLabelsConsistent(self._graph):
                logger.debug("ConsistentAreaTester !star.isAreaLabelsConsistent")
                self.invalidPoint = node.coord
                return False

        return True
    
    @property
    def hasDuplicateRings(self):
        """
         * Checks for two duplicate rings in an area.
         * Duplicate rings are rings that are topologically equal
         * (that is, which have the same sequence of points up to point order).
         * If the area is topologically consistent (determined by calling the
         * isNodeConsistentArea,
         * duplicate rings can be found by checking for EdgeBundles which contain
         * more than one geomgraph.EdgeEnd.
         * (This is because topologically consistent areas cannot have two rings sharing
         * the same line segment, unless the rings are equal).
         * The start point of one of the equal rings will be placed in
         * invalidPoint.
         *
         * @return true if this area Geometry is topologically consistent but has two duplicate rings
        """
        map = self._nodeGraph.nodes
        for node in map:
            # EdgeEndStar
            star = node.star
            # EdgeEndBundle
            for eeb in star:
                if len(eeb._edgeEnds) > 1:
                    logger.debug("ConsistentAreaTester.hasDuplicateRings")
                    self.invalidPoint = eeb.edge.coords[0]
                    return True

        return False


class ConnectedInteriorTester():
    """
     * This class tests that the interiors of an area Geometry
     * (Polygon or MultiPolygon)
     * is connected.
     *
     * An area Geometry is invalid if the interiors is disconnected.
     * This can happen if:
     *
     * - one or more interiors either form a chain touching the exterior at two places
     * - one or more interiors form a ring around a portion of the interiors
     *
     * If an inconsistency if found the location of the problem
     * is recorded.
    """
    def __init__(self, newGeomgraph):
        # GeometryFactory
        self._factory = None
        # GeometryGraph
        self._graph = newGeomgraph
        # Coordinate
        self.invalidPoint = None
        
    @property
    def isInteriorsConnected(self):
        # Edge
        splitEdges = []
        self._graph.computeSplitEdges(splitEdges)

        # PlanarGraph
        graph = PlanarGraph(OverlayNodeFactory())
        graph.addEdges(splitEdges)
        self._setInteriorEdgesInResult(graph)
        graph.linkResultDirectedEdges()

        # EdgeRing
        edgeRings = []
        self._buildEdgeRings(graph._edgeEnds, edgeRings)
        """
         * Mark all the edges for the edgeRings corresponding to the exteriors
         * of the input polygons.
         *
         * Only ONE ring gets marked for each exterior - if there are others
         * which remain unmarked this indicates a disconnected interiors.
        """
        self._visitShellInteriors(self._graph.geom, graph)
        """
         * If there are any unvisited exterior edges
         * (i.e. a ring which is not a hole and which has the interiors
         * of the parent area on the RHS)
         * this means that one or more interiors must have split the interiors of the
         * polygon into at least two pieces.  The polygon is thus invalid.
        """
        res = not self._hasUnvisitedShellEdge(edgeRings)

        return res

    @staticmethod
    def findDifferentPoint(coords, pt):

        for coord in coords:
            if coord is not pt:
                return coord

        return None

    def _visitLinkedDirectedEdges(self, start):
        # DirectedEdge
        startDe = start
        de = start
        while True:
            # found null Directed Edge
            assert(de is not None), "found null Directed Edge"
            de.isVisited = True
            de = de.next
            if de is startDe:
                break

    def _setInteriorEdgesInResult(self, planarGraph):
        # EdgeEnd
        ee = planarGraph._edgeEnds
        # DirectedEdge
        for de in ee:
            if de.label.getLocation(0, Position.RIGHT) == Location.INTERIOR:
                de.isInResult = True

    def _buildEdgeRings(self, dirEdges, minEdgeRings):
        """
         * Form DirectedEdges in graph into Minimal EdgeRings.
         *
         * Minimal Edgerings must be used, because only they are
         * guaranteed to provide a correct isHole computation.
         *
         * @param minEdgeRings : newly allocated minimal edge rings will
         *                       be push_back'ed here.
         *                       deletion responsibility is left to caller.
        """
        # DirectedEdge
        for de in dirEdges:
            # if this edge has not yet been processed
            if de.isInResult and de.edgeRing is None:
                # MaximalEdgeRing
                er = MaximalEdgeRing(de, self._factory)
                er.linkDirectedEdgesForMinimalEdgeRings()
                er.buildMinimalRings(minEdgeRings)

    def _visitShellInteriors(self, geom, planarGraph):
        """
         * Mark all the edges for the edgeRings corresponding to the exteriors
         * of the input polygons.  Note only ONE ring gets marked for each exterior.
        """
        type_id = geom.type_id
        if type_id == GeomTypeId.GEOS_POLYGON:
            self._visitInteriorRing(geom.exterior, planarGraph)

        elif type_id == GeomTypeId.GEOS_MULTIPOLYGON:
            for p in geom.geoms:
                self._visitInteriorRing(p.exterior, planarGraph)

    def _visitInteriorRing(self, ring, planarGraph):

        if ring.is_empty:
            return

        pts = ring.coords
        pt0 = pts[0]
        pt1 = ConnectedInteriorTester.findDifferentPoint(pts, pt0)

        # Edge
        e = planarGraph.findEdgeInSameDirection(pt0, pt1)

        # DirectedEdge
        de = planarGraph.findEdgeEnd(e)
        intDe = None

        if de.label.getLocation(0, Position.RIGHT) == Location.INTERIOR:
            intDe = de
        elif de.sym.label.getLocation(0, Position.RIGHT) == Location.INTERIOR:
            intDe = de.sym

        assert (intDe is not None), "unable to find dirEdge with Interior on RHS"

        self._visitLinkedDirectedEdges(intDe)

    def _hasUnvisitedShellEdge(self, edgeRings):
        """
         * Check if any exterior ring has an unvisited edge.
         * A exterior ring is a ring which is not a hole and which has the interiors
         * of the parent area on the RHS.
         * (Note that there may be non-hole rings with the interiors on the LHS,
         * since the interiors of interiors will also be polygonized into CW rings
         * by the linkAllDirectedEdges() step)
         *
         * @return true if there is an unvisited edge in a non-hole ring
        """
        for er in edgeRings:

            if er.isHole:
                continue

            # DirectedEdge
            edges = er.edges
            de = edges[0]

            if de.label.getLocation(0, Position.RIGHT) == Location.INTERIOR:
                continue
            """
             * the edgeRing is CW ring which surrounds the INT
             * of the area, so check all edges have been visited.
             * If any are unvisited, this is a disconnected part
             * of the interiors
            """
            for de in edges:
                if not de.isVisited:
                    self.invalidPoint = de.coord
                    return True

        return False


class IndexedNestedRingTester():
    """
     * Tests whether any of a set of {LinearRing}s are
     * nested inside another ring in the set, using a spatial
     * index to speed up the comparisons.
    """
    def __init__(self, newGraph):
        self._graph = newGraph
        self._index = None
        self.invalidPoint = None
        self._rings = []

    def getNestedPoint(self):
        return self._nestedPt

    def add(self, ring):
        self._rings.append(ring)

    def buildIndex(self):
        self._index = STRtree()
        for ring in self._rings:
            env = ring.envelope
            self._index.insert(env, ring)

    @property
    def isNonNested(self):
        self.buildIndex()
        for innerRing in self._rings:
            # CoordinateSequence
            innerRingPts = innerRing.coords
            results = []
            self._index.query(innerRing.envelope, results)
            for searchRing in results:
                searchRingPts = searchRing.coords
                if innerRing == searchRing:
                    continue
                if not innerRing.envelope.intersects(
                        searchRing.envelope):
                    continue
                # Coordinate
                innerRingPt = IsValidOp.findPtNotNode(self, innerRingPts, searchRing, self._graph)
                """
                 * If no non-node pts can be found, this means
                 * that the searchRing touches ALL of the innerRing vertices.
                 * This indicates an invalid polygon, since either
                 * the two interiors create a disconnected interiors,
                 * or they touch in an infinite number of points
                 * (i.e. along a line segment).
                 * Both of these cases are caught by other tests,
                 * so it is safe to simply skip this situation here.
                """
                if innerRingPt is None:
                    continue

                isInside = CGAlgorithms.isPointInRing(innerRingPt, searchRingPts)
                if isInside:
                    self.invalidPoint = innerRingPt
                return False
        return True


class IsValidOp():
    """
     * Implements the algorithsm required to compute the is_valid()
     * method for {Geometry}s.
    """
    def __init__(self, geometry):
        self.isSelfTouchingRingFormingHoleValid = False
        self.validErr = None
        self.geom = geometry
        self._checked = False

    def is_valid(self):

        if not self._checked:
            self._checked = True
            self.checkValid(self.geom)
        if self.validErr is not None:
            logger.debug("IsValidOp.is_valid() invalid geometry: %s", self.validErr)
        return self.validErr is None

    def findPtNotNode(self, testCoords, searchRing, graph):
        """
         * Find a point from the list of testCoords
         * that is NOT a node in the edge for the list of searchCoords
         *
         * @return the point found, or null if none found
        """
        # find edge corresponding to searchRing.
        # Edge
        searchEdge = graph.findEdge(searchRing)

        # find a point in the testCoords which is not a node of the searchRing
        # EdgeIntersectionList
        eiList = searchEdge.eiList

        for pt in testCoords:
            if not eiList.isIntersection(pt):
                return pt

        return None

    def checkValid(self, g):

        self.validErr = None

        if g is None:
            return

        # empty geometries are always valid!
        if g.is_empty:
            return

        type_id = g.type_id
        if type_id == GeomTypeId.GEOS_POINT:
            self.checkValidPoint(g)
        # LineString also handles LinearRings, so we check LinearRing first
        elif type_id == GeomTypeId.GEOS_LINEARRING:
            self.checkValidLinearRing(g)
        elif type_id == GeomTypeId.GEOS_LINESTRING:
            self.checkValidLineString(g)
        elif type_id == GeomTypeId.GEOS_POLYGON:
            self.checkValidPolygon(g)
        elif type_id == GeomTypeId.GEOS_MULTIPOLYGON:
            self.checkValidMultiPolygon(g)
        elif type_id in [GeomTypeId.GEOS_GEOMETRYCOLLECTION,
                    GeomTypeId.GEOS_MULTILINESTRING,
                    GeomTypeId.GEOS_MULTILINEARRING,
                    GeomTypeId.GEOS_MULTIPOINT]:
            self.checkValidGeometryCollection(g)

    def checkValidPoint(self, g) -> bool:
        return True

    def checkValidLineString(self, g):

        self.checkInvalidCoordinates(g.coords)

        if self.validErr is not None:
            return

        graph = GeometryGraph(0, g)
        self.checkTooFewPoints(graph)

    def checkValidLinearRing(self, g):

        self.checkInvalidCoordinates(g.coords)
        if self.validErr is not None:
            return

        self.checkClosedRing(g)
        if self.validErr is not None:
            return

        graph = GeometryGraph(0, g)
        self.checkTooFewPoints(graph)
        if self.validErr is not None:
            return

        li = LineIntersector()
        graph.computeSelfNodes(li, True, True)
        self.checkNoSelfIntersectingRings(graph)

    def checkValidPolygon(self, g):
        logger.debug("IsValidOp.checkValidPolygon")
        
        self.checkInvalidCoordinates(g.coords)
        if self.validErr is not None:
            return

        self.checkClosedRings(g)
        if self.validErr is not None:
            return

        graph = GeometryGraph(0, g)
        self.checkTooFewPoints(graph)
        if self.validErr is not None:
            return
            
        # compute intersections
        self.checkConsistentArea(graph)
        if self.validErr is not None:
            return

        if not self.isSelfTouchingRingFormingHoleValid:
            self.checkNoSelfIntersectingRings(graph)
            if self.validErr is not None:
                return

        self.checkHolesInShell(g, graph)
        if self.validErr is not None:
            return

        self.checkHolesNotNested(g, graph)
        if self.validErr is not None:
            return

        self.checkConnectedInteriors(graph)

    def checkValidMultiPolygon(self, g):
        logger.debug("IsValidOp.checkValidMultiPolygon")
        
        polys = []
        for poly in g.geoms:
            self.checkInvalidCoordinates(poly)
            if self.validErr is not None:
                return
            self.checkClosedRings(poly)
            if self.validErr is not None:
                return
            polys.append(poly)

        graph = GeometryGraph(0, g)

        self.checkTooFewPoints(graph)
        if self.validErr is not None:
            return

        self.checkConsistentArea(graph)
        if self.validErr is not None:
            return

        if not self.isSSelfTouchingRingFormingHoleValid:
            self.checkNoSelfIntersectingRings(graph)
            if self.validErr is not None:
                return

        for poly in polys:
            self.checkHolesInShell(poly, graph)
            if self.validErr is not None:
                return

        for poly in polys:
            self.checkHolesNotNested(poly, graph)
            if self.validErr is not None:
                return

        self.checkShellNotNested(g, graph)
        if self.validErr is not None:
            return

        self.checkConnectedInteriors(graph)

    def checkValidGeometryCollection(self, g):
        for geom in g.geoms:
            self.checkValid(geom)
            if self.validErr is not None:
                return

    def checkNoSelfIntersectingRings(self, graph):
        """
         * Check that there is no ring which self-intersects
         * (except of course at its endpoints).
         * This is required by OGC topology rules (but not by other models
         * such as ESRI SDE, which allow inverted exteriors and exverted interiors).
         *
         * @param graph the topology graph of the geometry
        """
        edges = graph.edges
        for edge in edges:
            self.checkNoSelfIntersectingRing(edge.intersections)
            if self.validErr is not None:
                return

    def checkNoSelfIntersectingRing(self, intersections):
        """
         * check that a ring does not self-intersect, except at its endpoints.
         * Algorithm is to count the number of times each node along edge
         * occurs.
         * If any occur more than once, that must be a self-intersection.
        """
        nodeSet = {}
        isFirst = True
        logger.debug("intersections:%s\n%s", len(intersections), "\n".join([str(it) for it in intersections]))
        
        for intersection in intersections:
            if isFirst:
                isFirst = False
                continue
            k = hash((intersection.coord.x, intersection.coord.y))
            if nodeSet.get(k) is not None:
                self.validErr = TopologyValidationError(
                    TopologyErrors.eRingSelfIntersection,
                    intersection.coord
                    )
                return
            else:
                nodeSet[k] = intersection

    def checkConsistentArea(self, graph):
        """
         * Checks that the arrangement of edges in a polygonal geometry graph
         * forms a consistent area.
         *
         * @param graph
         *
         * @see ConsistentAreaTester
        """
        # ConsistentAreaTester
        cat = ConsistentAreaTester(graph)
        if not cat.isNodeConsistentArea:
            self.validErr = TopologyValidationError(
                TopologyErrors.eSelfIntersection,
                cat.invalidPoint
                )
            return
        if cat.hasDuplicateRings:
            self.validErr = TopologyValidationError(
                TopologyErrors.eDuplicatedRings,
                cat.invalidPoint
                )
            return

    def checkTooFewPoints(self, graph):
        if graph.hasTooFewPoints:
            self.validErr = TopologyValidationError(
                TopologyErrors.eTooFewPoints,
                graph.invalidPoint
                )

    def checkInvalidCoordinates(self, coords):
        # check for numerical errors in coords
        return

    def checkClosedRings(self, poly):

        self.checkClosedRing(poly.exterior)
        if self.validErr is not None:
            return

        for hole in poly.interiors:
            self.checkClosedRing(hole)
            if self.validErr is not None:
                return

    def checkClosedRing(self, ring):
        if not ring.isClosed and not ring.is_empty:
            self.validErr = TopologyValidationError(
                TopologyErrors.eRingNotClosed,
                ring.coords[0]
                )

    def checkHolesInShell(self, p, graph):
        """
         * Test that each hole is inside the polygon exterior.
         * This routine assumes that the interiors have previously been tested
         * to ensure that all vertices lie on the exterior or inside it.
         * A simple test of a single point in the hole can be used,
         * provide the point is chosen such that it does not lie on the
         * boundary of the exterior.
         *
         * @param p the polygon to be tested for hole inclusion
         * @param graph a geomgraph.GeometryGraph incorporating the polygon
        """
        exterior = p.exterior

        if exterior.is_empty:
            for hole in p.interiors:
                if not hole.is_empty:
                    self.validErr = TopologyValidationError(
                        TopologyErrors.eHoleOutsideShell
                        )
                    return
            return

        pir = MCPointInRing(exterior)
        for hole in p.interiors:

            holePt = self.findPtNotNode(hole.coords, exterior, graph)
            """
             * If no non-node hole vertex can be found, the hole must
             * split the polygon into disconnected interiors.
             * This will be caught by a subsequent check.
            """
            if holePt is None:
                return

            outside = not pir.isInside(holePt)
            if outside:
                self.validErr = TopologyValidationError(
                    TopologyErrors.eHoleOutsideShell,
                    holePt
                    )
                return

    def checkHolesNotNested(self, p, graph):
        """
         * Tests that no hole is nested inside another hole.
         * This routine assumes that the interiors are disjoint.
         * To ensure this, interiors have previously been tested
         * to ensure that:
         *
         *  - they do not partially overlap
         *    (checked by checkRelateConsistency)
         *  - they are not identical
         *    (checked by checkRelateConsistency)
         *
        """
        nestedTester = IndexedNestedRingTester(graph)

        for innerHole in p.interiors:

            if innerHole.is_empty:
                continue

            nestedTester.add(innerHole)

        isNonNested = nestedTester.isNonNested
        if not isNonNested:
            self.validErr = TopologyValidationError(
                TopologyErrors.eNestedHoles,
                nestedTester.invalidPoint
                )

    def checkShellsNotNested(self, mp, graph):
        """
         * Tests that no element polygon is wholly in the interiors of another
         * element polygon.
         *
         * Preconditions:
         *
         * - exteriors do not partially overlap
         * - exteriors do not touch along an edge
         * - no duplicate rings exist
         *
         * This routine relies on the fact that while polygon exteriors
         * may touch at one or more vertices, they cannot touch at
         * ALL vertices.
        """
        for i, p in enumerate(mp):

            exterior = p.exterior

            if exterior.is_empty:
                continue

            for j, p2 in enumerate(mp):

                if j == i:
                    continue

                if p2.is_empty:
                    continue

                self.checkShellNotNested(exterior, p2, graph)

                if self.validErr is not None:
                    return

    def checkShellNotNested(self, exterior, p, graph):
        """
         * Check if a exterior is incorrectly nested within a polygon.
         * This is the case if the exterior is inside the polygon exterior,
         * but not inside a polygon hole.
         * (If the exterior is inside a polygon hole, the nesting is valid.)
         *
         * The algorithm used relies on the fact that the rings must be
         * properly contained.
         * E.g. they cannot partially overlap (this has been previously
         * checked by checkRelateConsistency
        """
        # CoordinateSequence
        exteriorPts = exterior.coords
        # LinearRing
        polyShell = p.exterior
        # CoordinateSequence
        polyPts = polyShell.coords
        # Coordinate
        exteriorPt = self.findPtNotNode(exteriorPts, polyShell, graph)
        # if no point could be found, we can assume that the exterior
        # is outside the polygon
        if exteriorPt is None:
            return

        insidePolyShell = CGAlgorithms.isPointInRing(exteriorPt, polyPts)
        if not insidePolyShell:
            return

        # if no interiors, this is an error !
        if len(p.interiors) <= 0:
            self.validErr = TopologyValidationError(
                TopologyErrors.eNestedShells,
                exteriorPt
                )
            return
        """
         * Check if the exterior is inside one of the interiors.
         * This is the case if one of the calls to checkShellInsideHole
         * returns a null coordinate.
         * Otherwise, the exterior is not properly contained in a hole, which is
         * an error.
        """
        badNestedPt = None
        for hole in p.interiors:
            badNestedPt = self.checkShellInsideHole(exterior, hole, graph)
            if badNestedPt is None:
                return
        self.validErr = TopologyValidationError(
            TopologyErrors.eNestedShells,
            badNestedPt
            )

    def checkShellInsideHole(self, exterior, hole, graph):
        """
         * This routine checks to see if a exterior is properly contained
         * in a hole.
         * It assumes that the edges of the exterior and hole do not
         * properly intersect.
         *
         * @return null if the exterior is properly contained, or
         *   a Coordinate which is not inside the hole if it is not
         *
        """
        # CoordinateSequence
        exteriorPts = exterior.coords
        holePts = hole.coords

        exteriorPt = self.findPtNotNode(exteriorPts, hole, graph)

        # if point is on exterior but not hole, check that the exterior is
        # inside the hole
        if exteriorPt is not None:
            insideHole = CGAlgorithms.isPointInRing(exteriorPt, holePts)
            if not insideHole:
                return exteriorPt

        holePt = self.findPtNotNode(holePts, exterior, graph)

        # if point is on hole but not exterior, check that the hole is
        # outside the exterior
        if holePt is not None:
            insideShell = CGAlgorithms.isPointInRing(holePt, exteriorPts)
            if insideShell:
                return holePt

        return None

    def checkConnectedInteriors(self, graph):

        cit = ConnectedInteriorTester(graph)
        cit._factory = self.geom._factory
        if not cit.isInteriorsConnected:
            self.validErr = TopologyValidationError(
                TopologyErrors.eDisconnectedInterior,
                cit.invalidPoint
                )
