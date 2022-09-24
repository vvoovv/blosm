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


import time
from .algorithms import (
    CGAlgorithms
    )
from .planargraph import (
    PlanarGraph,
    DirectedEdge,
    Edge,
    Node
    )
from .geom import GeometryFactory
from .shared import logger


class EdgeRing():

    """
        * Represents a ring of PolygonizeDirectedEdge which form
        * a ring of a polygon.  The ring may be either an outer exterior or a hole.
    """

    def __init__(self, factory):
        # geom.GeometryFactory
        self._factory = factory

        # DirectedEdge
        self._deList = []

        # Cache for efficiency
        # LinearRing
        self._ring = None

        # CoordinateSequence
        self._coords = None

        # geom.Geometry
        self._interior = []

    @property
    def coords(self):
        """
        * Computes the list of coordinates which are contained in this ring.
        * The coordinatea are computed once only and cached.
        *
        * @return an array of the Coordinate in this ring
        """
        if self._coords is None:
            self._coords = []
            for de in self._deList:
                edge = de.edge
                EdgeRing._addEdge(edge.line.coords, de.edgeDirection, self._coords)

        return self._coords

    @staticmethod
    def _addEdge(coords, isForward, coordList):
        if isForward:
            newco = coords
        else:
            newco = list(reversed(coords))
        coordList.extend(newco)

    @staticmethod
    def findEdgeRingContaining(testEr, exteriorList):
        """
        * Find the innermost enclosing exterior EdgeRing
        * containing the argument EdgeRing, if any.
        *
        * The innermost enclosing ring is the <i>smallest</i> enclosing ring.
        * The algorithm used depends on the fact that:
        *
        * ring A contains ring B iff envelope(ring A) contains envelope(ring B)
        *
        * This routine is only safe to use if the chosen point of the hole
        * is known to be properly contained in a exterior
        * (which is guaranteed to be the case if the hole does not touch
        * its exterior)
        *
        * @return containing EdgeRing, if there is one
        * @return null if no containing EdgeRing is found
        """
        # LinearRing
        testRing = testEr.linearRing

        if testRing is None:
            return None

        # Envelope
        testEnv = testRing.envelope

        # Coordinate
        testPt = testRing.coords[0]

        # EdgeRing
        minShell = None

        # Envelope
        minEnv = None

        for tryShell in exteriorList:

            # LinearRing
            tryRing = tryShell.linearRing

            # Envelope
            tryEnv = tryRing.envelope

            if minShell is not None:
                minEnv = minShell.linearRing.envelope

            isContained = False

            # the hole envelope cannot equal the exterior envelope
            if tryEnv.equals(testEnv):
                continue

            # CoordinateSequence
            tryCoords = tryRing.coords

            if tryEnv.contains(testEnv):

                testPt = EdgeRing.ptNotInList(testRing.coords, tryCoords)
                # testPt my be None !
                if testPt is not None:
                    if CGAlgorithms.isPointInRing(testPt, tryCoords):
                        isContained = True

            # check if this new containing ring is smaller
            # than the current minimum ring
            if isContained:
                if minShell is None or minEnv.contains(tryEnv):
                    minShell = tryShell

        return minShell

    @staticmethod
    def isInList(pt, pts):
        """
        * Tests whether a given point is in an array of points.
        * Uses a value-based test.
        *
        * @param pt a Coordinate for the test point
        * @param pts an array of Coordinate to test
        * @return true if the point is in the array
        """
        return pt in pts

    @staticmethod
    def ptNotInList(testPts, pts):
        """
        * Finds a point in a list of points which is not contained in
        * another list of points.
        *
        * @param testPts the CoordinateSequence to test
        * @param pts the CoordinateSequence to test the input points against
        * @return a Coordinate reference from testPts which is
        * not in pts, or Coordinate.nullCoord
        """
        for testPt in testPts:
            if testPt not in pts:
                return testPt

        return None

    def add(self, de):
        """
        * Adds a DirectedEdge which is known to form part of this ring.
        *
        * @param de the DirectedEdge to add. Ownership to the caller.
        """
        self._deList.append(de)

    @property
    def isHole(self):
        """
        * Tests whether this ring is a hole.
        *
        * Due to the way the edges in the polyongization graph are linked,
        * a ring is a hole if it is oriented counter-clockwise.
        * @return true if this ring is a hole
        """
        return CGAlgorithms.isCCW(self.linearRing.coords)

    def addHole(self, hole):
        """
        * Adds a hole to the polygon formed by this ring.
         *
         * @param hole the LinearRing forming the hole.
        void addHole(geom.LinearRing *hole)
        """
        if self._interior is None:
            self._interior = []
        self._interior.append(hole)

    @property
    def is_valid(self):
        """
        * Tests if the LinearRing ring formed by this edge ring
        * is topologically valid.
        """
        if self.linearRing is None:
            return False
        return self._ring.is_valid

    @property
    def lineString(self):
        """
        * Gets the coordinates for this ring as a LineString.
        *
        * Used to return the coordinates in this ring
        * as a valid geometry, when it has been detected that the ring
        * is topologically invalid.
        * @return a LineString containing the coordinates in this ring
        """
        return self._factory.createLineString(self.coords)

    @property
    def linearRing(self):
        """
        * Returns this ring as a LinearRing, or null if an Exception
        * occurs while creating it (such as a topology problem).
        *
        * Ownership of ring is retained by the object.
        * Details of problems are written to standard output.
        """
        if self._ring is not None:
            return self._ring
        try:
            self._ring = self._factory.createLinearRing(self.coords)
        except:
            logger.error("EdgeRing.getRingInternal")
            pass

        return self._ring

    def getLinearRing(self):
        """
        * Returns this ring as a LinearRing, or null if an Exception
        * occurs while creating it (such as a topology problem).
        *
        * Ownership of ring is retained by the object.
        * Details of problems are written to standard output.
        * Caller gets ownership of ring.
        """
        ret = self.linearRing
        self._ring = None
        return ret

    def getPolygon(self):
        """
        * Computes the Polygon formed by this ring and any contained interiors.
        *
        * LinearRings ownership is transferred to returned polygon.
        * Subsequent calls to the function will return NULL.
        *
        * @return the Polygon formed by this ring and its interiors.
        """
        # Polygon
        poly = self._factory.createPolygon(self._ring, self._interior)
        self._ring = None
        self._interior = None
        return poly


class PolygonizeDirectedEdge(DirectedEdge):
    """
    * A DirectedEdge of a PolygonizeGraph, which represents
    * an edge of a polygon formed by the graph.
    *
    * May be logically deleted from the graph by setting the
    * marked flag.
    """
    def __init__(self, newFrom, newTo, directionPt, newEdgeDirection):

        DirectedEdge.__init__(self, newFrom, newTo, directionPt, newEdgeDirection)

        # EdgeRing
        self.edgeRing = None
        # PolygonizeDirectedEdge
        self.next = None

    @property
    def isInRing(self):
        return self.edgeRing is not None


class PolygonizeEdge(Edge):
    """
    * An edge of a polygonization graph.
    *
    * @version 1.4
    """
    def __init__(self, line):
        Edge.__init__(self)
        # LineString
        self.line = line


class PolygonizeGraph(PlanarGraph):
    """
    * Represents a planar graph of edges that can be used to compute a
    * polygonization, and implements the algorithms to compute the
    * EdgeRings formed by the graph.
    *
    * The marked flag on DirectedEdge is used to indicate that a directed edge
    * has be logically deleted from the graph.
    """
    def __init__(self, factory):

        PlanarGraph.__init__(self)
        self._factory = factory

    @staticmethod
    def deleteAllEdges(node) -> None:
        """
        * Deletes all edges at a node
        * @param node: planargraph.Node
        """
        # DirectedEdge
        edges = node.deStar.edges
        for de in edges:
            # PolygonizeDirectedEdge
            de.marked = True
            # PolygonizeDirectedEdge
            sym = de.sym
            if sym is not None:
                sym.marked = True

    def addEdge(self, line) -> None:
        """
        * Add a LineString forming an edge of the polygon graph.
        * @param line the geom.LineString to add
        """
        if line.is_empty:
            return

        # CoordinateSequence *
        coords = line.coords

        """
        * This would catch invalid linestrings
        * (containing duplicated points only)
        """
        if len(coords) < 2:
            return

        # const Coordinate&
        startPt = coords[0]
        # const Coordinate&
        endPt = coords[-1]

        # Node *
        nStart = self.getNode(startPt)
        # Node *
        nEnd = self.getNode(endPt)

        # DirectedEdge *
        de0 = PolygonizeDirectedEdge(nStart, nEnd, coords[1], True)

        # DirectedEdge *
        de1 = PolygonizeDirectedEdge(nEnd, nStart, coords[-2], False)

        # Edge *
        edge = PolygonizeEdge(line)
        edge.setDirectedEdges(de0, de1)
        super(PolygonizeGraph, self).addEdge(edge)

    def getEdgeRings(self, edgeRingList: list) -> None:
        """
         * Computes the EdgeRings formed by the edges in this graph.
         *
         * @param edgeRingList : the EdgeRing found by the
         * 	polygonization process will be pushed here.
        """
        t = time.time()
        #  maybe could optimize this, since most of these pointers should
        #  be set correctly already
        #  by deleteCutEdges()
        self.computeNextCWEdges()

        #  clear labels of all edges in graph
        PolygonizeGraph.label(self.dirEdges, -1)

        # PolygonizeDirectedEdge
        maximalRings = []
        PolygonizeGraph._findLabeledEdgeRings(self.dirEdges, maximalRings)
        self._convertMaximalToMinimalEdgeRings(maximalRings)
        maximalRings.clear()  # not needed anymore

        #  find all edgerings
        for de in self.dirEdges:

            if de.marked or de.isInRing:
                continue

            er = self.findEdgeRing(de)
            edgeRingList.append(er)

        logger.debug("PolygonizeGraph.getEdgeRings() :%.4f seconds", (time.time() - t))

    @staticmethod
    def _getDegreeNonDeleted(node) -> int:
        """
            @param node: Planargraph.Node
        """
        # DirectedEdge
        edges = node.deStar.edges
        degree = 0
        for de in edges:
            # PolygonizeDirectedEdge
            if not de.marked:
                degree += 1
        return degree

    @staticmethod
    def _getDegree(node, label: int) -> int:
        # DirectedEdge
        edges = node.deStar.edges
        degree = 0
        for de in edges:
            # PolygonizeDirectedEdge
            if de.label == label:
                degree += 1
        return degree

    def getNode(self, coord):
        # Node
        node = self.findNode(coord)
        if node is None:
            node = Node(coord)
            #  ensure node is only added once to graph
            self.addNode(node)

        return node

    def computeNextCWEdges(self) -> None:
        # Node
        nodes = []
        self.getNodes(nodes)
        for node in nodes:
            PolygonizeGraph._computeNextCWEdges(node)

    @staticmethod
    def _computeNextCWEdges(node) -> None:

        # DirectedEdgeStar *
        deStar = node.deStar
        # PolygonizeDirectedEdge *
        startDE = None
        # PolygonizeDirectedEdge *
        prevDE = None

        #  the edges are stored in CCW order around the star
        # DirectedEdge
        edges = deStar.edges
        for outDE in edges:
            if outDE.marked:
                continue
            if startDE is None:
                startDE = outDE
            if prevDE is not None:
                sym = prevDE.sym
                sym.next = outDE

            prevDE = outDE

        if prevDE is not None:
            sym = prevDE.sym
            sym.next = startDE

    def _convertMaximalToMinimalEdgeRings(self, ringEdges: list) -> None:
        """
         * Convert the maximal edge rings found by the initial graph traversal
         * into the minimal edge rings required by JTS polygon topology rules.
         *
         * @param ringEdges
         * 	the list of start edges for the edgeRings to convert.
        """
        # Node
        intNodes = []
        for de in ringEdges:
            # PolygonizeDirectedEdge
            label = de.label
            PolygonizeGraph._findIntersectionNodes(de, label, intNodes)

            #  set the next pointers for the edges around each node
            for node in intNodes:
                PolygonizeGraph._computeNextCCWEdges(node, label)

            intNodes.clear()

    @staticmethod
    def _findIntersectionNodes(startDE, label: int, intNodes: list) -> None:
        """
         * Finds all nodes in a maximal edgering
         * which are self-intersection nodes
         *
         * @param startDE
         * @param label
         * @param intNodes : intersection nodes found will be pushed here
         *                   the vector won't be cleared before pushing.
        """
        # PolygonizeDirectedEdge
        de = startDE
        while (True):
            # Node
            node = de._from

            if PolygonizeGraph._getDegree(node, label) > 1:
                intNodes.append(node)

            de = de.next

            assert(de is not None), "found NULL DE in ring"
            assert(de == startDE or not de.isInRing), "found DE already in ring"

            if de is startDE:
                break

    @staticmethod
    def _findLabeledEdgeRings(dirEdges: list, edgeRingStarts: list) -> None:
        """
         * Finds and labels all edgerings in the graph.
         *
         * The edge rings are labelling with unique integers.
         * The labelling allows detecting cut edges.
         *
         * @param dirEdgesIn  a list of the DirectedEdges in the graph
         * @param dirEdgesOut each ring found will be pushed here
        """
        # DirectedEdge
        edges = []

        #  label the edge rings formed
        currLabel = 1
        for de in dirEdges:

            # PolygonizeDirectedEdge
            if de.marked or de.label >= 0:
                continue

            edgeRingStarts.append(de)

            PolygonizeGraph._findDirEdgesInRing(de, edges)
            PolygonizeGraph.label(edges, currLabel)
            edges.clear()

            currLabel += 1

    def deleteCutEdges(self, cutLines: list) -> None:
        """
         * Finds and removes all cut edges from the graph.
         *
         * @param cutLines : the list of the LineString forming the removed
         *                   cut edges will be pushed here.
        """

        self.computeNextCWEdges()

        # PolygonizeDirectedEdge label the current set of edgerings
        junk = []
        PolygonizeGraph._findLabeledEdgeRings(self.dirEdges, junk)
        junk.clear()  # not needed anymore

        """
         * Cut Edges are edges where both dirEdges have the same label.
         * Delete them, and record them
        """

        for de in self.dirEdges:
            # PolygonizeDirectedEdge
            if de.marked:
                continue

            # PolygonizeDirectedEdge
            sym = de.sym

            if de.label == sym.label:

                de.marked = True
                sym.marked = True

                # save the line as a cut edge
                # PolygonizeEdge
                ed = de.edge
                cutLines.append(ed.line)

    @staticmethod
    def label(dirEdges: list, label: int) -> None:
        for de in dirEdges:
            de.label = label

    @staticmethod
    def _computeNextCCWEdges(node, label: int) -> None:
        """
         * Computes the next edge pointers going CCW around the given node, for the
         * given edgering label.
         * This algorithm has the effect of converting maximal edgerings into
         * minimal edgerings
        """
        # DirectedEdgeStar *
        deStar = node.deStar
        # PolygonizeDirectedEdge *
        firstOutDE = None
        # PolygonizeDirectedEdge *
        prevInDE = None

        #  the edges are stored in CCW order around the star
        # DirectedEdge
        edges = deStar.edges

        for de in reversed(edges):

            # PolygonizeDirectedEdge *
            sym = de.sym

            outDE = None
            if de.label == label:
                outDE = de

            # PolygonizeDirectedEdge *
            inDE = None
            if sym.label == label:
                inDE = sym

            if outDE is None and inDE is None:
                # this edge is not in edgering
                continue

            if inDE is not None:
                prevInDE = inDE

            if outDE is not None:
                if prevInDE is not None:
                    prevInDE.next = outDE
                    prevInDE = None

                if firstOutDE is None:
                    firstOutDE = outDE

        if prevInDE is not None:
            prevInDE.next = firstOutDE

    @staticmethod
    def _findDirEdgesInRing(startDE, edgesInRing: list) -> None:
        """
         * Traverse a ring of DirectedEdges, accumulating them into a list.
         * This assumes that all dangling directed edges have been removed
         * from the graph, so that there is always a next dirEdge.
         *
         * @param startDE the DirectedEdge to start traversing at
         * @param edgesInRing : the DirectedEdges that form a ring will
         *                      be pushed here.
        """
        de = startDE
        while (True):

            edgesInRing.append(de)
            de = de.next
            assert(de is not None), "found NULL DE in ring"
            assert(de == startDE or not de.isInRing), "found DE already in ring"

            if (de is startDE):
                break

    def findEdgeRing(self, startDE):
        """
         * @param startDE the DirectedEdge to start traversing at
        """
        # PolygonizeDirectedEdge
        de = startDE
        # EdgeRing *
        er = EdgeRing(self._factory)

        while (True):
            er.add(de)
            de.edgeRing = er
            de = de.next
            assert(de is not None), "found NULL DE in ring"
            assert(de == startDE or not de.isInRing), "found DE already in ring"

            if (de is startDE):
                break

        return er

    def deleteDangles(self, dangleLines: list) -> None:
        """
         * Marks all edges from the graph which are "dangles".
         *
         * Dangles are which are incident on a node with degree 1.
         * This process is recursive, since removing a dangling edge
         * may result in another edge becoming a dangle.
         * In order to handle large recursion depths efficiently,
         * an explicit recursion stack is used
         *
         * @param dangleLines : the LineStrings that formed dangles will
         *                      be push_back'ed here
        """
        # Node
        nodeStack = []
        self.findNodesOfDegree(1, nodeStack)

        # LineString
        uniqueDangles = []

        while len(nodeStack) > 0:
            # Node *
            node = nodeStack.pop()
            PolygonizeGraph.deleteAllEdges(node)

            # DirectedEdge
            nodeOutEdges = node.deStar.edges

            for de in nodeOutEdges:
                # PolygonizeDirectedEdge
                #  delete this edge and its sym
                de.marked = True

                # PolygonizeDirectedEdge
                sym = de.sym
                if sym is not None:
                    sym.marked = True

                #  save the line as a dangle
                # PolygonizeEdge *
                edge = de.edge

                # LineString*
                ls = edge.line
                if ls not in uniqueDangles:
                    uniqueDangles.append(ls)
                    dangleLines.append(ls)

                # Node *
                toNode = de._to
                #  add the toNode to the list to be processed,
                #  if it is now a dangle
                if PolygonizeGraph._getDegreeNonDeleted(toNode) == 1:
                    nodeStack.append(toNode)


class Polygonizer():
    """
     * Polygonizes a set of Geometrys which contain linework that
     * represents the edges of a planar graph.
     *
     * Any dimension of Geometry is handled - the constituent linework is extracted
     * to form the edges.
     * The edges must be correctly noded that is, they must only meet
     * at their endpoints.  The Polygonizer will still run on incorrectly noded input
     * but will not form polygons from incorrected noded edges.
     *
     * The Polygonizer reports the follow kinds of errors:
     *
     * - <b>Dangles</b> - edges which have one or both ends which are
     *   not incident on another edge endpoint
     * - <b>Cut Edges</b> - edges which are connected at both ends but
     *   which do not form part of polygon
     * - <b>Invalid Ring Lines</b> - edges which form rings which are invalid
     *   (e.g. the component lines contain a self-intersection)
     *
    """
    def __init__(self, skip_validity_check):
        """
         * Create a polygonizer with the same GeometryFactory
         * as the input Geometry
        """

        self.graph = None

        # initialize with empty collections, in case nothing is computed
        # LineString
        self.dangles = []
        self.cutEdges = []
        self.invalidRingLines = []
        # EdgeRing
        self.holeList = []
        self.exteriorList = []
        # Polygon
        self.polyList = None
        self.skip_validity_check = skip_validity_check
        
    def addGeometryList(self, geomList):
        """
         * Add a collection of geometries to be polygonized.
         * May be called multiple times.
         * Any dimension of Geometry may be added
         * the constituent linework will be extracted and used
         *
         * @param geomList a list of {Geometry}s with linework to be polygonized
        """
        for geometry in geomList:
            self.addLinestring(geometry)

    def addLinestring(self, line):
        """
         * Add a linestring to the graph of polygon edges.
         *
         * @param line the LineString to add
        """
        if self.graph is None:
            self.graph = PolygonizeGraph(line._factory)

        self.graph.addEdge(line)

    def getPolygons(self):
        """
         * Gets the list of polygons formed by the polygonization.
         * @return a collection of Polygons
        """
        self.polygonize()
        ret = self.polyList
        self.polyList = None
        return ret

    def getDangles(self):
        self.polygonize()
        return self.dangles

    def getCutEdges(self):
        self.polygonize()
        return self.cutEdges

    def getInvalidRingLines(self):
        self.polygonize()
        return self.invalidRingLines

    def polygonize(self):
        t = time.time()
        # check if already computed
        if self.polyList is not None:
            return

        # vector<Polygon*>()
        self.polyList = []

        # if no geometries were supplied it's possible graph could be null
        if self.graph is None:
            return

        self.graph.deleteDangles(self.dangles)
        self.graph.deleteCutEdges(self.cutEdges)

        # vector<EdgeRing*>
        self.edgeRingList = []
        self.graph.getEdgeRings(self.edgeRingList)

        logger.debug("Polygonizer.polygonize(): %s edgeRings in graph", len(self.edgeRingList))

        # vector<EdgeRing*>
        self.validEdgeRingList = []
        """ what if it was populated already ? we should clean ! """
        self.invalidRingLines.clear()
        self._findValidRings(self.edgeRingList, self.validEdgeRingList, self.invalidRingLines)

        logger.debug("                           %s valid", len(self.validEdgeRingList))
        logger.debug("                           %s invalid", len(self.invalidRingLines))

        self._findShellsAndHoles(self.validEdgeRingList)

        logger.debug("                           %s interiors", len(self.holeList))
        logger.debug("                           %s exteriors", len(self.exteriorList))

        self._assignHolesToShells(self.holeList, self.exteriorList)

        for edgeRing in self.exteriorList:
            self.polyList.append(edgeRing.getPolygon())

        logger.debug("Polygonizer.polygonize() :%.4f seconds", (time.time() - t))

    def _findValidRings(self, edgeRingList, validEdgeRingList, invalidRingList):
        t = time.time()
        for edgeRing in edgeRingList:
            if self.skip_validity_check or edgeRing.is_valid:
                validEdgeRingList.append(edgeRing)
            else:
                invalidRingList.append(edgeRing.lineString)
        logger.debug("Polygonizer._findValidRings() :%.4f seconds", (time.time() - t))

    def _findShellsAndHoles(self, edgeRingList):
        t = time.time()
        self.holeList.clear()
        self.exteriorList.clear()
        for edgeRing in edgeRingList:
            if edgeRing.isHole:
                self.holeList.append(edgeRing)
            else:
                self.exteriorList.append(edgeRing)
        logger.debug("Polygonizer._findShellsAndHoles() :%.4f seconds", (time.time() - t))

    def _assignHolesToShells(self, holeList, exteriorList):
        t = time.time()
        for hole in holeList:
            self._assignHoleToShell(hole, exteriorList)
        logger.debug("Polygonizer._assignHolesToShells() :%.4f seconds", (time.time() - t))

    def _assignHoleToShell(self, hole, exteriorList):
        exterior = EdgeRing.findEdgeRingContaining(hole, exteriorList)
        if exterior is not None:
            exterior.addHole(hole.getLinearRing())


class PolygonizeOp():

    @staticmethod
    def polygonize_full(geoms: list, skip_validity_check=False):
        op = Polygonizer(skip_validity_check)
        op.addGeometryList(geoms)
        dangles = op.getDangles()
        cuts = op.getCutEdges()
        invalids = op.getInvalidRingLines()
        result = op.getPolygons()
        return result, dangles, cuts, invalids

    @staticmethod
    def polygonize(geoms: list, skip_validity_check=False):
        op = Polygonizer(skip_validity_check)
        op.addGeometryList(geoms)
        result = op.getPolygons()
        return result
