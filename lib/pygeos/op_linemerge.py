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
from .planargraph import (
    PlanarGraph,
    Node,
    Edge,
    DirectedEdge,
    GraphComponent
    )
from .shared import (
    logger,
    GeomTypeId,
    GeometryComponentFilter,
    CoordinateSequence
    )


class LMGeometryComponentFilter(GeometryComponentFilter):

    def __init__(self, lm):
        # LineMerger
        self.lm = lm

    def filter(self, geom):
        if geom.type_id == GeomTypeId.GEOS_LINESTRING:
            self.lm.addLineString(geom)


class LineSequencer():
    """
     * Builds a sequence from a set of LineStrings so that
     * they are ordered end to end.
     *
     * A sequence is a complete non-repeating list of the linear
     * components of the input.  Each linestring is oriented
     * so that identical endpoints are adjacent in the list.
     *
     * A typical use case is to convert a set of
     * unoriented geometric links
     * from a linear network
     * (e.g. such as block faces on a bus route)
     * into a continuous oriented path through the network.
     *
     * The input linestrings may form one or more connected sets.
     * The input linestrings should be correctly noded, or the results may
     * not be what is expected.
     * The computed output is a single MultiLineString containing the ordered
     * linestrings in the sequence.
     *
     * The sequencing employs the classic <b>Eulerian path</b> graph algorithm.
     * Since Eulerian paths are not uniquely determined,
     * further rules are used to
     * make the computed sequence preserve as much as possible of the input
     * ordering.
     * Within a connected subset of lines, the ordering rules are:
     *
     * - If there is degree-1 node which is the start
     *   node of an linestring, use that node as the start of the sequence
     * - If there is a degree-1 node which is the end
     *   node of an linestring, use that node as the end of the sequence
     * - If the sequence has no degree-1 nodes, use any node as the start
     *
     * Note that not all arrangements of lines can be sequenced.
     * For a connected set of edges in a graph,
     * <i>Euler's Theorem</i> states that there is a sequence
     * containing each edge once
     * <b>if and only if</b> there are no more than 2 nodes of odd degree.
     * If it is not possible to find a sequence, the isSequenceable method
     * will return false.
     *
    """
    def __init__(self):
        # LineMergeGraph
        self._graph = LineMergerGraph()
        # GeometryFactory
        self._factory = None
        # int
        self._lineCount = 0
        # bool
        self._isRun = False
        self._isSequencable = False
        self._sequencedGeometry = None

    def _addLine(self, lineString):
        if self._factory is None:
            self._factory = lineString._factory
        self._graph.addEdge(lineString)
        self._lineCount += 1

    def _computeSequence(self):
        if self._isRun:
            return
        self._isRun = True
        sequences = self._findSequences()

        if sequences is None:
            return

        self._sequencedGeometry = self._buildSequencedGeometry(sequences)
        self._isSequencable = True

        # Lines where missing from result
        if self._lineCount != len(self._sequencedGeometry):
            raise RuntimeError("Lines are missing from result")

    def _findSequences(self):
        # DirectedEdge
        sequences = []

        # csFinder = ConnectedSubgraphFinder(subGraph)

        subgraphs = []
        # csFinder.getConnectedSubGraphs(subgraphs)

        for sub in subgraphs:
            if self.hasSequence(sub):
                seq = self._findSequence(sub)
                sequences.append(seq)
            else:
                return None

        return sequences

    def _findSequence(self, subGraph):
        GraphComponent.setVisited(subGraph, False)
        # Node
        startNode = self._findLowestDegreeNode(subGraph)

        # DirectedEdge
        startDE = startNode.deStar.edges[0]
        startDESym = startDE.sym
        seq = []
        self._addReversedSubPath(startDESym, seq, seq[0], False)
        for prev in reversed(seq):
            # DirectedEdge
            unvisitedOutDE = self._findUnvisitedBestOrientedDE(prev._from)
            if unvisitedOutDE is not None:
                self._addReversedSubPath(unvisitedOutDE.sym, seq, prev, True)

        # At this point, we have a valid sequence of graph DirectedEdges,
        # but it is not necessarily appropriately oriented relative to
        # the underlying geometry.

        # DirectedEdge
        return self._orient(seq)

    def _delAll(self, sequences):
        sequences.clear()

    @staticmethod
    def reverse(lineString):
        return lineString._factory.createLineString(lineString.points[::-1])

    def _buildSequencedGeometry(self, sequences):
        """
         * Builds a geometry ({LineString} or {MultiLineString} )
         * representing the sequence.
         *
         * @param sequences
         *    a vector of vectors of const planarDirectedEdges
         *    with LineMergeEdges as their parent edges.
         *    Ownership of container _and_ contents retained by caller.
         *
         * @return the sequenced geometry, possibly NULL
         *         if no sequence exists
        """
        lines = []
        for seq in sequences:
            for de in seq:
                # LineString
                line = de.edge._line
                # lineToAdd will be a *copy* of input things
                # LineString*
                lineToAdd = None
                if not de.edgeDirection and not line.isClosed:
                    lineToAdd = LineSequencer._reverse(line)
                else:
                    lineToAdd = line.clone()
                lines.append(lineToAdd)

        if len(lines) == 0:
            return None
        else:
            return self._factory.buildGeometry(lines)

    @staticmethod
    def _findLowestDegreeNode(subGraph):
        """

        """
        minDegree = 1e64
        # Node
        minDegreeNode = None
        for node in subGraph.getNodes():
            if minDegreeNode is None or node.degree < minDegree:
                minDegree = node.degree
                minDegreeNode = node

        return minDegreeNode

    def _addReversedSubPath(self, de, deList, lit, expectedClosed):
            # Node
            endNode = de._to
            fromNode = None
            while (True):
                deList.extend([lit, de.sym])
                de.edge.visited = True
                fromNode = de._from
                # DirectedEdge
                unvisitedOutDE = LineSequencer._findUnvisitedBestOrientedDE(fromNode)
                # this must terminate, since we are continually marking edges as visited
                if unvisitedOutDE is None:
                    break
                de = unvisitedOutDE.sym
            if expectedClosed:
                # the path should end at the toNode of this de,
                # otherwise we have an error
                if fromNode is not endNode:
                    raise RuntimeError("path not contiguos")

    @staticmethod
    def _findUnvisitedBestOrientedDE(node):
        """
         * Finds an {DirectedEdge} for an unvisited edge (if any),
         * choosing the dirEdge which preserves orientation, if possible.
         *
         * @param node the node to examine
         * @return the dirEdge found, or null
         *         if none were unvisited
        """
        # DirectedEdge
        wellOrientedDE = None
        unvisitedDE = None

        # DirectedEdgeStar
        star = node.deStar
        for de in star.edges:
            if not de.edge.isVisited:
                unvisitedDE = de
                if de.edgeDirection:
                    wellOrientedDE = de

        if wellOrientedDE is not None:
            return wellOrientedDE

        return unvisitedDE

    def _orient(self, seq):
        """
         * Computes a version of the sequence which is optimally
         * oriented relative to the underlying geometry.
         *
         * Heuristics used are:
         *
         * - If the path has a degree-1 node which is the start
         *   node of an linestring, use that node as the start of the sequence
         * - If the path has a degree-1 node which is the end
         *   node of an linestring, use that node as the end of the sequence
         * - If the sequence has no degree-1 nodes, use any node as the start
         *   (NOTE: in this case could orient the sequence according to the
         *   majority of the linestring orientations)
         *
         * @param seq a List of planarDirectedEdges
         * @return the oriented sequence, possibly same as input if already
         *         oriented
        """
        # DirectedEdge
        startEdge = seq[0]
        endEdge = seq[-1]

        # Node
        startNode = startEdge._from
        endNode = endEdge._to

        # bool
        flipSeq = False
        hasDegreeNode = startNode.degree == 1 or endNode.degree == 1

        if hasDegreeNode:

            hasObviousStartNode = False

            # test end edge before start edge, to make result stable
            # (ie. if both are good starts, pick the actual start
            if endEdge._to.degree == 1 and not endEdge.edgeDirection:
                hasObviousStartNode = True
                flipSeq = True

            if startEdge._from.degree == 1 and startEdge.edgeDirection:
                hasObviousStartNode = True
                flipSeq = False

            # since there is no obvious start node,
            # use any node of degree 1
            if not hasObviousStartNode:
                # check if the start node should actually
                # be the end node
                if startEdge._from.degree == 1:
                    flipSeq = True
                # if the end node is of degree 1, it is
                # properly the end node

        # if there is no degree 1 node, just use the sequence as is
        # (Could insert heuristic of taking direction of majority of
        # lines as overall direction)
        if flipSeq:
            return reversed(seq)
        return seq

    def _reverse(self, seq):
        """
         * Reverse the sequence.
         * This requires reversing the order of the dirEdges, and flipping
         * each dirEdge as well
         *
         * @param seq a List of DirectedEdges, in sequential order
         * @return the reversed sequence
        """
        # DirectedEdge
        return reversed([de.sym for de in seq])

    def _hasSequence(self, subgraph):
        """
         * Tests whether a complete unique path exists in a graph
         * using Euler's Theorem.
         *
         * @param graph the subgraph containing the edges
         * @return true if a sequence exists
        """
        oddDegreeCount = 0
        for node in subgraph.getNodes():
            if (node.degree % 2) == 1:
                oddDegreeCount += 1

        return oddDegreeCount <= 2

    @staticmethod
    def sequence(geom):
        sequencer = LineSequencer()
        sequencer.add(geom)
        return sequencer.getSequencedLineStrings()

    @staticmethod
    def isSequenced(geom):
        """
         * Tests whether a {Geometry} is sequenced correctly.
         * {@llink LineString}s are trivially sequenced.
         * {MultiLineString}s are checked for correct sequencing.
         * Otherwise, isSequenced is defined
         * to be true for geometries that are not lineal.
         *
         * @param geom the geometry to test
         * @return true if the geometry is sequenced or is not lineal
        """
        mls = geom
        # the nodes in all subgraphs which have been completely scanned
        # Coordinate.ConstSet
        prevSubgraphNodes = []
        currNodes = []
        lastNode = None
        for lineString in mls.geoms:

            startNode = lineString.points[0]
            endNode = lineString.points[-1]
            """
             * If this linestring is connected to a previous subgraph,
             * geom is not sequenced
            """
            if prevSubgraphNodes.find(startNode) != prevSubgraphNodes[-1]:
                return False

            if lastNode is not None:
                prevSubgraphNodes.extend([currNodes[0], currNodes[-1]])
                currNodes.clear()

            currNodes.append(startNode)
            currNodes.append(endNode)
            lastNode = endNode

        return True

    def isSequenceable(self):
        """
         * Tests whether the arrangement of linestrings has a valid
         * sequence.
         *
         * @return true if a valid sequence exists.
        """
        self._computeSequence()
        return self._isSequenceable

    def add(self, geometry):
        """
         * Adds a {Geometry} to be sequenced.
         * May be called multiple times.
         * Any dimension of Geometry may be added; the constituent
         * linework will be extracted.
         *
         * @param geometry the geometry to add
        """

    def getSequencedLineStrings(self, release=1):
        """
         * Returns the LineString or MultiLineString
         * built by the sequencing process, if one exists.
         *
         * @param release release ownership of computed Geometry
         * @return the sequenced linestrings,
         *         or null if a valid sequence
         *         does not exist.
        """
        self._computeSequence()
        res = self._sequencedGeometry
        if release:
            self._sequencedGeometry = None
        return res


class LineMergeDirectedEdge(DirectedEdge):
    """
     * A planargraph.DirectedEdge of a LineMergeGraph.
    """
    def __init__(self, newFrom, newTo, directionPt, edgeDirection):
        """
         * Constructs a LineMergeDirectedEdge connecting the from
         * node to the to node.
         *
         * @param directionPt
         *        specifies this DirectedEdge's direction (given by an
         *    imaginary line from the from node to
         *    directionPt)
         *
         * @param edgeDirection
         *        whether this DirectedEdge's direction is the same as or
         *        opposite to that of the parent Edge (if any)
        """
        DirectedEdge.__init__(self, newFrom, newTo, directionPt, edgeDirection)

    @property
    def next(self):
        """
         * Returns the directed edge that starts at this directed edge's end point, or null
         * if there are zero or multiple directed edges starting there.
         * @return LineMergeDirectedEdge or None
        """
        if self._to.degree != 2:
            return None
        nextEdge = self._to.deStar.edges[0]
        if nextEdge is self.sym:
            return self._to.deStar.edges[1]
        return nextEdge


class LineMergeEdge(Edge):
    """
     * An edge of a LineMergeGraph. The marked field indicates
     * whether this Edge has been logically deleted from the graph.
    """
    def __init__(self, line):
        Edge.__init__(self)
        # LineString
        self._line = line


class EdgeString():
    """
     * A sequence of LineMergeDirectedEdge forming one of the lines that will
     * be output by the line-merging process.
    """
    def __init__(self, newFactory):
        self._factory = newFactory
        # LineMergeDirectedEdge
        self._directedEdges = []
        # CoordinateSequence
        self._coords = None

    @property
    def coords(self):
        if self._coords is None:
            forwardDirectedEdges = 0
            reverseDirectedEdges = 0
            self._coords = CoordinateSequence()
            for de in self._directedEdges:
                if de.edgeDirection:
                    forwardDirectedEdges += 1
                else:
                    reverseDirectedEdges += 1
                # LineMergeEdge
                lme = de.edge
                self._coords.add(lme._line.coords, allowRepeated=False, direction=de.edgeDirection)

            if reverseDirectedEdges > forwardDirectedEdges:
                self._coords.reverse()
        return self._coords

    def add(self, directedEdge):
        """
         * Adds a directed edge which is known to form part of this line.
        """
        self._directedEdges.append(directedEdge)

    def toLineString(self):
        """
         * Converts this EdgeString into a LineString.
        """
        return self._factory.createLineString(self.coords)


class LineMergerGraph(PlanarGraph):
    """
     * A planar graph of edges that is analyzed to sew the edges together.
     *
     * The marked flag on planargraph.Edge
     * and planargraph.Node indicates whether they have been
     * logically deleted from the graph.
    """
    def __init__(self):
        PlanarGraph.__init__(self)
        # Node
        self._newNodes = []
        # Edge
        self._newEdges = []
        # DirectedEdge
        self._newDirEdges = []

    def _getNode(self, coord):
        # return Node
        node = self.findNode(coord)
        if node is None:
            node = Node(coord)
            self._newNodes.append(node)
            self.addNode(node)
        return node

    def addEdge(self, lineString):
        """
         * Adds an Edge, DirectedEdges, and Nodes for the given
         * LineString representation of an edge.
         *
         * Empty lines or lines with all coordinates equal are not added.
         *
         * @param lineString the linestring to add to the graph
        """
        if lineString.is_empty:
            return

        # CoordinateSequence
        # should check for repeated points
        coords = lineString.coords
        nCoords = len(coords)

        # don't add lines with all coordinates equal
        if nCoords <= 1:
            return

        # Coordinate
        startCoordinate = coords[0]
        endCoordinate = coords[-1]

        # Node
        startNode = self._getNode(startCoordinate)
        endNode = self._getNode(endCoordinate)

        # DirectedEdge
        de0 = LineMergeDirectedEdge(startNode, endNode, coords[1], True)
        self._newDirEdges.append(de0)
        de1 = LineMergeDirectedEdge(endNode, startNode, coords[-2], False)
        self._newDirEdges.append(de1)
        # Edge
        edge = LineMergeEdge(lineString)
        self._newEdges.append(edge)
        edge.setDirectedEdges(de0, de1)
        super(LineMergerGraph, self).addEdge(edge)


class LineMerger():
    """
     * Sews together a set of fully noded LineStrings.
     *
     * Sewing stops at nodes of degree 1 or 3 or more.
     * The exception is an isolated loop, which only has degree-2 nodes,
     * in which case a node is simply chosen as a starting point.
     * The direction of each merged LineString will be that of the majority
     * of the LineStrings from which it was derived.
     *
     * Any dimension of Geometry is handled.
     * The constituent linework is extracted to form the edges.
     * The edges must be correctly noded; that is, they must only meet
     * at their endpoints.
     *
     * The LineMerger will still run on incorrectly noded input
     * but will not form polygons from incorrected noded edges.
    """
    def __init__(self):

        # GeometryFactory
        self._factory = None

        self._graph = LineMergerGraph()
        # LineString
        self._mergedLineStrings = None
        # EdgeString
        self._edgeStrings = []

    def _merge(self):
        if self._mergedLineStrings is not None:
            return

        GraphComponent.setMarkedMap(self._graph._newNodes, False)
        GraphComponent.setMarkedMap(self._graph._newEdges, False)

        self._edgeStrings.clear()
        self._buildEdgeStringsForObviousStartNodes()
        self._buildEdgeStringsForIsolatedLoops()

        self._mergedLineStrings = [es.toLineString() for es in self._edgeStrings]

    def _buildEdgeStringsForObviousStartNodes(self):
        self._buildEdgeStringsForNonDegree2Nodes()

    def _buildEdgeStringsForIsolatedLoops(self):
        self._buildEdgeStringsForUnprocessedNodes()

    def _buildEdgeStringsForUnprocessedNodes(self):
        # Nodes
        nodes = self._graph.nodes
        for node in nodes:
            if not node.marked:
                self._buildEdgeStringsStartingAt(node)
                node.marked = True

    def _buildEdgeStringsForNonDegree2Nodes(self):
        # Nodes
        nodes = self._graph.nodes
        for node in nodes:
            if node.degree != 2:
                self._buildEdgeStringsStartingAt(node)
                node.marked = True

    def _buildEdgeStringsStartingAt(self, node):
        """
         @param node: planargraph.Nod
        """
        # DirectedEdge
        edges = node.deStar.edges
        for de in edges:
            if de.edge.marked:
                continue
            self._edgeStrings.append(self._buildEdgeStringStartingWith(de))

    def _buildEdgeStringStartingWith(self, start):
        """
         @param start :  LineMergeDirectedEdge
        """
        # EdgeString
        es = EdgeString(self._factory)

        # LineMergeDirectedEdge
        current = start

        es.add(current)
        current.edge.marked = True
        current = current.next

        while (current is not None and current is not start):
            es.add(current)
            current.edge.marked = True
            current = current.next

        return es

    def add(self, geoms):
        """
         * Adds a collection of Geometries to be processed.
         * May be called multiple times.
         *
         * Any dimension of Geometry may be added; the constituent
         * linework will be extracted.
        """
        try:
            iter(geoms)
        except TypeError:
            return self.addGeometry(geoms)
            pass

        for geom in geoms:
            self.addGeometry(geom)

    def addGeometry(self, geom):
        """
         * Adds a Geometry to be processed.
         * May be called multiple times.
         *
         * Any dimension of Geometry may be added; the constituent
         * linework will be extracted.
        """
        lmgcf = LMGeometryComponentFilter(self)
        geom.applyComponentFilter(lmgcf)

    def getMergedLineStrings(self):
        """
         * Returns the LineStrings built by the merging process.
         *
         * Ownership of vector _and_ its elements to caller.
        """
        self._merge()
        ret = [ls for ls in self._mergedLineStrings if ls is not None]
        self._mergedLineStrings = None
        return ret

    def addLineString(self, lineString):
        if self._factory is None:
            self._factory = lineString._factory
        self._graph.addEdge(lineString)

    @staticmethod
    def merge(geoms):
        t = time.time()
        lm = LineMerger()
        lm.add(geoms)
        merged = lm.getMergedLineStrings()
        logger.debug("Linemerger.merge() %.2f seconds", time.time() - t)
        return merged
