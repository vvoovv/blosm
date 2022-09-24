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
from math import atan2
import logging
logger = logging.getLogger("pygeos.geomgraph")
from .shared import (
    quicksort,
    GeomTypeId,
    TopologyException,
    Location,
    Position,
    Quadrant,
    CoordinateSequence,
    Envelope
    )
from .algorithms import (
    CGAlgorithms,
    LineIntersector,
    SegmentIntersector,
    MonotoneChainEdge,
    SimpleMCSweepLineIntersector,
    BoundaryNodeRule,
    SimplePointInAreaLocator
    )
from .noding import (
    BasicSegmentString,
    FastNodingValidator,
    OrientedCoordinateArray
    )


class GeometryGraphOperation():

    def __init__(self, g0, g1=None, boundaryNodeRule=None):

        self._li = LineIntersector()
        # geomgraph.GeometryGraph
        self.arg = []

        if boundaryNodeRule is None:
            boundaryNodeRule = BoundaryNodeRule.getBoundaryOGCSFS()

        # PrecisionModel
        self._pm0 = g0.precisionModel
        self.arg.append(GeometryGraph(0, g0, boundaryNodeRule))

        if g1 is not None:
            self._pm1 = g1.precisionModel
            self.arg.append(GeometryGraph(1, g1, boundaryNodeRule))


class TopologyLocation():
    """
     * A TopologyLocation is the labelling of a
     * GraphComponent's topological relationship to a single Geometry.
     *
     * If the parent component is an area edge, each side and the edge itself
     * have a topological location.  These locations are named
     *
     *  - ON: on the edge
     *  - LEFT: left-hand side of the edge
     *  - RIGHT: right-hand side
     *
     * If the parent component is a line edge or node, there is a single
     * topological relationship attribute, ON.
     *
     * The possible values of a topological location are
     * {Location.UNDEF, Location.EXTERIOR, Location.BOUNDARY, Location.INTERIOR}
     *
     * The labelling is stored in an array location[j] where
     * where j has the values ON, LEFT, RIGHT
    """
    def __init__(self, newLocation, left=None, right=None):
        """
         * Constructs a TopologyLocation specifying how points on, to the
         * left of, and to the right of some GraphComponent relate to some
         * Geometry.
         *
         * Possible values for the
         * parameters are Location.UNDEF, Location.EXTERIOR, Location.BOUNDARY,
         * and Location.INTERIOR.
         *
         * @see Location
        """
        if type(newLocation).__name__ == 'TopologyLocation':
            self.location = [loc for loc in newLocation.location]
        else:
            self.location = [newLocation]
            if left is not None:
                self.location.append(left)
            if right is not None:
                self.location.append(right)

    def flip(self) -> None:
        if len(self.location) > 1:
            tmp = self.location[Position.LEFT]
            self.location[Position.LEFT] = self.location[Position.RIGHT]
            self.location[Position.RIGHT] = tmp

    def get(self, posIndex: int) -> int:
        if posIndex < len(self.location):
            return self.location[posIndex]
        return Location.UNDEF

    def setLocations(self, on: int, left: int, right: int) -> None:
        self.location[Position.ON] = on
        self.location[Position.LEFT] = left
        self.location[Position.RIGHT] = right

    def setLocation(self, posIndex: int, location=None) -> None:
        if location is None:
            self.location[Position.ON] = posIndex
        else:
            self.location[posIndex] = location

    def setAllLocationsIfNull(self, location: int) -> None:
        for i, loc in enumerate(self.location):
            if loc == Location.UNDEF:
                self.location[i] = location

    def setAllLocations(self, location: int) -> None:
        self.location = [location for i in range(len(self.location))]

    def merge(self, other) -> None:
        """
         * merge updates only the UNDEF attributes of this object
         * with the attributes of another.
        """
        sz = len(self.location)
        osz = len(other.location)
        # if the src is an Area label & and the dest is not, increase the dest to be an Area
        if osz > sz:
            self.location.extend([Location.UNDEF, Location.UNDEF])

        for i, loc in enumerate(other.location):
            if self.location[i] == Location.UNDEF:
                self.location[i] = loc

    def allPositionsEqual(self, location: int) -> bool:
        for loc in self.location:
            if loc != location:
                return False
        return True

    def isEqualOnSide(self, other, locIndex: int) -> bool:
        return self.location[locIndex] == other.location[locIndex]

    @property
    def isAnyNull(self) -> bool:
        """
         * @return true if any locations is Location.UNDEF
        """
        for loc in self.location:
            if loc == Location.UNDEF:
                return True
        return False

    @property
    def isNull(self) -> bool:
        """
         * @return true if all locations are Location.UNDEF
        """
        for loc in self.location:
            if loc != Location.UNDEF:
                return False
        return True

    @property
    def isArea(self) -> bool:
        return len(self.location) > 1

    @property
    def isLine(self) -> bool:
        return len(self.location) == 1

    def __str__(self) -> str:
        if len(self.location) > 1:
            return "[{}] {} {} {}".format(
                id(self),
                Location.toLocationSymbol(self.location[Position.LEFT]),
                Location.toLocationSymbol(self.location[Position.ON]),
                Location.toLocationSymbol(self.location[Position.RIGHT])
                )
        else:
            return "[{}] - {} -".format(
                id(self),
                Location.toLocationSymbol(self.location[Position.ON])
                )


class Depth():
    """
    """
    NULL_VALUE = -1

    def __init__(self):
        # initialize depth array to a sentinel value
        self._depth = [
            [Depth.NULL_VALUE, Depth.NULL_VALUE, Depth.NULL_VALUE],
            [Depth.NULL_VALUE, Depth.NULL_VALUE, Depth.NULL_VALUE]]

    @staticmethod
    def depthAtLocation(location: int) -> int:
        if location == Location.EXTERIOR:
            return 0
        if location == Location.INTERIOR:
            return 1
        return Depth.NULL_VALUE

    def add(self, geomIndex, posIndex=None, location=None) -> None:
        logger.debug("Depth.add(%s %s %s) at call time:%s", geomIndex, posIndex, location, self)

        if posIndex is None:
            # geomIndex is a Label
            for i in range(2):
                for j in range(1, 3):
                    loc = geomIndex.getLocation(i, j)
                    if loc == Location.EXTERIOR or loc == Location.INTERIOR:
                        if self.isNull(i, j):
                            self._depth[i][j] = Depth.depthAtLocation(loc)
                        else:
                            self._depth[i][j] += Depth.depthAtLocation(loc)
        elif location == Location.INTERIOR:
            self._depth[geomIndex][posIndex] += 1

        logger.debug("Depth.add() reslut:%s", self)

    def isNull(self, geomIndex=None, posIndex=None) -> bool:
        """
         * A Depth object is null (has never been initialized) if all depths are null.
        """
        if geomIndex is None:
            for i in range(2):
                for j in range(3):
                    if self._depth[i][j] != Depth.NULL_VALUE:
                        return False
            return True

        elif posIndex is None:
            posIndex = 1

        return self._depth[geomIndex][posIndex] == Depth.NULL_VALUE

    def getDepth(self, geomIndex: int, posIndex: int) -> int:
        return self._depth[geomIndex][posIndex]

    def setDepth(self, geomIndex: int, posIndex: int, depthValue) -> None:
        self._depth[geomIndex][posIndex] = depthValue

    def getLocation(self, geomIndex: int, posIndex: int) -> int:
        if self._depth[geomIndex][posIndex] <= 0:
            return Location.EXTERIOR
        return Location.INTERIOR

    def normalize(self) -> None:
        """
         * Normalize the depths for each geometry, if they are non-null.
         * A normalized depth
         * has depth values in the set { 0, 1 }.
         * Normalizing the depths
         * involves reducing the depths by the same amount so that at least
         * one of them is 0.  If the remaining value is > 0, it is set to 1.
        """
        for i in range(2):
            if not self.isNull(i):
                minDepth = self._depth[i][1]
                if self._depth[i][2] < minDepth:
                    minDepth = self._depth[i][2]
                if minDepth < 0:
                    minDepth = 0
                for j in range(1, 3):
                    newValue = 0
                    if self._depth[i][j] > minDepth:
                        newValue = 1
                    self._depth[i][j] = newValue

    def getDelta(self, geomIndex: int) -> int:
        return self._depth[geomIndex][Position.RIGHT] - self._depth[geomIndex][Position.LEFT]

    def __str__(self) -> str:
        _d = self._depth
        return "A:{}, {} B:{}, {}".format(_d[0][1], _d[0][2], _d[1][1], _d[1][2])


class Label():
    """
     * A Label indicates the topological relationship of a component
     * of a topology graph to a given Geometry.
     * This class supports labels for relationships to two Geometrys,
     * which is sufficient for algorithms for binary operations.
     *
     * Topology graphs support the concept of labeling nodes and edges in the graph.
     * The label of a node or edge specifies its topological relationship to one or
     * more geometries.  (In fact, since JTS operations have only two arguments labels
     * are required for only two geometries).  A label for a node or edge has one or
     * two elements, depending on whether the node or edge occurs in one or both of the
     * input Geometrys.  Elements contain attributes which categorize the
     * topological location of the node or edge relative to the parent
     * Geometry; that is, whether the node or edge is in the interiors,
     * boundary or exterior of the Geometry.  Attributes have a value
     * from the set {Interior, Boundary, Exterior}.  In a node each
     * element has  a single attribute &lt;On&gt;.  For an edge each element has a
     * triplet of attributes &lt;Left, On, Right&gt;.
     *
     * It is up to the client code to associate the 0 and 1 TopologyLocations
     * with specific geometries.
    """
    def __init__(self, geomIndex=None, onLoc=None, left=None, right=None):
        """
            Construct a label
            arguments : [none | Label | (geomIndex, onLoc) | (onLoc, leftLoc, rightLoc)]
        """
        if right is not None:
            # Area label
            self._elt = [
                TopologyLocation(Location.UNDEF, Location.UNDEF, Location.UNDEF),
                TopologyLocation(Location.UNDEF, Location.UNDEF, Location.UNDEF)]
            self._elt[geomIndex].setLocations(onLoc, left, right)

        elif left is not None:
            # Area label
            # construct using (onLoc, leftLoc, rightLoc)
            self._elt = [
                TopologyLocation(geomIndex, onLoc, left),
                TopologyLocation(geomIndex, onLoc, left)]

        elif type(geomIndex).__name__ == 'Label':
            # construct using Label
            self._elt = [
                TopologyLocation(geomIndex._elt[0]),
                TopologyLocation(geomIndex._elt[1])]
        else:
            # construct using None or (geomIndex, onLoc)
            self._elt = [
                TopologyLocation(Location.UNDEF),
                TopologyLocation(Location.UNDEF)]
            # handle (geomIndex, onLoc)
            if geomIndex is not None and onLoc is not None:
                self._elt[geomIndex].setLocation(onLoc)

    def flip(self) -> None:
        self._elt[0].flip()
        self._elt[1].flip()

    @staticmethod
    def toLineLabel(label):
        """
         * Converts a Label to a Line label
         * (that is, one with no side Locations)
        """
        lineLabel = Label()
        for i in range(2):
            lineLabel.setLocation(i, label.getLocation(i))
        return lineLabel

    def toLine(self, geomIndex: int) -> None:
        if self._elt[geomIndex].isArea:
            self._elt[geomIndex] = TopologyLocation(self._elt[geomIndex].location[Position.ON])

    def merge(self, label) -> None:
        """
         * Merge this label with another one.
         *
         * Merging updates any null attributes of this label with the attributes
         * from label

        """
        for i in range(2):
            self._elt[i].merge(label._elt[i])

    @property
    def geometryCount(self) -> int:
        count = 0
        if not self._elt[0].isNull:
            count += 1
        if not self._elt[1].isNull:
            count += 1
        return count

    def isEqualOnSide(self, label, side: int) -> bool:
        return (self._elt[0].isEqualOnSide(label._elt[0], side) and
            self._elt[1].isEqualOnSide(label._elt[1], side))

    def allPositionsEqual(self, geomIndex: int, location: int) -> bool:
        return self._elt[geomIndex].allPositionsEqual(location)

    def isNull(self, geomIndex=None) -> bool:
        if geomIndex is None:
            return self._elt[0].isNull and self._elt[1].isNull
        else:
            return self._elt[geomIndex].isNull

    def isArea(self, geomIndex=None) -> bool:
        if geomIndex is None:
            return self._elt[0].isArea or self._elt[1].isArea
        else:
            return self._elt[geomIndex].isArea

    def isLine(self, geomIndex: int) -> bool:
        return self._elt[geomIndex].isLine

    def setLocation(self, geomIndex: int, posIndex: int, location=None) -> None:
        if location is None:
            self._elt[geomIndex].setLocation(Position.ON, posIndex)
        else:
            self._elt[geomIndex].setLocation(posIndex, location)

    def getLocation(self, geomIndex: int, location=None) -> int:
        if location is None:
            return self._elt[geomIndex].get(Position.ON)
        else:
            return self._elt[geomIndex].get(location)

    def isAnyNull(self, geomIndex: int) -> bool:
        return self._elt[geomIndex].isAnyNull

    def setAllLocationsIfNull(self, geomIndex: int, location=None) -> None:
        if location is None:
            self.setAllLocationsIfNull(0, geomIndex)
            self.setAllLocationsIfNull(1, geomIndex)
        else:
            self._elt[geomIndex].setAllLocationsIfNull(location)

    def setAllLocations(self, geomIndex: int, location=None) -> None:
        if location is None:
            self.setAllLocations(0, geomIndex)
            self.setAllLocations(1, geomIndex)
        else:
            self._elt[geomIndex].setAllLocations(location)

    def __str__(self) -> str:
        return "[{}] A:{} B:{}".format(id(self), self._elt[0], self._elt[1])


class PlanarGraph():
    """
     * Represents a directed graph which is embeddable in a planar surface.
     *
     * The computation of the IntersectionMatrix relies on the use of a structure
     * called a "topology graph".  The topology graph contains nodes and edges
     * corresponding to the nodes and line segments of a Geometry. Each
     * node and edge in the graph is labeled with its topological location
     * relative to the source geometry.
     *
     * Note that there is no requirement that points of self-intersection
     * be a vertex.
     * Thus to obtain a correct topology graph, Geometry objects must be
     * self-noded before constructing their graphs.
     *
     * Two fundamental operations are supported by topology graphs:
     *
     *  - Computing the intersections between all the edges and nodes of
     *    a single graph
     *  - Computing the intersections between the edges and nodes of two
     *    different graphs
    """
    def __init__(self, nodeFact=None):

        # Edge
        self.edges = []

        if nodeFact is None:
            nodeFact = NodeFactory()

        self._nodes = NodeMap(nodeFact)

        # DirectedEdges
        self._edgeEnds = []

    @property
    def nodes(self):
        return self._nodes.values()

    def isBoundaryNode(self, geomIndex: int, coord) -> bool:
        # Node
        node = self._nodes.find(coord)

        if node is None:
            return False

        # Label
        label = node.label
        if (not label.isNull()) and label.getLocation(geomIndex) == Location.BOUNDARY:
            return True

        return False

    def add(self, edgeEnd) -> None:
        """
         * Add DirectedEdge to the graph
         * find or add a node and link the edge to the node
         * @param: edgeEnd  DirectedEdge
        """
        self._edgeEnds.append(edgeEnd)
        self._nodes.addEdge(edgeEnd)

    def getNodes(self, nodes) -> None:
        nodes.extend(self._nodes.values())

    def addNode(self, node):
        """
         * Add a Node or create a new one when not found
         * @param: node mixed Node or Coordinate
         * @return Node
        """
        logger.debug("PlanarGraph.addNode(%s)\n", node)
        return self._nodes.addNode(node)

    def find(self, coord):
        """
         * @return the node if found; None otherwise
        """
        return self._nodes.find(coord)

    def addEdges(self, edgesToAdd: list) -> None:
        """
         * Add a set of Edge to the graph.  For each edge two DirectedEdges
         * will be created.  DirectedEdges are NOT linked by this method.
        """
        for edge in edgesToAdd:

            self.edges.append(edge)

            de1 = DirectedEdge(edge, True)
            de2 = DirectedEdge(edge, False)
            de1.sym = de2
            de2.sym = de1

            self.add(de1)
            self.add(de2)

        logger.debug("%s.addEdges(%s)", type(self).__name__, len(self.edges))
        for node in self._nodes.values():
            logger.debug("%s", node)
            for de in node.star.edges:
                logger.debug("%s", de)

    @staticmethod
    def static_linkResultDirectedEdges(nodes: list) -> None:
        # NodeMap
        for node in nodes:
            # DirectedEdgeStar
            star = node.star
            # this might throw an exception
            star.linkResultDirectedEdges()

    def linkResultDirectedEdges(self) -> None:
        # NodeMap
        for node in self.nodes:
            # DirectedEdgeStar
            star = node.star
            # this might throw an exception
            star.linkResultDirectedEdges()

    def linkAllDirectedEdges(self) -> None:
        # NodeMap
        for node in self.nodes:
            # DirectedEdgeStar
            star = node.star
            star.linkAllDirectedEdges()

    def findEdgeEnd(self, edge):
        """
         * Returns the EdgeEnd which has edge e as its base edge
         * (MD 18 Feb 2002 - this should return a pair of edges)
         *
         * @return the edge, if found
         *    null if the edge was not found
        """
        # EdgeEnd
        edgeEnds = self._edgeEnds
        for de in edgeEnds:
            if de.edge is edge:
                return de
        return None

    def findEdge(self, p0, p1):
        """
         * Returns the edge whose first two coordinates are p0 and p1
         *
         * @return the edge, if found
         *    null if the edge was not found
        """
        # Edge
        for edge in self.edges:
            # CoordinateSequence
            coords = edge.coords
            if p0 == coords[0] and p1 == coords[1]:
                return edge
        return None

    def findEdgeInSameDirection(self, p0, p1):
        """
         * Returns the edge which starts at p0 and whose first segment is
         * parallel to p1
         *
         * @return the edge, if found
         *    null if the edge was not found
        """
        for edge in self.edges:
            coords = edge.coords

            if self._matchInSameDirection(p0, p1, coords[0], coords[1]):
                return edge

            if self._matchInSameDirection(p0, p1, coords[-1], coords[-2]):
                return edge

        return None

    def printEdges(self)-> str:
        return "Edges: \n".format("\n".join(
            ["edge {}\n{} {}".format(
                i,
                str(edge),
                str(edge.eiList)
                ) for i, edge in enumerate(self.edges)]))

    def insertEdge(self, edge) -> None:
        # Edge
        self.edges.append(edge)

    def _matchInSameDirection(self, p0, p1, ep0, ep1) -> bool:
        """
         * The coordinate pairs match if they define line segments
         * lying in the same direction.
         *
         * E.g. the segments are parallel and in the same quadrant
         * (as opposed to parallel and opposite!).
        """
        if not (p0 == ep0):
            return False

        if CGAlgorithms.computeOrientation(p0, p1, ep1) == CGAlgorithms.COLLINEAR and \
                Quadrant.from_coords(p0, p1) == Quadrant.from_coords(ep0, ep1):
            return True

        return False

    def __str__(self):
        return "[{}] {}:\nNodes:\n{}\nEdges:\n{}\nDirecedEdges:\n{}".format(
            id(self),
            type(self).__name__,
            str(self._nodes),
            "\n".join([str(edge) for edge in self.edges]),
            "\n".join([str(de) for de in self._edgeEnds]))


class NodeMap(dict):
    """
     * A map of Node, indexed by the coordinate of the node.
    """
    def __init__(self, newNodeFact):
        dict.__init__(self)
        # NodeFactory
        self._factory = newNodeFact

    def addNode(self, newNode):
        """
         * Adds a Node to the map, replacing any that is already at that location.
         * @param newNode : Coordinate or Node
        """
        if issubclass(type(newNode), Node):

            coord = newNode.coord
            node = self.find(coord)

            if node is None:
                node = newNode
                logger.debug("[%s] NodeMap.addNode new node     %s", id(self), node)
                self[(coord.x, coord.y)] = node
            else:
                logger.debug("[%s] NodeMap.addNode found node   %s", id(self), node)
                """
                # debug merge z
                zvals = newNode.getZ()
                for z in zvals:
                    node.addZ(z)
                """
                node.mergeLabel(newNode)

        else:
            coord = newNode
            node = self.find(coord)

            if node is None:
                logger.debug("[%s] NodeMap.addNode new node    %s", id(self), coord)
                node = self._factory.createNode(coord)
                self[(coord.x, coord.y)] = node
            else:
                logger.debug("[%s] NodeMap.addNode found node  %s = %s", id(self), coord, node.coord)
                """
                # add z
                node.addZ(coord.z)
                """
        return node

    def addEdge(self, edgeEnd) -> None:
        coord = edgeEnd.coord
        node = self.addNode(coord)
        node.add(edgeEnd)

    def remove(self, coord):
        """
         * Removes the Node at the given location, and returns it
         * (or null if no Node was there).
        """
        node = self.find(coord)
        if node is not None:
            del self[(coord.x, coord.y)]
        return node

    def find(self, coord):
        """
         * Returns the Node at the given location, or None if no Node was there.
        """
        return self.get((coord.x, coord.y))

    def getBoundaryNodes(self, geomIndex: int) -> list:
        nodes = self.values()
        return [node for node in nodes
            if node.label.getLocation(geomIndex) == Location.BOUNDARY]

    def __str__(self):
        return "\n".join([str(node) for node in self.values()])


def EdgeEndLT(s1, s2) -> bool:
    return s1.compareTo(s2) < 0


class DirectedEdgeMap(dict):

    def __init__(self):
        dict.__init__(self)
        self._edgeList = None

    @property
    def edges(self):
        if self._edgeList is None:
            self._edgeList = list(self.values())
            quicksort(self._edgeList, EdgeEndLT)

        return self._edgeList

    def add(self, de):
        key = (de.coord, de.direction)
        res = self.get(key)
        if res is None:
            res = de
        self[key] = res
        self._edgeList = None


class GraphComponent():

    def __init__(self, newLabel=None):
        if newLabel is None:
            self.label = Label(0, Location.UNDEF)
        else:
            self.label = newLabel

        self.isInResult = False
        self._isCovered = False
        self.isCoveredSet = False
        self.isVisited = False

    @property
    def isCovered(self) -> bool:
        return self._isCovered

    @isCovered.setter
    def isCovered(self, isCovered: bool) -> None:
        self._isCovered = isCovered
        self.isCoveredSet = True

    def computeIM(self, im) -> None:
        raise NotImplementedError()

    def updateIM(self, im) -> None:
        assert(self.label.geometryCount >= 2), "found partial label"
        self.computeIM(im)


class Node(GraphComponent):
    """
     * The node component of a geometry graph
     * @param newCoord Coordinate
     * @param newEdges EdgeEndStar
    """
    def __init__(self, coord, star):
        GraphComponent.__init__(self)
        # Coordinate
        self.coord = coord

        # EdgeEndStar
        self.star = star

    @property
    def isIsolated(self) -> bool:
        return self.label.geometryCount == 1

    def add(self, de) -> None:
        """
         * Add the edgeEnd to the list of edges at this node
        """
        # logger.debug("[%s] Node:add(%s)\n", id(self), de)

        if de.coord != self.coord:
            raise ValueError("EdgeEnd with coordinate {} invalid for node".format(de.coord))

        self.star.insert(de)
        de.node = self
        # self.addZ(edgeEnd.coord.z)

    def mergeLabel(self, other) -> None:
        """
         * To merge labels for two nodes,
         * the merged location for each LabelElement is computed.
         *
         * The location for the corresponding node LabelElement is set
         * to the result, as long as the location is non-null.
        """
        if issubclass(type(other), Node):
            return self.mergeLabel(other.label)

        for i in range(2):
            loc = self.computeMergedLocation(other, i)
            thisLoc = self.label.getLocation(i)
            if thisLoc == Location.UNDEF:
                self.label.setLocation(i, loc)

    def setLabel(self, geomIndex: int, onLocation: int) -> None:
        """
         *
        """
        self.label.setLocation(geomIndex, onLocation)

    def setLabelBoundary(self, geomIndex: int) -> None:
        """
         * Updates the label of a node to BOUNDARY,
         * obeying the mod-2 boundaryDetermination rule.
        """
        loc = self.label.getLocation(geomIndex)
        if loc == Location.BOUNDARY:
            newLoc = Location.INTERIOR

        elif loc == Location.INTERIOR:
            newLoc = Location.BOUNDARY

        else:
            newLoc = Location.BOUNDARY

        self.label.setLocation(geomIndex, newLoc)

    def computeMergedLocation(self, other, geomIndex: int) -> int:
        """
         * The location for a given eltIndex for a node will be one
         * of { null, INTERIOR, BOUNDARY }.
         * A node may be on both the boundary and the interiors of a geometry;
         * in this case, the rule is that the node is considered to be
         * in the boundary.
         * The merged location is the maximum of the two input values.
        """
        loc = self.label.getLocation(geomIndex)
        if loc != Location.BOUNDARY and not other.isNull(geomIndex):
            loc = other.getLocation(geomIndex)
        return loc

    @property
    def isIncidentEdgeInResult(self) -> bool:
        """
         * Tests whether any incident edge is flagged as
         * being in the result.
         *
         * This test can be used to determine if the node is in the result,
         * since if any incident edge is in the result, the node must be in
         * the result as well.
         *
         * @return true if any indicident edge in the in
         *         the result
        """
        if self.star is None:
            return False

        for de in self.star:
            # DirectedEdge
            if de.edge.isInResult:
                return True

        return False

    def testInvariant(self) -> None:
        # Each EdgeEnd in the star has this Node's
        # coordinate as first coordinate
        for de in self.star:
            if de.coord != self.coord:
                raise TopologyException("eInvariant", self.coord)

    def computeIM(self, im) -> None:
        # Basic nodes do not compute IMs
        return

    def __str__(self):
        return "Node[{}] Label:{} Point({})".format(id(self), self.label, self.coord)


class NodeFactory():

    def createNode(self, coord):
        return Node(coord, None)


class EdgeEnd():
    """
     * Models the end of an edge incident on a node.
     *
     * EdgeEnds have a direction
     * determined by the direction of the ray from the initial
     * point to the next point.
     * EdgeEnds are comparable under the ordering
     * "a has a greater angle with the x-axis than b".
     * This ordering is used to sort EdgeEnds around a node.
    """
    def __init__(self, newEdge=None, newP0=None, newP1=None, newLabel=None):

        # Label
        if newLabel is None:
            self.label = Label()
        else:
            self.label = Label(newLabel)

        # Edge: the parent edge of this edge end
        self.edge = newEdge

        # Node: the node this edge end originates at
        self.node = None

        self.dx = 0.0
        self.dy = 0.0
        self.quadrant = 0

        if newP0 is not None and newP1 is not None:
            self.init(newP0, newP1)

    def compareTo(self, edgeEnd) -> int:
        return self.compareDirection(edgeEnd)

    def compareDirection(self, edgeEnd) -> int:
        """
         * Implements the total order relation:
         *
         *    a has a greater angle with the positive x-axis than b
         *
         * Using the obvious algorithm of simply computing the angle
         * is not robust, since the angle calculation is obviously
         * susceptible to roundoff.
         * A robust algorithm is:
         * - first compare the quadrant.  If the quadrants
         *   are different, it it trivial to determine which vector
         *   is "greater".
         * - if the vectors lie in the same quadrant, the
         *   computeOrientation function can be used to decide
         *   the relative orientation of the vectors.
        """
        if self.dx == edgeEnd.dx and self.dy == edgeEnd.dy:
            return 0
        # if the rays are in different quadrants, determining the ordering is trivial
        if self.quadrant > edgeEnd.quadrant:
            return 1
        elif self.quadrant < edgeEnd.quadrant:
            return -1
        # vectors are in the same quadrant - check relative orientation of direction vectors
        # this is > e if it is CCW of e
        return CGAlgorithms.computeOrientation(edgeEnd.coord, edgeEnd.direction, self.direction)

    def computeLabel(self, bnr) -> None:
        # subclasses should override this if they are using labels
        pass

    def init(self, newP0, newP1) -> None:

        # the direction vector for this edge from its starting point
        self.coord = newP0
        self.direction = newP1

        self.dx = newP1.x - newP0.x
        self.dy = newP1.y - newP0.y

        self.quadrant = Quadrant.quadrant(self.dx, self.dy)

    def __str__(self) -> str:
        return "{}: {}:{:.3f} {} {}-{} ".format(
            type(self).__name__,
            self.quadrant,
            atan2(self.dy, self.dx),
            self.label,
            self.coord,
            self.direction)

    def __eq__(self, other) -> bool:
        return self.compareTo(other) == 0

    def __ne__(self, other) -> bool:
        return self.compareTo(other) != 0

    def __gt__(self, other) -> bool:
        return self.compareTo(other) == 1

    def __lt__(self, other) -> bool:
        return self.compareTo(other) == -1


class EdgeRing():
    """
    """
    def __init__(self, newStart, newFactory):

        # the directed edge which starts the list of edges for this EdgeRing
        # DirectedEdge
        self.startDe = newStart

        # GeometryFactory
        self._factory = newFactory

        # label stores the locations of each geometry on the
        # face surrounded by this ring
        self.label = Label(Location.UNDEF)

        # LinearRing
        self._ring = None

        # if non-null, the ring is a hole and this EdgeRing is its containing exterior
        # EdgeRing
        self._exterior = None

        # a list of EdgeRings which are interiors in this EdgeRing
        self._interior = []
        self._isHole = False

        self.coords = newFactory.coordinateSequenceFactory.create()

        self._maxNodeDegree = -1

        # DirectedEdge
        self.edges = []

    @property
    def isIsolated(self) -> bool:
        return self.label.geometryCount == 1

    @property
    def isHole(self) -> bool:
        return self._isHole

    @property
    def isShell(self) -> bool:
        return self._exterior is None

    def getLinearRing(self):
        """
         * Return a pointer to the LinearRing owned by
         * this object. Make a copy if you need it beyond
         * this objects's lifetime.
        """
        return self._ring

    def setShell(self, newShell) -> None:
        self._exterior = newShell
        if self._exterior is not None:
            self._exterior.addHole(self)

    def addHole(self, edgeRing) -> None:
        self._interior.append(edgeRing)

    def toPolygon(self, factory):
        """
         * Return a Polygon copying coordinates from this
         * EdgeRing and its interiors. Caller must remember
         * to delete the result
        """
        interiors = [hole._ring for hole in self._interior]
        return factory.createPolygon(self._ring, interiors)

    def computeRing(self) -> None:
        if self._ring is not None:
            return
        self._ring = self._factory.createLinearRing(self.coords)
        self._isHole = CGAlgorithms.isCCW(self.coords)

    def getNext(self, de):
        raise NotImplementedError()

    def setEdgeRing(self, de, er) -> None:
        raise NotImplementedError()

    @property
    def maxNodeDegree(self) -> int:
        if self._maxNodeDegree < 0:
            self.computeMaxNodeDegree()
        return self._maxNodeDegree

    @property
    def isInResult(self) -> bool:
        return True

    @isInResult.setter
    def isInResult(self, isInResult) -> None:
        de = self.startDe
        while (True):
            de.edge.isInResult = isInResult
            de = de.next
            if de is self.startDe:
                break

    def containsPoint(self, coord) -> bool:
        """
         * This method will use the computed ring.
         * It will also check any interiors, if they have been assigned.
        """
        env = self._ring.envelope

        if not env.contains(coord):
            return False

        if not CGAlgorithms.isPointInRing(coord, self._ring.coords):
            return False

        for hole in self._interior:
            if hole.containsPoint(coord):
                return False

        return True

    def computePoints(self, newStart) -> None:

        self.startDe = newStart
        de = newStart
        isFirstEdge = True
        while (True):

            if de is None:
                raise TopologyException("EdgeRing.computePoints: found null Directed Edge")

            if de.edgeRing is self:
                raise TopologyException(
                    "EdgeRing.computePoints: Directed Edge visited twice during ring-building",
                    de.coord
                    )

            self.edges.append(de)
            deLabel = de.label

            assert (de.label.isArea()), "de label must be area"

            self.mergeLabel(deLabel)
            self.addPoints(de.edge, de.isForward, isFirstEdge)
            isFirstEdge = False
            self.setEdgeRing(de, self)
            de = self.getNext(de)
            if de is self.startDe:
                break

    def mergeLabel(self, deLabel, geomIndex=None) -> None:
        """
         * Merge the RHS label from a DirectedEdge into the label for
         * this EdgeRing.
         *
         * The DirectedEdge label may be null.
         * This is acceptable - it results from a node which is NOT
         * an intersection node between the Geometries
         * (e.g. the end node of a LinearRing).
         * In this case the DirectedEdge label does not contribute any
         * information to the overall labelling, and is
         * simply skipped.
        """
        if geomIndex is None:
            self.mergeLabel(deLabel, 0)
            self.mergeLabel(deLabel, 1)
        else:
            loc = deLabel.getLocation(geomIndex, Position.RIGHT)

            # no information to be had from this label
            if loc == Location.UNDEF:
                return

            # if there is no current RHS value, set it
            if self.label.getLocation(geomIndex) == Location.UNDEF:
                self.label.setLocation(geomIndex, loc)

    def addPoints(self, edge, isForward: bool, isFirstEdge: bool) -> None:
        coords = edge.coords
        nCoords = len(coords)
        if isForward:
            startIndex = 1
            if isFirstEdge:
                startIndex = 0
            self.coords.extend(coords[startIndex:nCoords])
            # for i in range():
            #    self.coords.append(edgePts[i])
        else:
            startIndex = nCoords - 1
            if isFirstEdge:
                startIndex = nCoords
            self.coords.extend(reversed(coords[0:startIndex]))
            # startIndex = nCoords - 2
            # if isFirstEdge:
            #    startIndex = nCoords - 1
            # for i in range(startIndex, 0, -1):
            #    self.coords.append(edgePts[i])

    def computeMaxNodeDegree(self) -> None:
        self._maxNodeDegree = 0
        de = self.startDe
        while (True):
            node = de.node
            # DirectedEdgeStar
            star = node.star
            degree = star.getOutgoingDegree(self)
            if degree > self._maxNodeDegree:
                self._maxNodeDegree = degree
            de = self.getNext(de)
            if de is self.startDe:
                break
        self._maxNodeDegree *= 2

    def __str__(self) -> str:
        return "EdgeRing[{}] Points({})".format(id(self), self.coords)


class EdgeList(list):
    """
     * A EdgeList is a list of Edges.
     *
     * It supports locating edges
     * that are pointwise equals to a target edge.
    """
    def __init__(self):
        list.__init__(self)
        self._ocaMap = {}

    def add(self, edge) -> None:
        """
         * Insert an edge unless it is already in the list
        """
        self.append(edge)
        oca = OrientedCoordinateArray(edge.coords)
        self._ocaMap[hash(oca)] = edge

    def addAll(self, edgeColl) -> None:
        for edge in edgeColl:
            self.add(edge)

    def findEqualEdge(self, edge):
        # noding.OrientedCoordinateArray
        oca = OrientedCoordinateArray(edge.coords)
        found = self._ocaMap.get(hash(oca))
        return found

    def get(self, index: int):
        return self[index]

    def findEdgeIndex(self, edge) -> int:
        try:
            index = self.index(edge)
        except:
            index = -1
            pass
        return index

    def clearList(self) -> None:
        self.clear()

    def __str__(self) -> str:
        return "EdgeList:\n{}".format("\n".join([str(ed) for ed in self]))


class Edge(GraphComponent):
    """
    """
    def __init__(self, coords, label=None):

        GraphComponent.__init__(self, label)

        # MonotoneChainEdge Lazily-created, owned by Edge
        self._mce = None

        # Envelope Lazily-created, owned by Edge
        self._env = None

        # bool
        self.isIsolated = True

        self.depth = Depth()
        self.depthDelta = 0

        # CoordinateSequence
        self.coords = coords

        # EdgeIntersectionList
        self.eiList = EdgeIntersectionList(self)
    
    @property
    def coord(self):
        return self.coords[0]
    
    @property
    def intersections(self):
        return self.eiList.intersections

    @property
    def maximumSegmentIndex(self) -> int:
        return len(self.coords) - 1

    @property
    def monotoneChainEdge(self):
        """
         * Return this Edge's index.MonotoneChainEdge,
         * ownership is retained by this object.
        """
        if self._mce is None:
            self._mce = MonotoneChainEdge(self)
        return self._mce

    @property
    def isClosed(self) -> bool:
        return self.coords[0] == self.coords[-1]

    @property
    def isCollapsed(self) -> bool:
        """
         * An Edge is collapsed if it is an Area edge and it consists of
         * two segments which are equal and opposite (eg a zero-width V).
        """
        return (self.label.isArea() and
            len(self.coords) == 3 and
            self.coords[0] == self.coords[2])

    def getCollapsedEdge(self):
        coords = CoordinateSequence(self.coords[0:2])
        return Edge(coords, Label.toLineLabel(self.label))

    def addIntersections(self, li, segmentIndex: int, geomIndex: int) -> None:
        """
         * Adds EdgeIntersections for one or both
         * intersections found for a segment of an edge to the edge intersection list.
        """
        for i in range(li.intersections):
            self.addIntersection(li, segmentIndex, geomIndex, i)

    def addIntersection(self, li, segmentIndex: int, geomIndex: int, intIndex: int) -> None:
        """
         * Add an EdgeIntersection for intersection intIndex.
         *
         * An intersection that falls exactly on a vertex of the edge is normalized
         * to use the higher of the two possible segmentIndexes
         * @param li LineIntersector
        """

        # Coordinate
        intPt = li.getIntersection(intIndex)
        normalizedSegmentIndex = segmentIndex
        dist = li.getEdgeDistance(geomIndex, intIndex)
        """
        logger.debug("[%s] Edge.addIntersection(%s, geom:%s, seg:%s, int:%s, dist:%s)",
            id(self),
            li,
            geomIndex,
            segmentIndex,
            intIndex,
            dist)
        """
        nextPt = None
        # normalize the intersection point location
        nextSegIndex = normalizedSegmentIndex + 1
        npts = len(self.coords)
        if nextSegIndex < npts:
            # Normalize segment index if intPt falls on vertex
            # The check for point equality is 2D only - Z values are ignored
            nextPt = self.coords[nextSegIndex]
            if intPt == nextPt:
                normalizedSegmentIndex = nextSegIndex
                dist = 0.0
        """
        logger.debug("Edge.addIntersection point %s, next:%s, seg(norm):%s, dist(norm):%s normalize:%s pt==next:%s",
            intPt,
            nextPt,
            normalizedSegmentIndex,
            dist,
            nextSegIndex < npts,
            intPt == nextPt)
        """
        # Add the intersection point to edge intersection list
        self.eiList.add(intPt, normalizedSegmentIndex, dist)

    def computeIM(self, im):
        """
         * Update the IM with the contribution for this component.
         *
         * A component only contributes if it has a labelling for both
         * parent geometries
        """
        self.updateIM(self.label, im)

    def isPointwiseEqual(self, edge) -> bool:
        # return true if the coordinate sequences of the Edges are identical
        return self.coords == edge.coords

    def equals(self, edge)-> bool:
        """
         * equals is defined to be:
         *
         * e1 equals e2
         * <b>iff</b>
         * the coordinates of e1 are the same or the reverse of the coordinates in e2
        """
        return CoordinateSequence.equals_unoriented(self.coords, edge.coords)

    @property
    def envelope(self):

        if self._env is None:

            self._env = Envelope()
            for coord in self.coords:
                self._env.expandToInclude(coord)

        return self._env

    def updateIM(self, lbl, im) -> None:
        """
         * Updates an IM from the label for an edge.
         * Handles edges from both L and A geometrys.
        """
        im.setAtLeastIfValid(lbl.getLocation(0, Position.ON),
                              lbl.getLocation(1, Position.ON),
                              1)
        if lbl.isArea():
            im.setAtLeastIfValid(lbl.getLocation(0, Position.LEFT),
                                  lbl.getLocation(1, Position.LEFT),
                                  2)
            im.setAtLeastIfValid(lbl.getLocation(0, Position.RIGHT),
                                  lbl.getLocation(1, Position.RIGHT),
                                  2)

    def __str__(self) -> str:
        return "{}: label:{} depthDelta:{} coords:{}".format(
            type(self).__name__,
            self.label,
            self.depthDelta,
            self.coords)

    def printReverse(self) -> str:
        return "{} (rev): label:{} depthDelta:{} LineString({})".format(
            type(self).__name__,
            self.label,
            self.depthDelta,
            self.coords.printReverse())


class DirectedEdge(EdgeEnd):
    """
     * A directed EdgeEnd
    """
    def __init__(self, edge, isForward):

        EdgeEnd.__init__(self, edge)

        # bool
        self.isForward = isForward

        self.isInResult = False
        self.isVisited = False

        # DirectedEdge

        # the symmetric edge
        self.sym = None

        # the next edge in the edge ring for the polygon containing this edge
        self.next = None

        # the next edge in the MinimalEdgeRing that contains this edge
        self.nextMin = None

        # EdgeRing

        # the EdgeRing that this edge is part of
        self.edgeRing = None

        # the MinimalEdgeRing that this edge is part of
        self.minEdgeRing = None

        self._depth = [0, -999, -999]

        coords = edge.coords

        # EdgeEnd method
        if isForward:
            self.init(coords[0], coords[1])
        else:
            self.init(coords[-1], coords[-2])

        self._computeDirectedLabel()

    def setVisitedEdge(self, isVisited: bool) -> None:
        self.isVisited = isVisited
        self.sym.isVisited = isVisited

    def setDepth(self, position: int, newDepth: int) -> None:

        if self._depth[position] != -999:
            if self._depth[position] != newDepth:
                raise TopologyException("{} [{}] assigned depths do not match new:{} current:{} Depth:{}/{}".format(
                        type(self).__name__,
                        id(self),
                        newDepth,
                        self._depth[position],
                        self._depth[Position.LEFT],
                        self._depth[Position.RIGHT]
                        ),
                    self.coord)

        self._depth[position] = newDepth

    def getDepth(self, position: int) -> int:
        return self._depth[position]

    @property
    def depthDelta(self) -> int:
        depthDelta = self.edge.depthDelta
        if not self.isForward:
            depthDelta = -depthDelta
        return depthDelta

    @property
    def isLineEdge(self) -> bool:
        """
         * Tells wheter this edge is a Line
         *
         * This edge is a line edge if
         * - at least one of the labels is a line label
         * - any labels which are not line labels have all Locations = EXTERIOR
         *
        """
        label = self.label
        isLine = label.isLine(0) or label.isLine(1)
        isExteriorIfArea0 = (not label.isArea(0)) or label.allPositionsEqual(0, Location.EXTERIOR)
        isExteriorIfArea1 = (not label.isArea(1)) or label.allPositionsEqual(1, Location.EXTERIOR)
        return isLine and isExteriorIfArea0 and isExteriorIfArea1

    @property
    def isInteriorAreaEdge(self) -> bool:
        """
         * Tells wheter this edge is an Area
         *
         * This is an interiors Area edge if
         * - its label is an Area label for both Geometries
         * - and for each Geometry both sides are in the interiors.
         *
         * @return true if this is an interiors Area edge
        """
        label = self.label
        for i in range(2):
            if (not (label.isArea(i) and
                    label.getLocation(i, Position.LEFT) == Location.INTERIOR and
                    label.getLocation(i, Position.RIGHT) == Location.INTERIOR)):
                return False

        return True

    def _computeDirectedLabel(self) -> None:
        # Compute the label in the appropriate orientation for this DirEdge
        self.label = Label(self.edge.label)
        if not self.isForward:
            self.label.flip()

    def setEdgeDepths(self, position: int, depth: int) -> None:
        """
         * Set both edge depths.
         *
         * One depth for a given side is provided.
         * The other is computed depending on the Location transition and the
         * depthDelta of the edge.
        """
        # if moving from L to R instead of R to L must change sign of delta
        if position == Position.LEFT:
            oppositeDepth = depth - self.depthDelta
        else:
            oppositeDepth = depth + self.depthDelta

        oppositePos = Position.opposite(position)
        self.setDepth(position, depth)
        self.setDepth(oppositePos, oppositeDepth)

        logger.debug("DirectedEdge.setEdgeDepths() [%s] Depth:%s/%s",
            id(self),
            self._depth[Position.LEFT],
            self._depth[Position.RIGHT],
            )

    def depthFactor(self, currLocation: int, nextLocation: int) -> int:
        """
         * Computes the factor for the change in depth when moving from
         * one location to another.
         * E.g. if crossing from the INTERIOR to the EXTERIOR the depth
         * decreases, so the factor is -1
        """
        if currLocation == Location.EXTERIOR and nextLocation == Location.INTERIOR:
            return 1
        elif currLocation == Location.INTERIOR and nextLocation == Location.EXTERIOR:
            return -1
        return 0

    def __str__(self) -> str:
        return EdgeEnd.__str__(self)

    def printEdge(self) -> str:
        if self.isForward:
            return str(self.edge)
        else:
            return self.edge.printReverse()


class EdgeEndStar():
    """
     * A EdgeEndStar is an ordered list of EdgeEnds around a node.
     *
     * They are maintained in CCW order (starting with the positive x-axis)
     * around the node for efficient lookup and topology building.
    """
    def __init__(self):
        self._edgeMap = DirectedEdgeMap()
        """
         * The location of the point for this star in
         * Geometry i Areas
        """
        self._ptInAreaLocation = [Location.UNDEF, Location.UNDEF]

    def insert(self, de) -> None:
        """
         * Insert a EdgeEnd into this EdgeEndStar
        """
        raise NotImplementedError()

    @property
    def edges(self):
        return self._edgeMap.edges

    def index(self, de):
        return self.edges.index(de)

    @property
    def coord(self):
        """
         * @return the coordinate for the node this star is based at
         *         or NULL if this is still an unbound star.
         * Be aware that the returned pointer will point to
         * a Coordinate owned by the specific EdgeEnd happening
         * to be the first in the star (ordered CCW)
        """
        if len(self.edges) == 0:
            return None
        return self.edges[0].coord

    def getNextCW(self, de):
        index = self.find(de)
        if index is None or index == len(self.edges) - 1:
            return None
        return self.edges[index - 1]

    def computeLabelling(self, graphs) -> None:

        self._computeEdgeEndLabels(graphs[0].boundaryNodeRule)
        # Propagate side labels  around the edges in the star
        # for each parent Geometry
        # these calls can throw a TopologyException
        self.propagateSideLabels(0)
        self.propagateSideLabels(1)
        """
         * If there are edges that still have null labels for a geometry
         * this must be because there are no area edges for that geometry
         * incident on this node.
         * In this case, to label the edge for that geometry we must test
         * whether the edge is in the interiors of the geometry.
         * To do this it suffices to determine whether the node for the
         * edge is in the interiors of an area.
         * If so, the edge has location INTERIOR for the geometry.
         * In all other cases (e.g. the node is on a line, on a point, or
         * not on the geometry at all) the edge
         * has the location EXTERIOR for the geometry.
         *
         * Note that the edge cannot be on the BOUNDARY of the geometry,
         * since then there would have been a parallel edge from the
         * Geometry at this node also labelled BOUNDARY
         * and this edge would have been labelled in the previous step.
         *
         * This code causes a problem when dimensional collapses are present,
         * since it may try and determine the location of a node where a
         * dimensional collapse has occurred.
         * The point should be considered to be on the EXTERIOR
         * of the polygon, but locate() will return INTERIOR, since it is
         * passed the original Geometry, not the collapsed version.
         *
         * If there are incident edges which are Line edges labelled BOUNDARY,
         * then they must be edges resulting from dimensional collapses.
         * In this case the other edges can be labelled EXTERIOR for this
         * Geometry.
         *
         * MD 8/11/01 - NOT TRUE!  The collapsed edges may in fact be in the
         * interiors of the Geometry, which means the other edges should be
         * labelled INTERIOR for this Geometry.
         * Not sure how solve this...  Possibly labelling needs to be split
         * into several phases:
         * area label propagation, symLabel merging, then finally null label
         * resolution.
        """
        hasDimentionalCollapseEdges = [False, False]
        # EdgeEndStar
        for de in self.edges:
            # EdgeEnd
            label = de.label
            for i in range(2):
                if label.isLine(i) and label.getLocation(i) == Location.BOUNDARY:
                    hasDimentionalCollapseEdges[i] = True

        for de in self.edges:
            label = de.label
            for i in range(2):
                if label.isAnyNull(i):
                    loc = Location.UNDEF
                    if hasDimentionalCollapseEdges[i]:
                        loc = Location.EXTERIOR
                    else:
                        coord = de.coord
                        loc = self._getLocation(i, coord, graphs)
                    label.setAllLocationsIfNull(i, loc)

            logger.debug("EdgeEndStar.computeLabelling()     %s [%s] %s", type(de).__name__, id(de), label)

    def isAreaLabelsConsistent(self, geomGraph) -> bool:
        self._computeEdgeEndLabels(geomGraph.boundaryNodeRule)
        return self._checkAreaLabelsConsistent(0)

    def propagateSideLabels(self, geomIndex: int) -> None:
        # Since edges are stored in CCW order around the node,
        # As we move around the ring we move from the right to the
        # left side of the edge
        startLoc = Location.UNDEF

        # initialize loc to location of last L side (if any)
        for de in self.edges:
            label = de.label
            if label.isArea(geomIndex) and label.getLocation(geomIndex, Position.LEFT) != Location.UNDEF:
                startLoc = label.getLocation(geomIndex, Position.LEFT)

        # no labelled sides found, so no labels to propagate
        if startLoc == Location.UNDEF:
            return

        currLoc = startLoc

        for de in self.edges:
            label = de.label

            # set null ON values to be in current location
            if label.getLocation(geomIndex, Position.ON) == Location.UNDEF:
                label.setLocation(geomIndex, Position.ON, currLoc)

            # set side labels (if any)
            if label.isArea(geomIndex):
                leftLoc = label.getLocation(geomIndex, Position.LEFT)
                rightLoc = label.getLocation(geomIndex, Position.RIGHT)

                # if there is a right location, that is the next
                # location to propagate

                if rightLoc != Location.UNDEF:

                    if rightLoc != currLoc:
                        raise TopologyException(
                            "side location conflict left:{} right:{} != current:{} Label:{}".format(
                                Location.toLocationSymbol(leftLoc),
                                Location.toLocationSymbol(rightLoc),
                                Location.toLocationSymbol(currLoc),
                                label),
                            self.coord
                            )

                    if leftLoc == Location.UNDEF:
                        assert(0), "found single null side at e->getCoordinate()"

                    currLoc = leftLoc
                else:
                    """
                     * RHS is null - LHS must be null too.
                     * This must be an edge from the other
                     * geometry, which has no location
                     * labelling for this geometry.
                     * This edge must lie wholly inside or
                     * outside the other geometry (which is
                     * determined by the current location).
                     * Assign both sides to be the current
                     * location.
                    """
                    assert(label.getLocation(geomIndex, Position.LEFT) == Location.UNDEF), "found single null side"

                    label.setLocation(geomIndex, Position.RIGHT, currLoc)
                    label.setLocation(geomIndex, Position.LEFT, currLoc)
            logger.debug("EdgeEndStar.propagateSideLabels(%s) %s [%s] %s", geomIndex, type(de).__name__, id(de), label)

    def _getLocation(self, geomIndex: int, coord, graphs) -> int:
        # if self._ptInAreaLocation[geomIndex] == Location.UNDEF:
        #    self._ptInAreaLocation[geomIndex] =
        return SimplePointInAreaLocator.locate(coord, graphs[geomIndex].geom)
        # return self._ptInAreaLocation[geomIndex]

    def _computeEdgeEndLabels(self, bnr) -> None:
        for de in self.edges:
            # EdgeEnd
            de.computeLabel(bnr)

    def _checkAreaLabelsConsistent(self, geomIndex: int) -> bool:
        # Since edges are stored in CCW order around the node,
        # As we move around the ring we move from the right to
        # the left side of the edge

        # if no edges, trivially consistent
        if len(self.edges) == 0:
            return True

        # initialize startLoc to location of last L side (if any)
        startLabel = self.edges[0].label
        startLoc = startLabel.getLocation(geomIndex, Position.LEFT)
        assert(startLoc != Location.UNDEF), "Found unlabelled area edge"

        currLoc = startLoc

        for de in self.edges:
            label = de.label
            # we assume that we are only checking a area
            assert(label.isArea(geomIndex)), "Found non-area edge"

            leftLoc = label.getLocation(geomIndex, Position.LEFT)
            rightLoc = label.getLocation(geomIndex, Position.RIGHT)

            # check that edge is really a boundary between inside and outside!
            if leftLoc == rightLoc:
                return False

            # check side location conflict
            if rightLoc != currLoc:
                return False

            currLoc = leftLoc

        return True

    def insertEdgeEnd(self, de) -> None:

        self._edgeMap.add(de)
        """
        logger.debug("[%s] %s.insertEdgeEnd(%s)", id(self), type(self).__name__, len(self.edges))
        for de in self.edges:
            logger.debug("%s", de)
        """

    @property
    def degree(self) -> int:
        return len(self.edges)

    def find(self, de):
        for i, candidate in enumerate(self.edges):
            if candidate == de:
                return i
        return None

    def __str__(self) -> str:
        return "{} {}\n{}".format(
            type(self).__name__,
            self.coord,
            "\n".join([str(de) for de in self.edges]))


class DirectedEdgeStar(EdgeEndStar):
    """
     * A DirectedEdgeStar is an ordered list of <b>outgoing</b> DirectedEdges around a node.
     *
     * It supports labelling the edges as well as linking the edges to form both
     * MaximalEdgeRings and MinimalEdgeRings.
    """

    # States for linResultDirectedEdges
    SCANNING_FOR_INCOMING = 1
    LINKING_TO_OUTGOING = 2

    def __init__(self):
        EdgeEndStar.__init__(self)
        self._resultAreaEdgeList = None
        self.label = Label()

    def insert(self, de):
        self.insertEdgeEnd(de)

    def getOutgoingDegree(self, edgeRing=None) -> int:
        degree = 0
        if edgeRing is None:
            for de in self.edges:
                if de.isInResult:
                    degree += 1
        else:
            for de in self.edges:
                if de.edgeRing == edgeRing:
                    degree += 1
        return degree

    def getRightmostEdge(self):
        if len(self.edges) == 0:
            return None

        # DirectedEdge
        de0 = self.edges[0]

        if len(self.edges) == 1:
            return de0

        # DirectedEdge
        deLast = self.edges[-1]

        quad0 = de0.quadrant
        quad1 = deLast.quadrant

        if Quadrant.isNorthern(quad0) and Quadrant.isNorthern(quad1):
            return de0
        elif (not Quadrant.isNorthern(quad0)) and (not Quadrant.isNorthern(quad1)):
            return deLast
        else:
            # edges are in different hemispheres - make sure we return one that is non-horizontal
            if de0.dy != 0:
                return de0
            elif deLast.dy != 0:
                return deLast

        assert(0), "found two horizontal edges incident on node"

        return None

    def computeLabelling(self, geomGraphs) -> None:
        """
         * Compute the labelling for all dirEdges in this star, as well
         * as the overall labelling
        """

        # this call can throw a TopologyException
        EdgeEndStar.computeLabelling(self, geomGraphs)

        # determine the overall labelling for this DirectedEdgeStar
        # (i.e. for the node it is based at)
        self.label = Label(Location.UNDEF)
        for de in self.edges:
            label = de.edge.label
            for i in range(2):
                loc = label.getLocation(i)
                if loc == Location.INTERIOR or loc == Location.BOUNDARY:
                    self.label.setLocation(i, Location.INTERIOR)

    def mergeSymLabels(self) -> None:
        """
         * For each dirEdge in the star,
         * merge the label from the sym dirEdge into the label
        """
        for de in self.edges:
            label = de.label
            labelToMerge = de.sym.label
            label.merge(labelToMerge)
            logger.debug("DirectedEdgeStar.mergeSymLabels()  %s [%s] %s", type(de).__name__, id(de), label)

    def updateLabelling(self, nodeLabel) -> None:
        # Update incomplete dirEdge labels from the labelling for the node
        for de in self.edges:
            label = de.label
            label.setAllLocationsIfNull(0, location=nodeLabel.getLocation(0))
            label.setAllLocationsIfNull(1, location=nodeLabel.getLocation(1))

            logger.debug("DirectedEdgeStar.updateLabelling() %s [%s] %s", type(de).__name__, id(de), label)

    def linkResultDirectedEdges(self) -> None:
        """
         * Traverse the star of DirectedEdges, linking the included edges together.
         * To link two dirEdges, the <next> pointer for an incoming dirEdge
         * is set to the next outgoing edge.
         *
         * DirEdges are only linked if:
         *
         * - they belong to an area (i.e. they have sides)
         * - they are marked as being in the result
         *
         * Edges are linked in CCW order (the order they are stored).
         * This means that rings have their face on the Right
         * (in other words,
         * the topological location of the face is given by the RHS label of the DirectedEdge)
         *
         * PRECONDITION: No pair of dirEdges are both marked as being in the result
        """
        # make sure edges are copied to resultAreaEdges list
        self._getResultAreaEdges()

        # find first area edge (if any) to start linking at
        firstOut = None
        incoming = None
        state = DirectedEdgeStar.SCANNING_FOR_INCOMING

        # link edges in CCW order
        for nextOut in self._resultAreaEdgeList:

            assert(nextOut), "Found null edge in resultAreaList"

            # skip de's that we're not interested in
            if not nextOut.label.isArea():
                continue

            nextIn = nextOut.sym

            assert(nextIn), "Found null edge.sym"

            # record first outgoing edge, in order to link the last incoming edge
            if firstOut is None and nextOut.isInResult:
                firstOut = nextOut

            if state == DirectedEdgeStar.SCANNING_FOR_INCOMING:

                if not nextIn.isInResult:
                    continue

                incoming = nextIn
                state = DirectedEdgeStar.LINKING_TO_OUTGOING

            elif state == DirectedEdgeStar.LINKING_TO_OUTGOING:

                if not nextOut.isInResult:
                    continue

                incoming.next = nextOut
                state = DirectedEdgeStar.SCANNING_FOR_INCOMING

        if state == DirectedEdgeStar.LINKING_TO_OUTGOING:

            if firstOut is None:
                raise TopologyException("no outgoing dirEdge found", self.coord)

            assert(firstOut.isInResult), "unable to link last incoming dirEdge"
            assert(incoming), "no incoming edge found"

            incoming.next = firstOut

    def linkMinimalDirectedEdges(self, edgeRing) -> None:

        # find first area edge (if any) to start linking at
        firstOut = None
        incoming = None
        state = DirectedEdgeStar.SCANNING_FOR_INCOMING

        # link edges in CW order
        for nextOut in reversed(self._resultAreaEdgeList):

            nextIn = nextOut.sym

            # record first outgoing edge, in order to link the last incoming edge
            if firstOut is None and nextOut.edgeRing == edgeRing:
                firstOut = nextOut

            if state == DirectedEdgeStar.SCANNING_FOR_INCOMING:
                if nextIn.edgeRing != edgeRing:
                    continue
                incoming = nextIn
                state = DirectedEdgeStar.LINKING_TO_OUTGOING

            elif state == DirectedEdgeStar.LINKING_TO_OUTGOING:
                if nextOut.edgeRing != edgeRing:
                    continue
                incoming.nextMin = nextOut
                state = DirectedEdgeStar.SCANNING_FOR_INCOMING

        if state == DirectedEdgeStar.LINKING_TO_OUTGOING:

            assert(firstOut is not None), "found null for first outgoing dirEdge"
            assert(firstOut.edgeRing == edgeRing), "unable to link last incoming dirEdge"

            incoming.nextMin = firstOut

    def linkAllDirectedEdges(self) -> None:

        # find first area edge (if any) to start linking at
        prevOut = None
        firstIn = None

        # Link edges in CW order
        for nextOut in reversed(self.edges):
            nextIn = nextOut.sym

            if firstIn is None:
                firstIn = nextIn

            if prevOut is not None:
                nextIn.next = prevOut
            # record outgoing edge, in order to link the last incoming edge
            prevOut = nextOut

        firstIn.next = prevOut

    def findCoveredLineEdges(self) -> None:
        """
         * Traverse the star of edges, maintaing the current location in the result
         * area at this node (if any).
         *
         * If any L edges are found in the interiors of the result, mark them as covered.
        """
        """
         * Find first DirectedEdge of result area (if any).
         * The interiors of the result is on the RHS of the edge,
         * so the start location will be:
         * - INTERIOR if the edge is outgoing
         * - EXTERIOR if the edge is incoming
        """
        startLoc = Location.UNDEF
        for nextOut in self.edges:

            nextIn = nextOut.sym

            if not nextOut.isLineEdge:

                if nextOut.isInResult:
                    startLoc = Location.INTERIOR
                    break

                if nextIn.isInResult:
                    startLoc = Location.EXTERIOR
                    break

        # no A edges found, so can't determine if L edges are covered or not
        if startLoc == Location.UNDEF:
            return

        """
         * move around ring, keeping track of the current location
         * (Interior or Exterior) for the result area.
         * If L edges are found, mark them as covered if they are in the interiors
        """
        currLoc = startLoc
        for nextOut in self.edges:

            nextIn = nextOut.sym

            if nextOut.isLineEdge:
                nextOut.edge.isCovered = bool(currLoc == Location.INTERIOR)
            else:
                if nextOut.isInResult:
                    currLoc = Location.EXTERIOR

                if nextIn.isInResult:
                    currLoc = Location.INTERIOR

    def computeDepths(self, de) -> None:

        index = self.index(de)

        startDepth = de.getDepth(Position.LEFT)
        targetLastDepth = de.getDepth(Position.RIGHT)

        # compute the depths from this edge up to the end of the edge array
        nextDepth = self._computeDepths(index + 1, len(self.edges), startDepth)

        # compute the depths for the initial part of the array
        lastDepth = self._computeDepths(0, index, nextDepth)

        logger.debug("DirectedEdgeStar.computeDepths(%s) index:%s\n%s",
            len(self.edges),
            index,
            "\n".join(
            ["[{}] {} LEFT:{} - RIGHT:{} ".format(
                id(d),
                i,
                d.getDepth(Position.LEFT),
                d.getDepth(Position.RIGHT))
                for i, d in enumerate(self.edges)]))

        if lastDepth != targetLastDepth:
            raise TopologyException("depth mismatch", self.coord)

    def _getResultAreaEdges(self):

        if self._resultAreaEdgeList is None:

            self._resultAreaEdgeList = [de for de in self.edges
                    if de.isInResult or de.sym.isInResult]

        return self._resultAreaEdgeList

    def _computeDepths(self, start: int, end: int, startDepth: int) -> int:
        """
         * Compute the DirectedEdge depths for a subsequence of the edge array.
         *
         * @return the last depth assigned (from the R side of the last edge visited)
        """
        currDepth = startDepth
        for i in range(start, end):
            de = self.edges[i]
            de.setEdgeDepths(Position.RIGHT, currDepth)
            currDepth = de.getDepth(Position.LEFT)

        return currDepth

    def __str__(self) -> str:
        return "DirectedEdgeStar {}\n{}".format(
            self.coord,
            "\n".join([
                "out:{}\nin:{}".format(str(de), str(de.sym)) for de in self.edges
            ])
        )


def EdgeIntersectionLessThen(ei1, ei2) -> bool:
    return ei1.compareTo(ei2) < 0


class EdgeIntersection():
    """
     * Represents a point on an edge which intersects with another edge.
     *
     * The intersection may either be a single point, or a line segment
     * (in which case this point is the start of the line segment)
     * The intersection point must be precise.
    """
    def __init__(self, coord, segmentIndex: int, dist: float):

        # Coordinate the point of intersection
        self.coord = coord
        # the index of the containing line segment in the parent edge
        self.segmentIndex = segmentIndex
        # the edge distance of this point along the containing line segment
        self.dist = dist

    def isEndPoint(self, maxSegmentIndex: int) -> bool:
        if self.segmentIndex == 0 and self.dist == 0.0:
            return True
        if self.segmentIndex == maxSegmentIndex:
            return True
        return False

    def __str__(self) -> str:
        return "{} seg # = {} dist = {}".format(
            self.coord,
            self.segmentIndex,
            self.dist)

    def compareTo(self, other):
        if self.segmentIndex < other.segmentIndex:
            return -1
        elif self.segmentIndex > other.segmentIndex:
            return 1

        if self.dist < other.dist:
            return -1

        elif self.dist > other.dist:
            return 1

        return 0

    def __gt__(self, other):
        return self.compareTo(other) == 1

    def __lt__(self, other):
        return self.compareTo(other) == -1

    def __eq__(self, other):
        return self.compareTo(other) == 0

    def __ne__(self, other):
        return self.compareTo(other) != 0


class EdgeIntersectionList(dict):
    """
    """
    def __init__(self, newEdge):
        dict.__init__(self)
        self.edge = newEdge
        # EdgeIntersection
        self._ei = None
        self._sorted = False

    def add(self, coord, segmentIndex: int, dist: float):
        """
         * Adds an intersection into the list, if it isn't already there.
         * The input segmentIndex and dist are expected to be normalized.
         * @return the EdgeIntersection found or added
        """
        # key = hash((segmentIndex, dist))
        key = hash((segmentIndex, coord.x, coord.y))
        ei = self.get(key)
        if ei is None:
            ei = EdgeIntersection(coord, segmentIndex, dist)
            self[key] = ei
            self._sorted = False
        return ei

    @property
    def intersections(self) -> list:
        if not self._sorted:
            self._ei = list(self.values())
            quicksort(self._ei, EdgeIntersectionLessThen)
            self._sorted = True
        return self._ei

    @property
    def is_empty(self) -> bool:
        return len(self) < 1

    def isIntersection(self, coord) -> bool:
        for intersection in self.intersections:
            if intersection.coord == coord:
                return True
        return False

    def addEndpoints(self) -> None:
        coords = self.edge.coords
        maxSegIndex = len(coords) - 1
        self.add(coords[0], 0, 0.0)
        self.add(coords[maxSegIndex], maxSegIndex, 0.0)

    def addSplitEdges(self, edgeList: list) -> None:
        """
         * Creates new edges for all the edges that the intersections in this
         * list split the parent edge into.
         * Adds the edges to the input list (this is so a single list
         * can be used to accumulate all split edges for a Geometry).
         *
         * @param edgeList a list of EdgeIntersections
        """
        # ensure that the list has entries for the first and last point
        # of the edge
        self.addEndpoints()

        intersections = list(self.intersections)

        # there should always be at least two entries in the list
        eiPrev = intersections.pop(0)

        for ei in intersections:
            newEdge = self.createSplitEdge(eiPrev, ei)
            edgeList.append(newEdge)
            eiPrev = ei

    def createSplitEdge(self, ei0, ei1):

        npts = ei1.segmentIndex - ei0.segmentIndex + 2

        # Coordinate
        lastSegStartPt = self.edge.coords[ei1.segmentIndex]

        # if the last intersection point is not equal to the its segment
        # start pt, add it to the points list as well.
        # (This check is needed because the distance metric is not totally
        # reliable!). The check for point equality is 2D only - Z values
        # are ignored
        useIntPtl = ei1.dist > 0.0 or ei1.coord != lastSegStartPt

        end = ei1.segmentIndex + 1

        if not useIntPtl:
            end -= 1
            npts -= 1

        # CoordinateSequence
        _coords = self.edge.coords

        coords = []
        coords.append(ei0.coord)

        coords.extend(_coords[ei0.segmentIndex + 1:end])

        coords.append(ei1.coord)
        """
        logger.debug("[%s] EdgeIntersectionList.createSplitEdge() npts:%s\nei0:%s\nei1:%s\nedge.coords:%s\ncoords:%s",
            id(self),
            npts,
            ei0,
            ei1,
            _coords,
            ",".join([str(co) for co in coords]))
        """
        coords = CoordinateSequence.removeRepeatedPoints(coords)

        assert(len(coords) == npts), "Intersection removed multiple points at same location"

        label = Label(self.edge.label)
        return Edge(coords, label)

    def __str__(self) -> str:
        return "Intersections:\n{}".format(
            "\n".join([
                str(ei) for ei in self.intersections
            ]))


class EdgeNodingValidator():
    """
     * Validates that a collection of SegmentStrings is correctly noded.
     *
     * Throws an appropriate exception if an noding error is found.
    """
    def __init__(self, edges):
        # noding.SegmentString
        self.segStr = []
        self.nv = FastNodingValidator(self.toSegmentStrings(edges))

    def toSegmentStrings(self, edges):
        for edge in edges:
            coords = edge.coords.clone()
            self.segStr.append(BasicSegmentString(coords, edge))
        return self.segStr

    @staticmethod
    def checkValid(edges) -> None:
        """
         * Checks whether the supplied {@link Edge}s
         * are correctly noded.
         *
         * Throws a  {@link TopologyException} if they are not.
         *
         * @param edges a collection of Edges.
         * @throws TopologyException if the SegmentStrings are not
         *         correctly noded
         *
        """
        validator = EdgeNodingValidator(edges)
        validator._checkValid()

    def _checkValid(self) -> None:
        self.nv.checkValid()


class GeometryGraph(PlanarGraph):

    def __init__(self, geomIndex: int=-1, geom=None, boundaryNodeRule=None):

        PlanarGraph.__init__(self)

        # Geometry
        self.geom = geom
        """
         * The lineEdgeMap is a map of the linestring components of the
         * parentGeometry to the edges which are derived from them.
         * This is used to efficiently perform findEdge queries
         *
         * Following the above description there's no need to
         * compare LineStrings other then by pointer value.
        """
        # Edge
        self._lineEdgeMap = {}

        """
         * If this flag is true, the Boundary Determination Rule will
         * used when deciding whether nodes are in the boundary or not
        """
        self._useBoundaryDeterminationRule = True

        if boundaryNodeRule is None:
            boundaryNodeRule = BoundaryNodeRule.getBoundaryOGCSFS()

        self.boundaryNodeRule = boundaryNodeRule

        """
         * the index of this geometry as an argument to a spatial function
         * (used for labelling)
        """
        self.geomIndex = geomIndex

        # Cache for fast responses to getBoundaryPoints
        # CoordinateSequence
        self._boundaryPoints = None

        # Node
        self._boundaryNodes = None

        self.hasTooFewPoints = False

        # Coordinate
        self.invalidPoint = None

        if geom is not None:
            self._add(geom)

    def _createEdgeSetIntersector(self):
        # Allocates a new EdgeSetIntersector
        return SimpleMCSweepLineIntersector()

    def _add(self, geom) -> None:

        if geom.is_empty:
            return

        type_id = geom.type_id

        if type_id == GeomTypeId.GEOS_POLYGON:
            self._addPolygon(geom)

        elif type_id in (
                GeomTypeId.GEOS_LINEARRING,
                GeomTypeId.GEOS_LINESTRING
                ):
            # LineString also handles LinearRings
            self._addLineString(geom)

        elif type_id == GeomTypeId.GEOS_POINT:
            self._addPoint(geom)

        elif type_id in (
                GeomTypeId.GEOS_MULTILINESTRING,
                GeomTypeId.GEOS_MULTIPOINT,
                GeomTypeId.GEOS_MULTIPOLYGON,
                GeomTypeId.GEOS_GEOMETRYCOLLECTION
                ):
            self._addCollection(geom)

        else:
            raise ValueError("GeometryGraph.add(Geometry): unknown geometry type:{}".format(type(geom).__name__))

    def _addCollection(self, geoms) -> None:
        for geom in geoms.geoms:
            self._add(geom)

    def _addPoint(self, geom) -> None:
        self._insertPoint(self.geomIndex, geom.coord, Location.INTERIOR)

    def _addPolygonRing(self, geom, cwLeft: int, cwRight: int) -> None:

        # skip empty component
        if geom.is_empty:
            return

        coords = CoordinateSequence.removeRepeatedPoints(geom.coords)

        if len(coords) < 4:
            self.hasTooFewPoints = True
            self.invalidPoint = coords[0]
            return

        left = cwLeft
        right = cwRight

        """
         * the isCCW call might throw an
         * IllegalArgumentException if degenerate ring does
         * not contain 3 distinct points.
        """
        if CGAlgorithms.isCCW(coords):
            left, right = cwRight, cwLeft

        edge = Edge(coords, Label(self.geomIndex, Location.BOUNDARY, left, right))
        self._lineEdgeMap[id(geom)] = edge
        self.insertEdge(edge)
        self._insertPoint(self.geomIndex, coords[0], Location.BOUNDARY)

    def _addPolygon(self, geom) -> None:
        # LinearRing
        linearRing = geom.exterior
        self._addPolygonRing(linearRing, Location.EXTERIOR, Location.INTERIOR)
        if geom.interiors is not None:
            for linearRing in geom.interiors:
                self._addPolygonRing(linearRing, Location.INTERIOR, Location.EXTERIOR)

    def _addLineString(self, geom) -> None:

        coords = geom.coords
        if len(coords) < 2:
            self.hasTooFewPoints = True
            self.invalidPoint = coords[0]
            return

        edge = Edge(coords, Label(self.geomIndex, Location.INTERIOR))
        self._lineEdgeMap[id(geom)] = edge
        self.insertEdge(edge)
        """
         * Add the boundary points of the LineString, if any.
         * Even if the LineString is closed, add both points as if they
         * were endpoints.
         * This allows for the case that the node already exists and is
         * a boundary point.
        """
        self._insertBoundaryPoint(self.geomIndex, coords[0])
        self._insertBoundaryPoint(self.geomIndex, coords[-1])

    def _insertPoint(self, geomIndex: int, coord, onLocation: int) -> None:

        logger.debug("GeometryGraph.insertPoint(%s) called", coord)

        # Node
        node = self._nodes.addNode(coord)
        label = node.label
        if label.isNull():
            node.label = Label(geomIndex, onLocation)
        else:
            label.setLocation(geomIndex, onLocation)

    def _insertBoundaryPoint(self, geomIndex: int, coord) -> None:
        """
         * Adds candidate boundary points using the current
         * algorithm.BoundaryNodeRule.
         *
         * This is used to add the boundary
         * points of dim-1 geometries (Curves/MultiCurves).
        """
        # Node
        node = self._nodes.addNode(coord)

        # Label nodes always have labels
        label = node.label

        # the new point to insert is on a boundary
        boundaryCount = 1

        # determine the current location for the point (if any)
        loc = label.getLocation(geomIndex, Position.ON)
        if loc == Location.BOUNDARY:
            boundaryCount += 1

        # determine the boundary status of the point according to the
        # Boundary Determination Rule
        newLoc = GeometryGraph.determineBoundary(boundaryCount, self.boundaryNodeRule)
        label.setLocation(geomIndex, newLoc)

    def _addSelfIntersectionNodes(self, geomIndex: int) -> None:
        """
         * Add a node for a self-intersection.
         *
         * If the node is a potential boundary node (e.g. came from an edge
         * which is a boundary) then insert it as a potential boundary node.
         * Otherwise, just add it as a regular node.
        """
        for edge in self.edges:
            loc = edge.label.getLocation(geomIndex)
            # EdgeIntersectionList
            intersections = edge.intersections
            for intersection in intersections:
                self._addSelfIntersectionNode(geomIndex, intersection.coord, loc)

    def _addSelfIntersectionNode(self, geomIndex: int, coord, loc: int) -> None:
        #  if this node is already a boundary node, don't change it
        if self.isBoundaryNode(geomIndex, coord):
            return

        if loc == Location.BOUNDARY and self._useBoundaryDeterminationRule:
            self._insertBoundaryPoint(geomIndex, coord)
        else:
            self._insertPoint(geomIndex, coord, loc)

    @staticmethod
    def isInBoundary(boundaryCount: int) -> bool:
        """
         * This method implements the Boundary Determination Rule
         * for determining whether
         * a component (node or edge) that appears multiple times in elements
         * of a MultiGeometry is in the boundary or the interiors of the Geometry
         *
         * The SFS uses the "Mod-2 Rule", which this function implements
         *
         * An alternative (and possibly more intuitive) rule would be
         * the "At Most One Rule":
         *    isInBoundary = (componentCount == 1)
        """
        return boundaryCount % 2 == 1

    @staticmethod
    def determineBoundary(boundaryCount: int, boundaryNodeRule=None) -> int:

        if boundaryNodeRule is None:
            if GeometryGraph.isInBoundary(boundaryCount):
                return Location.BOUNDARY
            else:
                return Location.INTERIOR

        if boundaryNodeRule.isInBoundary(boundaryCount):
            return Location.BOUNDARY
        else:
            return Location.INTERIOR

    @property
    def boundaryNodes(self) -> list:

        if self._boundaryNodes is None:
            self._boundaryNodes = self._nodes.getBoundaryNodes(self.geomIndex)

        return self._boundaryNodes

    @property
    def boundaryPoints(self):

        if self._boundaryPoints is None:

            boundaryNodes = self.boundaryNodes
            coords = [node.coord for node in boundaryNodes]
            self._boundaryPoints = CoordinateSequence(coords)

        return self._boundaryPoints

    def findEdge(self, line):
        # Edge
        return self._lineEdgeMap.get(id(line))

    def computeSplitEdges(self, edgelist: list) -> None:

        oldListLen = len(edgelist)
        logger.debug("[%s] GeometryGraph.computeSplitEdges() scanning %s local and %s provided edges",
            id(self),
            len(self.edges),
            oldListLen)

        for edge in self.edges:
            # logger.debug("Adding split edges for %s", edge)
            edge.eiList.addSplitEdges(edgelist)

        logger.debug("[%s] GeometryGraph.computeSplitEdges() completed add:%s Edges",
            id(self),
            len(edgelist) - oldListLen)

    def addEdge(self, edge) -> None:
        """
         * Add an Edge computed externally.  The label on the Edge is assumed
         * to be correct.
        """
        self.insertEdge(edge)
        # CoordinateSequence
        coords = edge.coords
        # insert the endpoint as a node, to mark that it is on the boundary
        self._insertPoint(self.geomIndex, coords[0], Location.BOUNDARY)
        self._insertPoint(self.geomIndex, coords[-1], Location.BOUNDARY)

    def addPoint(self, coord) -> None:
        """
         * Add a point computed externally.  The point is assumed to be a
         * Point Geometry part, which has a location of INTERIOR.
         * @param coord Coordinate
        """
        self._insertPoint(self.geomIndex, coord, Location.INTERIOR)

    def computeSelfNodes(self, li, computeRingSelfNodes: bool, isDoneIfProperInt=False, env=None):
        """
         * Compute self-nodes, taking advantage of the Geometry type to
         * minimize the number of intersection tests.  (E.g. rings are
         * not tested for self-intersection, since
         * they are assumed to be valid).
         *
         * @param li the LineIntersector to use
         *
         * @param computeRingSelfNodes if <false>, intersection checks are
         *  optimized to not test rings for self-intersection
         *
         * @return the SegmentIntersector used, containing information about
         *  the intersections found
        """
        logger.debug("[%s] GeometryGraph.computeSelfNodes(%s) nodes:%s", id(self), self.geomIndex, len(self.nodes))

        if env is None and type(isDoneIfProperInt).__name__ == 'Envelope':
            env, isDoneIfProperInt = isDoneIfProperInt, False
        # li, includeProper, recordIsolated
        si = SegmentIntersector(li, True, False)
        si.isDoneWhenProperInt = isDoneIfProperInt

        # SimpleMCSweepLineIntersector
        esi = self._createEdgeSetIntersector()

        # Edge
        edges = self.edges

        if (env is not None) and not env.covers(self.geom.envelope):
            edges = [edge for edge in edges if edge.envelope.intersects(env)]

        isRings = self.geom.type_id in (
            GeomTypeId.GEOS_LINEARRING,
            GeomTypeId.GEOS_POLYGON,
            GeomTypeId.GEOS_MULTIPOLYGON)

        computeAllSegments = computeRingSelfNodes or not isRings

        esi.computeSelfIntersections(edges, si, computeAllSegments)

        self._addSelfIntersectionNodes(self.geomIndex)

        logger.debug("[%s] GeometryGraph.computeSelfNodes() completed # tests = %s nodes:%s",
            id(self),
            si.numTests,
            len(self.nodes))

        return si

    def computeEdgeIntersections(self, graph, li, includeProper: bool, env=None):

        si = SegmentIntersector(li, includeProper, True)
        si.setBoundaryNodes(self.boundaryNodes, graph.boundaryNodes)

        # SimpleMCSweepLineIntersector
        esi = self._createEdgeSetIntersector()

        se = self.edges
        oe = graph.edges

        if env is not None:

            if not env.covers(self.geom.envelope):
                se = [edge for edge in se if edge.envelope.intersects(env)]

            if not env.covers(graph.geom.envelope):
                oe = [edge for edge in oe if edge.envelope.intersects(env)]

        esi.computeIntersections(se, oe, si)

        return si
