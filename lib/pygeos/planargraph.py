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
from .shared import Quadrant
from .algorithms import CGAlgorithms


class GraphComponent():
    """
     * A GraphComponent is the parent class for the objects'
     * that form a graph.
     *
     * Each GraphComponent can carry a Label.
    """
    def __init__(self):
        self.marked = False
        self.visited = False
        self.label = -1

    @staticmethod
    def setMarkedMap(components, marked):
        for c in components:
            c.marked = marked

    @staticmethod
    def setVisited(components, visited):
        for c in components:
            c.visited = visited


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
    def __init__(self):
        # Edge
        self.edges = []

        # DirectedEdge
        self.dirEdges = []
        self._nodeMap = NodeMap()
    
    @property
    def nodes(self):
        return self._nodeMap.values()
    
    def addNode(self, node):
        """
         * Adds a node to the nodeMap, replacing any that is already at that
         * location.
         *
         * Only subclasses can add Nodes, to ensure Nodes are
         * of the right type.
         * @return the added node
        """
        return self._nodeMap.add(node)

    def addEdge(self, edge):
        """
         * Adds the Edge and its DirectedEdges with this PlanarGraph.
         *
         * Assumes that the Edge has already been created with its associated
         * DirectEdges.
         * Only subclasses can add Edges, to ensure the edges added are of
         * the right class.
        """
        self.edges.append(edge)
        self.addDirEdge(edge.dirEdge[0])
        self.addDirEdge(edge.dirEdge[1])

    def addDirEdge(self, de):
        """
         * Adds the Edge to this PlanarGraph.
         *
         * Only subclasses can add DirectedEdges,
         * to ensure the edges added are of the right class.
        """
        self.dirEdges.append(de)

    def _removeDirEdge(self, de):
        """
         * Removes DirectedEdge from its from-Node and from this PlanarGraph.
        """
        to_remove = [i for i, dirEdge in enumerate(self.dirEdges) if dirEdge is de]
        for i in reversed(to_remove):
            self.dirEdges.pop(i)

    def removeDirEdge(self, de):
        """
         * Removes DirectedEdge from its from-Node and from this PlanarGraph. Note:
         * This method does not remove the Nodes associated with the DirectedEdge,
         * even if the removal of the DirectedEdge reduces the degree of a Node to
         * zero.
        """
        sym = de.sym
        if sym is not None:
            sym.sym = None
        de._from.deStar.remove(de)
        self._removeDirEdge(de)

    def removeEdge(self, edge):
        to_remove = [i for i, ed in enumerate(self.edges) if ed is edge]
        for i in reversed(to_remove):
            self.edges.pop(i)

    def removeNode(self, node):
        """
         * Removes a node from the graph, along with any associated
         * DirectedEdges and Edges.
        """
        outEdges = node.deStar.edges
        for de in outEdges:
            sym = de.sym
            if sym is not None:
                self.removeDirEdge(sym)
            self._removeDirEdge(de)
            edge = de.edge
            if edge is not None:
                self.removeEdge(edge)
        self._nodeMap.remove(node.coord)

    def findNodesOfDegree(self, degree, nodesFound=None):
        """
         * Get all Nodes with the given number of Edges around it.
        """
        # NodeMap.container &
        nm = []
        self.getNodes(nm)

        res = [node for node in nm if node.degree == degree]

        if nodesFound is None:
            return res

        nodesFound.extend(res)

    def getNodes(self, nodes):
        """
         * Returns the Nodes in this PlanarGraph.
         *
         * @param nodes : the nodes are push_back'ed here
        """
        self._nodeMap.getNodes(nodes)

    def findNode(self, pt):
        return self._nodeMap.find(pt)


class SubGraph():
    def __init__(self, parent):
        self.parentGraph = parent
        # Edge
        self.edges = []
        # DirectedEdge
        self.dirEdges = []
        self._nodeMap = NodeMap()


class Edge(GraphComponent):

    """
     * Represents an undirected edge of a PlanarGraph.
     *
     * An undirected edge in fact simply acts as a central point of reference
     * for two opposite DirectedEdge.
     *
     * Usually a client using a PlanarGraph will subclass Edge
     * to add its own application-specific data and methods.
    """

    def __init__(self, de0=None, de1=None):

        # The two DirectedEdges associated with this Edge
        self.dirEdge = []

        if de0 is not None and de1 is not None:
            self.setDirectedEdges(de0, de1)

    def setDirectedEdges(self, de0, de1):
        self.dirEdge.append(de0)
        self.dirEdge.append(de1)
        de0.edge = self
        de1.edge = self
        de0.sym = de1
        de1.sym = de0
        de0._from.addOutEdge(de0)
        de1._from.addOutEdge(de1)

    def getDireEdge(self, fromNode):

        if type(fromNode).__name__ == 'int':
            return self.dirEdge[fromNode]

        if self.dirEdge[0]._from is fromNode:
            return self.dirEdge[0]

        if self.dirEdge[1]._from is fromNode:
            return self.dirEdge[1]

        return None

    def getOppositeNode(self, node):

        if self.dirEdge[0]._from is node:
            return self.dirEdge[0]._to

        if self.dirEdge[1]._from is node:
            return self.dirEdge[1]._to

        return None


class DirectedEdge(GraphComponent):
    """
     * Represents a directed edge in a PlanarGraph.
     *
     * A DirectedEdge may or may not have a reference to a parent Edge
     * (some applications of planar graphs may not require explicit Edge
     * objects to be created). Usually a client using a PlanarGraph
     * will subclass DirectedEdge to add its own application-specific
     * data and methods.
    """
    def __init__(self, newFrom, newTo, directionPt, newEdgeDirection):

        GraphComponent.__init__(self)

        # Edge, parentEdge
        self.edge = None

        # Node
        self._from = newFrom
        self._to = newTo

        # Coordinate
        self.coord = newFrom.coord
        self.direction = directionPt

        # DirectedEdge
        self.sym = None

        # bool
        self.edgeDirection = newEdgeDirection

        dx = directionPt.x - self.coord.x
        dy = directionPt.y - self.coord.y

        # int
        self.quadrant = Quadrant.quadrant(dx, dy)

        # float
        self.angle = atan2(dy, dx)

    def getEdge(self):
        return self.edge

    def setEdge(self, newParentEdge):
        self.edge = newParentEdge

    def compareDirection(self, de):
        # if the rays are in different quadrants, determining the ordering is trivial
        if self.quadrant > de.quadrant:
            return 1
        if self.quadrant < de.quadrant:
            return -1
        # vectors are in the same quadrant - check relative orientation of direction vectors
        # this is > e if it is CCW of e
        return CGAlgorithms.computeOrientation(de.coord, de.direction, self.direction)

    def compareTo(self, de):
        return self.compareDirection(de)

    @staticmethod
    def toEdges(dirEdges, edges=None):

        if edges is None:
            edges = []

        for edge in dirEdges:
            edges.append(edge.edge)

        return edges


class Node(GraphComponent):
    """
     * A node in a PlanarGraph is a location where 0 or more Edge meet.
     *
     * A node is connected to each of its incident Edges via an outgoing
     * DirectedEdge. Some clients using a PlanarGraph may want to
     * subclass Node to add their own application-specific
     * data and methods.
    """
    def __init__(self, coord, deStar=None):
        # Coordinate The location of this Node
        self.coord = coord
        # The collection of DirectedEdges that leave this Node
        # DirectedEdgeStar
        if deStar is None:
            self.deStar = DirectedEdgeStar()
        else:
            self.deStar = deStar

    def addOutEdge(self, de):
        """
            Adds an outgoing DirectedEdge to this Node.
        """
        self.deStar.add(de)

    def getOutEdges(self):
        """
            Returns the collection of DirectedEdges that
            leave this Node.
        """
        return self.deStar

    @property
    def degree(self):
        """
            Returns the number of edges around this Node.
        """
        return self.deStar.degree

    def getIndex(self, edge):
        """
            Returns the zero-based index of the given Edge,
            after sorting in ascending order by angle with
            the positive x-axis.
        """
        return self.deStar.getIndex(edge)

    def toEdges(dirEdges, edges=None):

        if edges is None:
            edges = []

        for de in dirEdges:
            edges.append(de.edge)

        return edges

    def getEdgesBetween(self, node0, node1):
        # Edge
        edges0 = set(self.toEdges(node0.deStar.edges))

        # Edge
        edges1 = set(self.toEdges(node1.deStar.edges))

        return list(edges0.intersection(edges1))


class DirectedEdgeStar():
    """
     * A sorted collection of DirectedEdge which leave a Node in a PlanarGraph.
    """
    def __init__(self):
        # DirectedEdge
        self._outEdges = []
        self._sorted = False

    def sortEdges(self):

        if not self._sorted:
            self._outEdges = sorted(self._outEdges, key=lambda de: de.angle)
            self._sorted = True

    def add(self, de):
        """
            Adds a new member to this DirectedEdgeStar.
        """
        self._outEdges.append(de)
        self._sorted = False

    def remove(self, de):
        """
            Drops a member of this DirectedEdgeStar.
        """
        for i, outEdge in enumerate(self._outEdges):
            if outEdge is de:
                self._outEdges.pop(i)
                break

    @property
    def degree(self):
        """
            Returns the number of edges around the Node associated
            with this DirectedEdgeStar.
        """
        return len(self._outEdges)

    @property
    def coord(self):
        """
         * Return the coordinate of the root node of this Edge
        """
        if len(self._outEdges) == 0:
            return None
        return self._outEdges[0].coord

    @property
    def edges(self):
        """
            Returns the DirectedEdges, in ascending order
            by angle with the positive x-axis.
        """
        self.sortEdges()
        return self._outEdges

    def getIndex(self, i):

        if type(i).__name__ == 'int':
            """
                Returns the remainder when i is divided by the number of
                edges in this DirectedEdgeStar.
            """
            maxi = len(self._outEdges)
            modi = i % maxi
            if modi < 0:
                modi += maxi
            return modi

        elif type(i).__name__ == 'DirectedEdge':
            """
                Returns the zero-based index of the given DirectedEdge,
                after sorting in ascending order
                by angle with the positive x-axis.
            """
            self.sortEdges()
            return self._outEdges.index(i)

    def getNextEdge(self, dirEdge):
        """
            Returns the DirectedEdge on the left-hand side
            of the given DirectedEdge (which must be a member of this
            DirectedEdgeStar).
        """
        i = self.getIndex(dirEdge)
        return self._outEdges[self.getIndex(i + 1)]


class NodeMap(dict):
    """
     * A map of Node, indexed by the coordinate of the node.
    """
    def add(self, node):
        """
         * Adds a node to the map, replacing any that is already at that location.
        """
        coord = node.coord
        self[coord] = node
        return node

    def remove(self, coord):
        """
         * Removes the Node at the given location, and returns it
         * (or null if no Node was there).
        """
        node = self.find(coord)
        if node is not None:
            del self[coord]
        return node
    
    def getNodes(self, nodes):
        nodes.extend(self.values())

    def find(self, coord):
        """
         * Returns the Node at the given location, or null if no Node was there.
        """
        return self.get(coord)
