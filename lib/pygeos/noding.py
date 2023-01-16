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


from .index_strtree import (
    STRtree
    )
from .algorithms import (
    LineIntersector,
    MonotoneChainBuilder,
    MonotoneChainOverlapAction
    )
from .shared import (
    logger,
    quicksort,
    TopologyException,
    CoordinateFilter,
    CoordinateSequence,
    LinearComponentExtracter
    )


class Octant():
    """
     * Methods for computing and working with octants of the Cartesian plane.
     *
     * Octants are numbered as follows:
     *
     *   2|1
     * 3  |  0
     * ---+--
     * 4  |  7
     *   5|6
     *
     * If line segments lie along a coordinate axis, the octant is the lower of the two
     * possible values.
    """

    @staticmethod
    def octant(p0, p1) -> int:

        if type(p0).__name__ == 'float':
            dx, dy = p0, p1

        else:
            dx = p1.x - p0.x
            dy = p1.y - p0.y

        if dx == 0.0 and dy == 0.0:
            raise ValueError("Cannot compute the octant for point")

        adx = abs(dx)
        ady = abs(dy)

        if dx >= 0:
            if dy >= 0:
                if adx >= ady:
                    return 0
                else:
                    return 1
            else:
                if adx >= ady:
                    return 7
                else:
                    return 6
        else:
            if dy >= 0:
                if adx >= ady:
                    return 3
                else:
                    return 2
            else:
                if adx >= ady:
                    return 4
                else:
                    return 5


class OrientedCoordinateArray():
    """
     * Allows comparing CoordinateSequences
     * in an orientation-independent way.
    """
    def __init__(self, coords):

        self.coords = coords
        self._orientation = OrientedCoordinateArray.orientation(coords)

    @staticmethod
    def orientation(coords) -> bool:
        """
         * Computes the canonical orientation for a coordinate array.
         *
         * @param coords the array to test
         * @return true if the points are oriented forwards
         * @return false</code if the points are oriented in reverse
        """
        return CoordinateSequence.increasingDirection(coords) == 1

    def compareTo(self, oca) -> int:
        """
         * Compares two OrientedCoordinateArray for their
         * relative order
         *
         * @return -1 this one is smaller
         * @return 0 the two objects are equal
         * @return 1 this one is greater
        """
        return OrientedCoordinateArray.compareOriented(
                self.coords, self._orientation,
                oca.coords, oca._orientation)

    @staticmethod
    def compareOriented(coords1, orientation1: bool, coords2, orientation2: bool) -> int:

        if orientation1:
            i1 = 0
            dir1 = 1
            limit1 = len(coords1)
        else:
            i1 = len(coords1) - 1
            dir1 = -1
            limit1 = -1

        if orientation2:
            i2 = 0
            dir2 = 1
            limit2 = len(coords2)
        else:
            i2 = len(coords2) - 1
            dir2 = -1
            limit2 = -1

        while (True):
            compPt = coords1[i1].compareTo(coords2[i2])
            if compPt != 0:
                return compPt
            i1 += dir1
            i2 += dir2
            done1 = i1 == limit1
            done2 = i2 == limit2
            if done1 and not done2:
                return -1
            elif not done1 and done2:
                return 1
            elif done1 and done2:
                return 0

    def __lt__(self, other) -> bool:
        self.compareTo(other) == -1

    def __gt__(self, other) -> bool:
        self.compareTo(other) == 1

    def __eq__(self, other) -> bool:
        self.compareTo(other) == 0

    def __hash__(self):
        # equivalent coords sequences share same hash
        if self._orientation:
            _co = tuple(self.coords)
        else:
            _co = tuple(reversed(self.coords))
        return hash((_co))


class SegmentPointComparator():
    """
     * Implements a robust method of comparing the relative position of two
     * points along the same segment.
     * The coordinates are assumed to lie "near" the segment.
     * This means that this algorithm will only return correct results
     * if the input coordinates
     * have the same precision and correspond to rounded values
     * of exact coordinates lying on the segment.
    """

    @staticmethod
    def compare(octant: int, p0, p1) -> int:
        if p0 == p1:
            return 0

        xSign = SegmentPointComparator.relativeSign(p0.x, p1.x)
        ySign = SegmentPointComparator.relativeSign(p0.y, p1.y)

        return [
            SegmentPointComparator.compareValue(xSign, ySign),
            SegmentPointComparator.compareValue(ySign, xSign),
            SegmentPointComparator.compareValue(ySign, -xSign),
            SegmentPointComparator.compareValue(-xSign, ySign),
            SegmentPointComparator.compareValue(-xSign, -ySign),
            SegmentPointComparator.compareValue(-ySign, -xSign),
            SegmentPointComparator.compareValue(-ySign, xSign),
            SegmentPointComparator.compareValue(xSign, -ySign)
            ][octant]

    @staticmethod
    def relativeSign(x0: float, x1: float) -> int:
        if x0 < x1:
            return -1
        if x0 > x1:
            return 1
        return 0

    @staticmethod
    def compareValue(s0: int, s1: int) -> int:
        if s0 < 0:
            return -1
        if s0 > 0:
            return 1
        if s1 < 0:
            return -1
        if s1 > 0:
            return 1
        return 0


def segmentNodeLT(s1, s2):
    """
     * sort segment nodes
    """
    return s1.compareTo(s2) < 0


class SegmentNode():
    """
     * Represents an intersection point between two NodedSegmentString
    """
    def __init__(self, edge, coord, segmentIndex: int, segmentOctant: int):

        # NodedSegmentString
        self.segString = edge
        # Coordinate
        self.coord = coord
        self.segmentIndex = segmentIndex
        self.segmentOctant = segmentOctant
        self.isInterior = coord != edge.coords[segmentIndex]

    def isEndPoint(self, maxSegIndex: int) -> bool:

        if self.segmentIndex == 0 and not self.isInterior:
            return True

        return self.segmentIndex == maxSegIndex

    def compareTo(self, other) -> int:
        """
         * @return -1 this EdgeIntersection is located before
         *            the argument location
         * @return 0 this EdgeIntersection is at the argument location
         * @return 1 this EdgeIntersection is located after the
         *           argument location
        """
        if self.segmentIndex < other.segmentIndex:
            return -1
        if self.segmentIndex > other.segmentIndex:
            return 1
        if self.coord == other.coord:
            return 0

        return SegmentPointComparator.compare(self.segmentOctant, self.coord, other.coord)

    def __str__(self):
        return "{} seg#={} octant#={}".format(self.coord, self.segmentIndex, self.segmentOctant)


class SegmentNodeMap(dict):
    """
     * A map of Node, indexed by the segmentIndex.
    """
    def __init__(self):
        dict.__init__(self)
        self._sorted = False
        # SegmentNode
        self._nodes = []

    @property
    def nodes(self):
        if not self._sorded:
            self._sorted = True
            self._nodes = list(self.values())
            quicksort(self._nodes, segmentNodeLT)
        # SegmentNode
        return self._nodes

    def insert(self, newNode):
        """
         * Adds a Node to the map, replacing any that is already at that location.
         * NOTE:
         * The above statement conflicts with SegmentNodeList.add()
         * return -> SegmentIntersection found or added
         * @param newNode
        """
        key = (newNode.segmentIndex, newNode.coord.x, newNode.coord.y)
        node = self.get(key)

        if node is None:
            node = newNode

        self._sorded = False

        self[key] = node

        return node

    def __str__(self):
        nodes = self.values()
        return "\n".join([str(node) for node in nodes])


class SegmentNodeList():
    """
     * A list of the SegmentNode present along a
     * NodedSegmentString.
    """
    def __init__(self, edge):
        # NodedSegmentString parent edge
        self.edge = edge
        # SegmentNode segmentNodeLT
        self.nodeMap = SegmentNodeMap()

    @property
    def nodes(self):
        return self.nodeMap.nodes

    def checkSplitEdgesCorrectness(self, splitEdges: list) -> None:
        """
         * Checks the correctness of the set of split edges corresponding
         * to this edge
         *
         * @param splitEdges the split edges for this edge (in order)
        """
        if len(splitEdges) == 0:
            return

        # CoordinateSequence
        coords = self.edge.coords

        split0 = splitEdges[0]
        p0 = split0.coords[0]
        if not p0 == coords[0]:
            raise Exception("bad split edge start point")

        splitn = splitEdges[-1]
        p1 = splitn.coords[-1]
        if not p1 == coords[-1]:
            raise Exception("bad split edge end point")

    def createSplitEdge(self, ei0, ei1):
        """
         * Create a new "split edge" with the section of points between
         * (and including) the two intersections.
         * The label for the new edge is the same as the label for the
         * parent edge.
         *
         * ownership of return value is transferred
        """
        coords = self.edge.coords
        npts = ei1.segmentIndex - ei0.segmentIndex + 2
        # Coordinate
        lastSegStartPt = coords[ei1.segmentIndex]

        # if the last intersection point is not equal to the its
        # segment start pt, add it to the points list as well.
        # (This check is needed because the distance metric is not
        # totally reliable!)

        # The check for point equality is 2D only - Z values are ignored

        # Added check for npts being == 2 as in that case NOT using second point
        # would mean creating a SegmentString with a single point
        useIntPt1 = npts == 2 or (ei1.isInterior or ei1.coord != lastSegStartPt)

        if not useIntPt1:
            npts -= 1

        pts = CoordinateSequence()
        # ipt = 1
        pts.append(ei0.coord.clone())
        pts.extend(coords[ei0.segmentIndex + 1:ei1.segmentIndex + 1])

        if useIntPt1:
            pts.append(ei1.coord.clone())

        logger.debug("SegmentNodeList.createSplitEdge(%s) s0:%s s1:%s %s-%s",
            len(pts),
            ei0.segmentIndex,
            ei1.segmentIndex, 
            pts[0],
            pts[-1]
            )
        return NodedSegmentString(pts, self.edge.context)

    def addCollapsedNodes(self) -> None:
        """
         * Adds nodes for any collapsed edge pairs.
         * Collapsed edge pairs can be caused by inserted nodes, or they
         * can be pre-existing in the edge vertex list.
         * In order to provide the correct fully noded semantics,
         * the vertex at the base of a collapsed pair must also be added
         * as a node.
        """
        collapsedVertexIndexes = []
        self.findCollapsesFromInsertedNodes(collapsedVertexIndexes)
        self.findCollapsesFromExistingVertices(collapsedVertexIndexes)

        # Node the collapses
        coords = self.edge.coords
        for vertexIndex in collapsedVertexIndexes:
            self.add(coords[vertexIndex], vertexIndex)

    def findCollapsesFromExistingVertices(self, collapsedVertexIndexes: list) -> None:
        """
         * Adds nodes for any collapsed edge pairs
         * which are pre-existing in the vertex list.
        """
        coords = self.edge.coords
        # or we'll never exit the loop below
        if len(coords) < 2:
            return
        for i in range(len(coords) - 2):
            p0 = coords[i]
            p2 = coords[i + 2]
            if p0 == p2:
                collapsedVertexIndexes.append(i + 1)

    def findCollapsesFromInsertedNodes(self, collapsedVertexIndexes: list) -> None:
        """
         * Adds nodes for any collapsed edge pairs caused by inserted nodes
         * Collapsed edge pairs occur when the same coordinate is inserted
         * as a node both before and after an existing edge vertex.
         * To provide the correct fully noded semantics,
         * the vertex must be added as a node as well.
        """
        nodes = self.nodes
        # SegmentNode
        eiPrev = nodes[0]
        # there should always be at least two entries in the list,
        # since the endpoints are nodes
        for i in range(1, len(nodes)):
            ei = nodes[i]
            collapsedVertexIndex = self.findCollapseIndex(eiPrev, ei)
            if collapsedVertexIndex > -1:
                collapsedVertexIndexes.append(collapsedVertexIndex)
            eiPrev = ei

    def findCollapseIndex(self, ei0, ei1) -> int:
        if ei0.coord != ei1.coord:
            return -1

        numVerticesBetween = ei1.segmentIndex - ei0.segmentIndex

        if not ei1.isInterior:
            numVerticesBetween -= 1

        # if there is a single vertex between the two equal nodes,
        # this is a collapse
        if numVerticesBetween == 1:
            return ei0.segmentIndex + 1

        return -1

    def add(self, coord, segmentIndex: int):
        """
         * Adds an intersection into the list, if it isn't already there.
         * The input segmentIndex is expected to be normalized.
         *
         * @return the SegmentIntersection found or added. It will be
         *     destroyed at SegmentNodeList destruction time.
         *
         * @param coord the intersection Coordinate, will be copied
         * @param segmentIndex
        """
        eiNew = SegmentNode(self.edge, coord, segmentIndex, self.edge.getSegmentOctant(segmentIndex))
        return self.nodeMap.insert(eiNew)

    def addEndpoints(self) -> None:
        """
         * Adds entries for the first and last points of the edge to the list
        """
        coords = self.edge.coords
        maxSegIndex = len(coords) - 1
        self.add(coords[0], 0)
        self.add(coords[-1], maxSegIndex)

    def addSplitEdges(self, edgeList: list) -> None:
        """
         * Creates new edges for all the edges that the intersections in this
         * list split the parent edge into.
         * Adds the edges to the input list (this is so a single list
         * can be used to accumulate all split edges for a Geometry).
        """

        # @NOTE: debug only, should remove
        testingSplitEdges = []

        # ensure that the list has entries for the first and last
        # since the endpoints are nodes
        self.addEndpoints()
        self.addCollapsedNodes()

        # there should always be at least two entries in the list
        # since the endpoints are nodes
        logger.debug("SegmentNodeList.addSplitEdges edgelist:%s", len(edgeList))

        nodes = self.nodes
        eiPrev = nodes[0]
        for i in range(1, len(nodes)):
            ei = nodes[i]

            if not ei.compareTo(eiPrev):
                continue

            # SegmentString
            newEdge = self.createSplitEdge(eiPrev, ei)
            edgeList.append(newEdge)
            # @NOTE: debug only, should remove
            testingSplitEdges.append(newEdge)
            eiPrev = ei

        # @NOTE: debug only, should remove
        logger.debug("SegmentNodeList.addSplitEdges(%s) for %s Nodes check for correctness",
            len(edgeList),
            len(nodes))

        self.checkSplitEdgesCorrectness(testingSplitEdges)

    def __str__(self):
        return "Intersections: ({}):\n{}".format(
            len(self.nodes),
            "\n".join([str(n) for n in self.nodes])
            )


class SegmentIntersector():
    """
     * Processes possible intersections detected by a Noder.
     *
     * The SegmentIntersector is passed to a Noder.
     * The addIntersections method is called whenever the Noder
     * detects that two SegmentStrings <i>might</i> intersect.
     * This class may be used either to find all intersections, or
     * to detect the presence of an intersection.  In the latter case,
     * Noders may choose to short-circuit their computation by calling the
     * isDone method.
     * This class is an example of the <i>Strategy</i> pattern.
    """
    def processIntersections(self, e0, segIndex0: int, e1, segIndex1: int) -> None:
        """
         * This method is called by clients
         * of the SegmentIntersector interface to process
         * intersections for two segments of the SegmentStrings
         * being intersected.
        """
        raise NotImplementedError()


class SegmentIntersectionDetector(SegmentIntersector):
    """
     * Detects and records an intersection between two {@link SegmentString}s,
     * if one exists.
     *
     * This strategy can be configured to search for proper intersections.
     * In this case, the presence of any intersection will still be recorded,
     * but searching will continue until either a proper intersection has been found
     * or no intersections are detected.
     *
     * Only a single intersection is recorded.
     *
     * @version 1.7
    """
    def __init__(self, li):
        SegmentIntersector.__init__(self)

        # LineIntersector
        self.li = li

        # bool
        self.findProper = False
        self.findAllTypes = False

        self.hasIntersection = False
        self.hasProperIntersection = False
        self.hasNonProperIntersection = False
        # geom.Coordinate
        self.intPt = None
        # geom.CoordinateSequence
        self.intSegments = None

    def isDone(self) -> bool:

        # If finding all types, we can stop
        # when both possible types have been found.
        if self.findAllTypes:
            return self.hasProperIntersection and self.hasNonProperIntersection

        # If searching for a proper intersection, only stop if one is found
        if self.findProper:
            return self.hasProperIntersection

        return self.hasIntersection

    def processIntersections(self, e0, segIndex0: int, e1, segIndex1: int)-> None:
        """
         * This method is called by clients
         * of the {@link SegmentIntersector} class to process
         * intersections for two segments of the {@link SegmentStrings} being intersected.
         * Note that some clients (such as {@link MonotoneChain}s) may optimize away
         * this call for segment pairs which they have determined do not intersect
         * (e.g. by an disjoint envelope test).
        """
        # don't bother intersecting a segment with itself
        if e0 == e1 and segIndex0 == segIndex1:
            return

        p0 = e0.coords[segIndex0]
        p1 = e0.coords[segIndex0 + 1]
        q0 = e1.coords[segIndex1]
        q1 = e1.coords[segIndex1 + 1]

        self.li.computeLinesIntersection(p0, p1, q0, q1)

        if self.li.hasIntersection:

            # record intersection info
            self.hasIntersection = True

            isProper = self.li.isProper

            if isProper:
                self.hasProperIntersection = True
            else:
                self.hasNonProperIntersection = True

            # If this is the kind of intersection we are searching for
            # OR no location has yet been recorded
            # save the location data
            saveLocation = True

            if self.findProper and not isProper:
                saveLocation = False

            if self.intPt is None or saveLocation:

                # record intersection location (approximate)
                self.intPt = self.li.intersectionPts[0]

                # record intersecting segments
                self.intSegments = CoordinateSequence([p0, p1, q0, q1])


class SegmentSetMutualIntersector():
    """
     * An intersector for the red-blue intersection problem.
     *
     * In this class of line arrangement problem,
     * two disjoint sets of linestrings are provided.
     * It is assumed that within
     * each set, no two linestrings intersect except possibly at their endpoints.
     * Implementations can take advantage of this fact to optimize processing.
     *
     * @author Martin Davis
     * @version 1.10
    """
    def __init__(self):
        # SegmentIntersector
        self.si = None

    def setBaseSegments(self, segStrings) -> None:
        """
         * @param segStrings0 a collection of {@link SegmentString}s to node
        """
        raise NotImplementedError()

    def process(self, segStrings) -> None:
        """
         * Computes the intersections for two collections of {@link SegmentString}s.
         *
         * @param segStrings1 a collection of {@link SegmentString}s to node
        """
        raise NotImplementedError()


class MCIndexSegmentSetMutualIntersector(SegmentSetMutualIntersector):
    """
     * Intersects two sets of {@link SegmentStrings} using a index based
     * on {@link MonotoneChain}s and a {@link SpatialIndex}.
     *
     * @version 1.7
    """
    def __init__(self):
        SegmentSetMutualIntersector.__init__(self)
        """
         * The {@link SpatialIndex} used should be something that supports
         * envelope (range) queries efficiently (such as a {@link Quadtree}
         * or {@link STRtree}.
        """
        # STRtree
        self.index = STRtree()
        # MonotoneChain
        self.monoChains = []
        self.indexCounter = 0
        self.processCounter = 0
        # statistics
        self.nOverlaps = 0

    def setBaseSegments(self, segStrings) -> None:
        for css in segStrings:
            self.addToIndex(css)

    def process(self, segStrings) -> None:
        # NOTE: re-populates the MonotoneChain vector with newly created chains
        self.processCounter = self.indexCounter + 1
        self.nOverlaps = 0
        self.monoChains.clear()

        for seg in segStrings:
            self.addToMonoChains(seg)
        self.intersectChains()

    def addToIndex(self, segStr) -> None:
        # MonoChains
        segChains = []
        MonotoneChainBuilder.getChains(segStr.coords, segStr, segChains)
        for mc in segChains:
            mc.id = self.indexCounter
            self.indexCounter += 1
            self.index.insert(mc.envelope, mc)

    def intersectChains(self) -> None:
        overlapAction = SegmentOverlapAction(self.si)

        for queryChain in self.monoChains:
            overlapChains = []
            self.index.query(queryChain.envelope, overlapChains)

            for testChain in overlapChains:
                queryChain.computeOverlaps(testChain, overlapAction)
                self.nOverlaps += 1

                if self.si.isDone:
                    return

    def addToMonoChains(self, segStr) -> None:
        segChains = []
        MonotoneChainBuilder.getChains(segStr.coords, segStr, segChains)
        for mc in segChains:
            mc.id = self.processCounter
            self.processCounter += 1
            self.monoChains.append(mc)


class FastSegmentSetIntersectionFinder():
    """
     * Finds if two sets of {@link SegmentStrings}s intersect.
     *
     * Uses indexing for fast performance and to optimize repeated tests
     * against a target set of lines.
     * Short-circuited to return as soon an intersection is found.
     *
     * @version 1.7
    """
    def __init__(self, baseSegStrings):

        # MCIndexSegmentSetMutualIntersector
        self.si = MCIndexSegmentSetMutualIntersector()
        # LineIntersector
        self.li = LineIntersector()

        self.si.setBaseSegments(baseSegStrings)

    def intersects(self, segStrings, intDetector=None) -> bool:

        if intDetector is None:
            intDetector = SegmentIntersectionDetector(self.li)

        self.si.si = intDetector
        self.si.process(segStrings)
        logger.debug("FastSegmentSetIntersectionFinder.intersects(intersect:%s) lines:%s",
            intDetector.hasIntersection, len(segStrings))
        return intDetector.hasIntersection


class SingleInteriorIntersectionFinder(SegmentIntersector):
    """
     * Finds an interior intersection in a set of SegmentString,
     * if one exists.  Only the first intersection found is reported.
     *
     * @version 1.7
    """
    def __init__(self, li):
        """
         * Creates an intersection finder which finds an interior intersection
         * if one exists
         *
         * @param li the LineIntersector to use
        """
        self.li = li
        self.interiorIntersection = None
        self.intSegments = []

    @property
    def hasIntersection(self) -> bool:
        """
         * Tests whether an intersection was found.
         *
         * @return true if an intersection was found
        """
        return self.interiorIntersection is not None

    def processIntersections(self, e0, segIndex0: int, e1, segIndex1: int) -> None:
        """
         * This method is called by clients
         * of the {@link SegmentIntersector} class to process
         * intersections for two segments of the {@link SegmentStrings} being intersected.
         *
         * Note that some clients (such as {@link MonotoneChain}s) may optimize away
         * this call for segment pairs which they have determined do not intersect
         * (e.g. by an disjoint envelope test).
        """
        if self.hasIntersection:
            return

        # don't bother intersecting a segment with itself
        if e0 == e1 and segIndex0 == segIndex1:
            return

        p0 = e0.coords[segIndex0]
        p1 = e0.coords[segIndex0 + 1]
        q0 = e1.coords[segIndex1]
        q1 = e1.coords[segIndex1 + 1]

        self.li.computeLinesIntersection(p0, p1, q0, q1)

        if self.li.hasIntersection:
            if self.li.isInteriorIntersection:
                self.intSegments = [p0, p1, q0, q1]
                self.interiorIntersection = self.li.intersectionPts[0]

    @property
    def isDone(self) -> bool:
        return self.interiorIntersection is not None


class FastNodingValidator():
    """
     * Validates that a collection of {@link SegmentString}s is correctly noded.
     *
     * Uses indexes to improve performance.
     * Does NOT check a-b-a collapse situations.
     * Also does not check for endpt-interior vertex intersections.
     * This should not be a problem, since the noders should be
     * able to compute intersections between vertices correctly.
     * User may either test the valid condition, or request that a
     * {@link TopologyException}
     * be thrown.
     *
     * @version 1.7
    """
    def __init__(self, segStrings: list) -> None:
        self.li = LineIntersector()
        self.segStrings = segStrings
        self.si = None
        self._isValid = True

    @property
    def isValid(self) -> bool:
        """
         * Checks for an intersection and
         * reports if one is found.
         *
         * @return true if the arrangement contains an interior intersection
        """
        self.execute()
        return self._isValid

    def getErrorMessage(self) -> str:
        """
         * Returns an error message indicating the segments containing
         * the intersection.
         *
         * @return an error message documenting the intersection location
        """
        if self._isValid:
            return "no intersection found"

        return "found non noded intersection"

    def checkValid(self) -> None:
        """
         * Checks for an intersection and throws
         * a TopologyException if one is found.
         *
         * @throws TopologyException if an intersection is found
        """
        self.execute()
        if not self._isValid:
            raise TopologyException(self.getErrorMessage(), self.si.interiorIntersection)

    def execute(self) -> None:
        if self.si is not None:
            return
        self.checkInteriorIntersections()

    def checkInteriorIntersections(self):
        self._isValid = True
        self.si = SingleInteriorIntersectionFinder(self.li)
        noder = MCIndexNoder(self.si)
        noder.computeNodes(self.segStrings)
        if self.si.hasIntersection:
            self._isValid = False
            return


class IntersectionAdder(SegmentIntersector):
    """
     * Computes the intersections between two line segments in SegmentString
     * and adds them to each string.
     * The {@link SegmentIntersector} is passed to a {@link Noder}.
     * The {@link addIntersections} method is called whenever the {@link Noder}
     * detects that two SegmentStrings <i>might</i> intersect.
     * This class is an example of the <i>Strategy</i> pattern.
     *
    """
    def __init__(self, newLi):

        self.hasIntersection = False
        self.hasProper = False
        self.hasInterior = False
        self.hasProperInterior = False
        # the proper intersection point found
        # Coordinate
        self.properIntersectionPoint = None
        self.isSelfIntersection = False
        self.numInteriorIntersections = 0
        self.numProperIntersections = 0
        self.numIntersections = 0

        self._li = newLi
        self.isDone = False
        # testing only
        self.numTests = 0

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
            if self._li.intersections == 1:

                if self.isAdjacentSegments(segIndex0, segIndex1):
                    return True

                if e0.isClosed:
                    maxSegIndex = len(e0.coords) - 1
                    if ((segIndex0 == 0 and segIndex1 == maxSegIndex) or
                            (segIndex1 == 0 and segIndex0 == maxSegIndex)):
                        return True
        return False

    def processIntersections(self, e0, segIndex0: int, e1, segIndex1: int) -> None:
        """
         * This method is called by clients
         * of the {@link SegmentIntersector} class to process
         * intersections for two segments of the SegmentStrings being
         * intersected.
         * Note that some clients (such as MonotoneChains) may optimize away
         * this call for segment pairs which they have determined do not
         * intersect (e.g. by an disjoint envelope test).
        """
        # don't bother intersecting a segment with itself
        if e0 == e1 and segIndex0 == segIndex1:
            return

        _li = self._li

        self.numTests += 1

        p0 = e0.coords[segIndex0]
        p1 = e0.coords[segIndex0 + 1]
        q0 = e1.coords[segIndex1]
        q1 = e1.coords[segIndex1 + 1]

        _li.computeLinesIntersection(p0, p1, q0, q1)

        # No intersection, nothing to do
        if not _li.hasIntersection:
            return

        self.numIntersections += 1

        if _li.isInteriorIntersection:
            self.numInteriorIntersections += 1
            self.hasInterior = True

        # if the segments are adjacent they have at least
        # one trivial intersection,
        # the shared endpoint.  Don't bother adding it if it
        # is the only intersection.
        if not self.isTrivialIntersection(e0, segIndex0, e1, segIndex1):

            self.hasIntersection = True
            logger.debug("seg1:%s seg2:%s", segIndex0, segIndex1)
            e0.addIntersections(_li, segIndex0, 0)
            e1.addIntersections(_li, segIndex1, 1)

            if _li.isProper:

                self.numProperIntersections += 1
                self.hasProper = True
                self.hasProperInterior = True


class SegmentString():
    """
     * An interface for classes which represent a sequence of contiguous
     * line segments.
     * SegmentStrings can carry a context object, which is useful
     * for preserving topological or parentage information.
    """
    def __init__(self, context):
        # User-defined data for this segment string.
        self.context = context

    @property
    def isClosed(self) -> bool:
        raise NotImplementedError()


class SegmentStringUtil():
    """
     * Utility methods for processing {@link SegmentString}s.
     *
     * @author Martin Davis
     *
    """

    @staticmethod
    def extractSegmentStrings(geom, segStr: list) -> None:
        """
         * Extracts all linear components from a given {@link Geometry}
         * to {@link SegmentString}s.
         *
         * The SegmentString data item is set to be the source Geometry.
         *
         * @param geom the geometry to extract from
         * @param segStr a List of SegmentStrings (output parameter).
        """
        lines = []

        LinearComponentExtracter.getLines(geom, lines)
        logger.debug("SegmentStringUtil.extractSegmentStrings(%s)", len(lines))
        for line in lines:
            segStr.append(NodedSegmentString(line.coords, geom))


class BasicSegmentString(SegmentString):
    """
     * Represents a list of contiguous line segments,
     * and supports noding the segments.
     * The line segments are represented by an array of {@link Coordinate}s.
     * Intended to optimize the noding of contiguous segments by
     * reducing the number of allocated objects.
     * SegmentStrings can carry a context object, which is useful
     * for preserving topological or parentage information.
     * All noded substrings are initialized with the same context object.
    """
    def __init__(self, coords, context):
        SegmentString.__init__(self, context)
        self.coords = coords

    @property
    def isClosed(self) -> bool:
        return self.coords[0] == self.coords[-1]

    def getSegmentOctant(self, index: int) -> int:
        """
         * Gets the octant of the segment starting at vertex index.
         *
         * @param index the index of the vertex starting the segment.
         *        Must not be the last index in the vertex list
         * @return the octant of the segment at the vertex
        """
        if index >= len(self.coords) - 1:
            return -1

        return Octant.octant(self.coords[index], self.coords[index + 1])


class NodedSegmentString(SegmentString):
    """
     * Represents a list of contiguous line segments,
     * and supports noding the segments.
     *
     * The line segments are represented by an array of {@link Coordinate}s.
     * Intended to optimize the noding of contiguous segments by
     * reducing the number of allocated objects.
     * SegmentStrings can carry a context object, which is useful
     * for preserving topological or parentage information.
     * All noded substrings are initialized with the same context object.
    """
    def __init__(self, coords, context=None):
        """
         * Creates a new segment string from a list of vertices.
         *
         * @param coords CoordinateSequence representing the string,
         *               ownership transferred.
         *
         * @param data the user-defined data of this segment string
         *             (may be null)
        """
        SegmentString.__init__(self, context)

        # SegmentNodeList
        self.nodeList = SegmentNodeList(self)

        # CoordinateSequence
        self.coords = coords

    @property
    def size(self) -> int:
        return len(self.coords)

    def addIntersectionNode(self, coord, segmentIndex: int):
        """
         * Adds an intersection node for a given point and segment to this segment string.
         * If an intersection already exists for this exact location, the existing
         * node will be returned.
         *
         * @param coord the location of the intersection
         * @param segmentIndex the index of the segment containing the intersection
         * @return the intersection node for the point
        """
        normalizedSegmentIndex = segmentIndex
        # normalize the intersection point location
        nextSegIndex = normalizedSegmentIndex + 1

        if nextSegIndex < self.size:
            nextPt = self.coords[nextSegIndex]
            # Normalize segment index if intPt falls on vertex
            # The check for point equality is 2D only - Z values are ignored
            if coord == nextPt:
                normalizedSegmentIndex = nextSegIndex

        # SegmentNode
        ei = self.nodeList.add(coord, normalizedSegmentIndex)
        return ei

    @property
    def isClosed(self) -> bool:
        return self.coords[0] == self.coords[-1]

    def getSegmentOctant(self, index: int) -> int:
        """
         * Gets the octant of the segment starting at vertex index.
         *
         * @param index the index of the vertex starting the segment.
         *        Must not be the last index in the vertex list
         * @return the octant of the segment at the vertex
        """
        if index >= self.size - 1:
            return -1

        return NodedSegmentString.safeOctant(self.coords[index], self.coords[index + 1])

    def addIntersections(self, li, segmentIndex: int, geomIndex: int) -> None:
        """
         * Add {SegmentNode}s for one or both
         * intersections found for a segment of an edge to the edge
         * intersection list.
        """
        for i in range(li.intersections):
            self.addIntersection(li, segmentIndex, geomIndex, i)

    def addIntersection(self, li, segmentIndex: int, geomIndex: int, intIndex: int) -> None:
        """
         * Add an SegmentNode for intersection intIndex.
         *
         * An intersection that falls exactly on a vertex
         * of the SegmentString is normalized
         * to use the higher of the two possible segmentIndexes
        """
        coord = li.intersectionPts[intIndex]
        self._addIntersection(coord, segmentIndex)

    def _addIntersection(self, coord, segmentIndex: int) -> None:
        """
         * Add an SegmentNode for intersection intIndex.
         *
         * An intersection that falls exactly on a vertex
         * of the SegmentString is normalized
         * to use the higher of the two possible segmentIndexes
        """
        normalizedSegmentIndex = segmentIndex
        size = self.size

        if segmentIndex > size - 2:
            raise ValueError("SegmentString.addIntersection: SegmentIndex out of range")

        # normalize the intersection point location
        nextSegIndex = normalizedSegmentIndex + 1

        if nextSegIndex < size:
            # Coordinate
            nextPt = self.coords[nextSegIndex]

            # Normalize segment index if intPt falls on vertex
            # The check for point equality is 2D only
            if coord == nextPt:
                normalizedSegmentIndex = nextSegIndex
        """
         * Add the intersection point to edge intersection list
         * (unless the node is already known)
        """
        self.nodeList.add(coord, normalizedSegmentIndex)

    @staticmethod
    def safeOctant(p0, p1) -> int:
        if p0 == p1:
            return 0
        return Octant.octant(p0, p1)

    @staticmethod
    def getNodedSubStrings(segStrings: list, resultEdgeList: list=[]) -> list:

        for ss in segStrings:
            ss.nodeList.addSplitEdges(resultEdgeList)

        logger.debug("NodedSegmentString.getNodedSubStrings(%s)", len(resultEdgeList))

        return resultEdgeList


class SinglePassNoder():
    """
     * Base class for {@link Noder}s which make a single
     * pass to find intersections.
     * This allows using a custom {@link SegmentIntersector}
     * (which for instance may simply identify intersections, rather than
     * insert them).
    """
    def __init__(self, nSegInt=None):
        # SegmentIntersector
        self.si = nSegInt


class SegmentOverlapAction(MonotoneChainOverlapAction):

    def __init__(self, si):
        MonotoneChainOverlapAction.__init__(self)
        # SegmentIntersector
        self.si = si

    def overlap(self, mc1, start1: int, mc2, start2: int) -> None:
        # SegmentString
        ss1 = mc1.context
        ss2 = mc2.context
        self.si.processIntersections(ss1, start1, ss2, start2)


class MCIndexNoder(SinglePassNoder):
    """
     * Nodes a set of SegmentString using a index based
     * on index.chain.MonotoneChain and a index.SpatialIndex.
     *
     * The {@link SpatialIndex} used should be something that supports
     * envelope (range) queries efficiently (such as a index.quadtree.Quadtree
     * or index.strtree.STRtree.
    """
    def __init__(self, si=None):
        SinglePassNoder.__init__(self, si)
        self.idCounter = 0
        # SegmentString
        self.nodedSegStrings = None
        self.nOverlaps = 0
        # MonotoneChain
        self.monoChains = []
        self.index = STRtree()

    def computeNodes(self, inputSegStrings: list) -> None:
        self.nodedSegStrings = inputSegStrings

        for segStr in inputSegStrings:
            self.add(segStr)

        self.intersectChains()
        logger.debug("noder:[%s] index:[%s] MCIndexNoder.computeNodes(%s) overlaps:%s",
            id(self),
            id(self.index),
            len(self.monoChains),
            self.nOverlaps)

    def getNodedSubStrings(self) -> list:
        res = []
        NodedSegmentString.getNodedSubStrings(self.nodedSegStrings, res)
        return res

    def intersectChains(self) -> None:
        overlapAction = SegmentOverlapAction(self.si)

        for queryChain in self.monoChains:
            #
            overlapChains = []
            self.index.query(queryChain.envelope, overlapChains)
            for testChain in overlapChains:
                """
                 * following test makes sure we only compare each
                 * pair of chains once and that we don't compare a
                 * chain to itself
                """
                if testChain.id > queryChain.id:
                    queryChain.computeOverlaps(testChain, overlapAction)
                    self.nOverlaps += 1

                if self.si.isDone:
                    return

    def add(self, segStr) -> None:
        # MonotoneChain
        segChains = []
        # segChains will contain nelwy allocated MonotoneChain objects
        MonotoneChainBuilder.getChains(segStr.coords, segStr, segChains)

        for mc in segChains:
            mc.id = self.idCounter
            self.idCounter += 1
            self.index.insert(mc.envelope, mc)
            self.monoChains.append(mc)


class ScaledNoder():
    """
     * Wraps a {@link Noder} and transforms its input
     * into the integer domain.
     *
     * This is intended for use with Snap-Rounding noders,
     * which typically are only intended to work in the integer domain.
     * Offsets can be provided to increase the number of digits of
     * available precision.
     *
    """
    def __init__(self, noder, scale: float=1.0, offsetX: float=0.0, offsetY: float=0.0):
        # Noder
        self.noder = noder
        self.scale = scale
        self.offsetX = offsetX
        self.offsetY = offsetY
        self.isScaled = scale != 1.0

    @property
    def isIntegerPrecision(self) -> bool:
        return self.scale == 1.0

    def getNodedSubStrings(self):
        splitSS = self.noder.getNodedSubStrings()
        if self.isScaled:
            self._rescale(splitSS)
        return splitSS

    def computeNodes(self, inputSegStr):
        if self.isScaled:
            self._scale(inputSegStr)
        self.noder.computeNodes(inputSegStr)

    def _rescale(self, segStrings) -> None:
        rescaler = ScaledNoderRescaler(self)
        for ss in segStrings:
            ss.coords.apply_rw(rescaler)

    def _scale(self, segStrings) -> None:
        rescaler = ScaledNoderScaler(self)
        for ss in segStrings:
            coords = ss.coords.clone()
            coords.apply_rw(rescaler)
            ss.coords = CoordinateSequence.removeRepeatedPoints(coords)


class ScaledNoderScaler(CoordinateFilter):
    def __init__(self, scaleNoder):
        self.sn = scaleNoder

    def filter_rw(self, coord):
        coord.x = round((coord.x - self.sn.offsetX) * self.sn.scale)
        coord.y = round((coord.y - self.sn.offsetY) * self.sn.scale)


class ScaledNoderRescaler(CoordinateFilter):
    def __init__(self, scaleNoder):
        self.sn = scaleNoder

    def filter_rw(self, coord):
        coord.x = coord.x / self.sn.scale + self.sn.offsetX
        coord.y = coord.y / self.sn.scale + self.sn.offsetY
