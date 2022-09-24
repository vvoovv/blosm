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


from math import pi, cos, sin, log, pow, atan2, sqrt
from .shared import (
    logger,
    quicksort,
    TopologyException,
    GeomTypeId,
    PrecisionModel,
    Location,
    Position,
    Envelope,
    Coordinate,
    CoordinateSequence,
    Triangle,
    CAP_STYLE,
    JOIN_STYLE
    )
from .precision import GeometryPrecisionReducer
from .algorithms import (
    Angle,
    CGAlgorithms,
    HCoordinate,
    LineSegment,
    LineIntersector
    )
from .geomgraph import (
    PlanarGraph,
    EdgeList,
    Edge,
    Label
    )
from .noding import (
    ScaledNoder,
    NodedSegmentString,
    IntersectionAdder,
    MCIndexNoder
    )
from .op_overlay import (
    OverlayOp,
    SnapOverlayOp,
    PolygonBuilder,
    OverlayNodeFactory
    )
from .op_linemerge import (
    LineMerger
    )


class BUFFER_DEFAULT():
    # The default number of facets into which to divide a fillet
    # of 90 degrees.
    # A value of 8 gives less than 2% max error in the buffer distance.
    # For a max error of < 1%, use QS = 12.
    # For a max error of < 0.1%, use QS = 18.
    #
    QUADRANT_SEGMENTS = 8

    # The default mitre limit
    # Allows fairly pointy mitres.
    #
    MITRE_LIMIT = 5.0


class BufferParameters():

    def __init__(self,
            quadrantSegments=BUFFER_DEFAULT.QUADRANT_SEGMENTS,
            endCapStyle: int=CAP_STYLE.round,
            joinStyle: int=JOIN_STYLE.round,
            mitreLimit: float=BUFFER_DEFAULT.MITRE_LIMIT
            ):
        if type(quadrantSegments).__name__ == 'BufferParameters':
            bp = quadrantSegments
            # Defaults to QUADRANT_SEGMENTS
            self.quadrantSegments = bp.quadrantSegments
            # Defaults to CAP_ROUND
            self.endCapStyle = bp.endCapStyle
            # Defaults to JOIN_ROUND
            self.joinStyle = bp.joinStyle
            # Defaults to MITRE_LIMIT
            self.mitreLimit = bp.mitreLimit
            """
             * Tests whether the buffer is to be generated on a single side only.
             *
             * @return true if the generated buffer is to be single-sided
            """
            self.isSingleSided = bp.isSingleSided

        else:

            # Defaults to QUADRANT_SEGMENTS
            self.quadrantSegments = quadrantSegments
            # Defaults to CAP_ROUND
            self.endCapStyle = endCapStyle
            # Defaults to JOIN_ROUND
            self.joinStyle = joinStyle
            # Defaults to MITRE_LIMIT
            self.mitreLimit = mitreLimit
            """
             * Tests whether the buffer is to be generated on a single side only.
             *
             * @return true if the generated buffer is to be single-sided
            """
            if quadrantSegments != BUFFER_DEFAULT.QUADRANT_SEGMENTS:
                self.setQuadrantSegments(quadrantSegments)

            if joinStyle != JOIN_STYLE.round:
                self.joinStyle = joinStyle

            if mitreLimit != BUFFER_DEFAULT.MITRE_LIMIT:
                self.mitreLimit = mitreLimit

            self.isSingleSided = False

    def setQuadrantSegments(self, quadSegs: int) -> None:
        self.quadrantSegments = quadSegs
        """
         * Indicates how to construct fillets.
         * If qs >= 1, fillet is round, and qs indicates number of
         * segments to use to approximate a quarter-circle.
         * If qs = 0, fillet is bevelled flat (i.e. no filleting is performed)
         * If qs < 0, fillet is mitred, and absolute value of qs
         * indicates maximum length of mitre according to
         *
         * mitreLimit = |qs|
        """
        if quadSegs == 0:
            self.joinStyle = JOIN_STYLE.bevel

        elif quadSegs < 0:
            self.joinStyle = JOIN_STYLE.mitre
            self.mitreLimit = abs(quadSegs)

        if quadSegs <= 0:
            self.quadrantSegments = 1
        """
         * If join style was set by the quadSegs value,
         * use the default for the actual quadrantSegments value.
        """
        if self.joinStyle != JOIN_STYLE.round:
            self.quadrantSegments = BUFFER_DEFAULT.QUADRANT_SEGMENTS

    @staticmethod
    def bufferDistanceError(quadSegs: int) -> float:
        """
         * Computes the maximum distance error due to a given level
         * of approximation to a true arc.
         * @param quadSegs the number of segments used to approximate
         *          a quarter-circle
         * @return the error of approximation
        """
        alpha = pi / 2.0 / quadSegs
        return 1 - cos(alpha / 2.0)


class BufferOp():
    """
     * Computes the buffer of a geometry, for both positive and negative
     * buffer distances.
     *
     * In GIS, the positive (or negative) buffer of a geometry is defined as
     * the Minkowski sum (or difference) of the geometry
     * with a circle with radius equal to the absolute value of the buffer
     * distance.
     * In the CAD/CAM world buffers are known as </i>offset curves</i>.
     * In morphological analysis the operation of positive and negative buffering
     * is referred to as <i>erosion</i> and <i>dilation</i>.
     *
     * The buffer operation always returns a polygonal result.
     * The negative or zero-distance buffer of lines and points is always
     * an empty Polygon.
     *
     * Since true buffer curves may contain circular arcs,
     * computed buffer polygons can only be approximations to the true geometry.
     * The user can control the accuracy of the curve approximation by specifying
     * the number of linear segments with which to approximate a curve.
     *
     * The end cap style of a linear buffer may be specified.
     * The following end cap styles are supported:
     * - CAP_ROUND - the usual round end caps
     * - CAP_BUTT - end caps are truncated flat at the line ends
     * - CAP_SQUARE - end caps are squared off at the buffer distance
     *   beyond the line ends
    """

    """
     *  A number of digits of precision which leaves some computational "headroom"
     *  for floating point operations.
     *
     *  This value should be less than the decimal precision of double-precision values (16).
    """

    MAX_PRECISION_DIGITS = 12
    MIN_PRECISION_DIGITS = 6

    def __init__(self, geom, params):
        """
         * Initializes a buffer computation for the given geometry
         * with the given set of parameters
         *
         * @param g the geometry to buffer
         * @param params the buffer parameters to use. This class will
         *               copy it to private memory.
        """
        # Geometry
        self.geom = geom
        self.bufParams = params

        # TopologyException
        self.saveException = None

        self.distance = 0

        # Geometry
        self.result = None
        
    def computeGeometry(self) -> None:
        self.bufferOriginalPrecision()

        if self.result is not None:
            return

        self.bufferReducedPrecision()

    def bufferOriginalPrecision(self) -> None:
        bufBuilder = BufferBuilder(self.bufParams)
        try:
            self.result = bufBuilder.buffer(self.geom, self.distance)
            logger.debug("Buffer original precision success")

        except TopologyException as ex:
            self.saveException = ex
            logger.warning("Buffer original precision failed %s", ex)

            # self.geom._factory.output(self.geom, "buffer error", False)

            # don't propagate the exception - it will be detected by fact that resultGeometry is null
            pass

    def bufferReducedPrecision(self) -> None:
        # try and compute with decreasing precision,
        # up to a min, to avoid gross results
        for precDigits in range(BufferOp.MAX_PRECISION_DIGITS, BufferOp.MIN_PRECISION_DIGITS, -1):
            try:
                self._bufferReducedPrecision(precDigits)
                logger.debug("Buffer reduced precision success with %s digits", precDigits)
            except TopologyException as ex:
                self.saveException = ex
                logger.warning("Buffer reduced precision failed digits:%s %s", precDigits, ex)

                # self.geom._factory.output(self.geom, "buffer error", False)

                # don't propagate the exception - it will be detected by fact that resultGeometry is null
                pass

            if self.result is not None:
                return
        # tried everything - have to bail
        raise self.saveException

    def _bufferReducedPrecision(self, precisionDigits: int) -> None:
        sizeBasedScaleFactor = BufferOp.precisionScaleFactor(self.geom, self.distance, precisionDigits)
        fixedPM = PrecisionModel(scale=sizeBasedScaleFactor)
        self.bufferFixedPrecision(fixedPM)

    def bufferFixedPrecision(self, fixedPM):
        pm = PrecisionModel(scale=1.0)

        # MCIndexSnapRounder seems to be still bogus
        # inoder = MCIndexSnapRounder(pm)
        
        li = LineIntersector(pm)
        ia = IntersectionAdder(li)
        inoder = MCIndexNoder(ia)
        noder = ScaledNoder(inoder, scale=fixedPM.scale)
        bufBuilder = BufferBuilder(self.bufParams)
        bufBuilder.workingPrecisionModel = fixedPM
        bufBuilder.workingNoder = noder

        # Reduce precision of the input geometry
        #
        # NOTE: this reduction is not in JTS and should supposedly
        #       not be needed because the PrecisionModel we pass
        #       to the BufferBuilder above (with setWorkingPrecisionModel)
        #       should be used to round coordinates emitted by the
        #       OffsetCurveBuilder, thus effectively producing a fully
        #       rounded input to the noder.
        #       Nonetheless the amount of scrambling done by rounding here
        #       is known to fix at least one case in which MCIndexNoder
        #       would fail: http://trac.osgeo.org/geos/ticket/605
        #
        # TODO: follow JTS in MCIndexSnapRounder usage

        argPm = self.geom._factory.precisionModel
        workGeom = self.geom
        if argPm.modelType != PrecisionModel.FIXED or argPm.scale != fixedPM.scale:
            # prevent Topology check infinite loop
            workGeom = GeometryPrecisionReducer.reduce(self.geom, precisionModel=fixedPM, preventTopologyCheck=False)

        self.result = bufBuilder.buffer(workGeom, self.distance)

    @staticmethod
    def bufferOp(geom,
            distance: float,
            quadrantSegments: int=BUFFER_DEFAULT.QUADRANT_SEGMENTS,
            endCapStyle: int=CAP_STYLE.round,
            joinStyle: int=JOIN_STYLE.round,
            mitreLimit: float=BUFFER_DEFAULT.MITRE_LIMIT,
            singleSided: bool=False
            ):
        """
         * Computes the buffer for a geometry for a given buffer distance
         * and accuracy of approximation.
         *
         * @param g the geometry to buffer
         * @param distance the buffer distance
         * @param quadrantSegments the number of segments used to
         *        approximate a quarter circle
         * @return the buffer of the input geometry
        """
        bufParams = BufferParameters(quadrantSegments,
                        endCapStyle,
                        joinStyle,
                        mitreLimit)
        bufParams.isSingleSided = singleSided
        bufOp = BufferOp(geom, bufParams)
        logger.debug("******************************\n")
        logger.debug("BufferOp.bufferOp(%s)\n", distance)
        logger.debug("******************************")
        return bufOp.getResultGeometry(distance)

    @staticmethod
    def offsetCurveOp(geom,
            distance: float,
            quadrantSegments: int=BUFFER_DEFAULT.QUADRANT_SEGMENTS,
            joinStyle: int=JOIN_STYLE.round,
            mitreLimit: float=BUFFER_DEFAULT.MITRE_LIMIT,
            ):

        if joinStyle > JOIN_STYLE.bevel:
            joinStyle = JOIN_STYLE.bevel

        bufParams = BufferParameters(quadrantSegments,
                        CAP_STYLE.flat,
                        joinStyle,
                        mitreLimit)

        isLeftSide = True

        bufParams.isSingleSided = True

        if distance < 0:
            isLeftSide = False
            distance = -distance

        bufBuilder = BufferBuilder(bufParams)
        logger.debug("******************************\n")
        logger.debug("BufferOp.offsetCurveOp(%s)\n", distance)
        logger.debug("******************************")

        return bufBuilder.bufferLineSingleSided(geom, distance, isLeftSide)

    def setQuadrantSegments(self, quadSegs: int) -> None:
        """
         * Specifies the end cap style of the generated buffer.
         * The styles supported are CAP_ROUND, CAP_BUTT, and CAP_SQUARE.
         * The default is CAP_ROUND.
         *
         * @param endCapStyle the end cap style to specify
        """
        self.bufParams.setQuadrantSegments(quadSegs)

    def setEndCapStyle(self, endCapStyle: int) -> None:
        """
         * Specifies the end cap style of the generated buffer.
         * The styles supported are CAP_ROUND, CAP_BUTT, and CAP_SQUARE.
         * The default is CAP_ROUND.
         *
         * @param endCapStyle the end cap style to specify
        """
        self.bufParams.endCapStyle = endCapStyle

    def setSingleSided(self, singleSided: bool) -> None:
        """
         * Sets whether the computed buffer should be single-sided.
         *
         * A single-sided buffer is constructed on only one side
         * of each input line.
         *
         * The side used is determined by the sign of the buffer distance:
         * - a positive distance indicates the left-hand side
         * - a negative distance indicates the right-hand side
         *
         * The single-sided buffer of point geometries is
         * the same as the regular buffer.
         *
         * The End Cap Style for single-sided buffers is
         * always ignored,
         * and forced to the equivalent of <tt>CAP_FLAT</tt>.
         *
         * @param isSingleSided true if a single-sided buffer
         *                      should be constructed
        """
        self.bufParams.isSingleSided = singleSided

    def getResultGeometry(self, distance: float):
        """
         * Returns the buffer computed for a geometry for a given buffer
         * distance.
         *
         * @param g the geometry to buffer
         * @param distance the buffer distance
         * @return the buffer of the input geometry
        """
        self.distance = distance
        self.computeGeometry()
        return self.result

    @staticmethod
    def precisionScaleFactor(geom, distance: float, maxPrecisionDigits: int) -> float:
        """
         * Compute a reasonable scale factor to limit the precision of
         * a given combination of Geometry and buffer distance.
         * The scale factor is based on a heuristic.
         *
         * @param g the Geometry being buffered
         *
         * @param distance the buffer distance
         *
         * @param maxPrecisionDigits the mzx # of digits that should be
         *        allowed by the precision determined by the
         *        computed scale factor
         *
         * @return a scale factor that allows a reasonable amount of
         *         precision for the buffer computation
        """
        env = geom.envelope
        envMax = max([abs(env.maxx), abs(env.minx),
            abs(env.maxy), abs(env.miny)])
        if distance > 0.0:
            expandByDistance = distance
        else:
            expandByDistance = 0.0

        buffEnvMax = envMax + 2 * expandByDistance

        # the smallest power of 10 greater than the buffer envelope
        buffEnvPrecisionDigits = int(log(buffEnvMax) / log(10) + 1.0)
        minUnitLog10 = maxPrecisionDigits - buffEnvPrecisionDigits

        scaleFactor = pow(10, minUnitLog10)

        return scaleFactor


def bufferSubgraphGT(first, second) -> bool:
    return first.compareTo(second) > 0


class RightmostEdgeFinder():
    """
     * A RightmostEdgeFinder find the geomgraph.DirectedEdge in a list which has
     * the highest coordinate, and which is oriented L to R at that point.
     * (I.e. the right side is on the RHS of the edge.)
    """
    def __init__(self):
        # int
        self.minIndex = -1

        # geom.Coordinate minCoord
        self.coord = None

        # geomgraph.DirectedEdge
        self.minDe = None

        # geomgraph.DirectedEdge orientedDe
        self.edge = None

    def findRightmostEdgeAtNode(self) -> None:
        node = self.minDe.node
        star = node.star
        # Warning! NULL could be returned if the star is empty!
        self.minDe = star.getRightmostEdge()

        # the DirectedEdge returned by the previous call is not
        # necessarily in the forward direction. Use the sym edge if it isn't.
        if not self.minDe.isForward:
            self.minDe = self.minDe.sym
            minEdge = self.minDe.edge

            minEdgeCoords = minEdge.coords
            self.minIndex = len(minEdgeCoords) - 1

    def findRightmostEdgeAtVertex(self) -> None:
        """
         * The rightmost point is an interiors vertex, so it has
         * a segment on either side of it.
         * If these segments are both above or below the rightmost
         * point, we need to determine their relative orientation
         * to decide which is rightmost.
        """
        minEdge = self.minDe.edge
        coords = minEdge.coords

        pPrev = coords[self.minIndex - 1]
        pNext = coords[self.minIndex + 1]
        orientation = CGAlgorithms.computeOrientation(self.coord, pNext, pPrev)

        usePrev = False
        # both segments are below min point
        if (pPrev.y < self.coord.y and pNext.y < self.coord.y and
                orientation == CGAlgorithms.COUNTERCLOCKWISE):
            usePrev = True
        elif (pPrev.y > self.coord.y and pNext.y > self.coord.y and
                orientation == CGAlgorithms.CLOCKWISE):
            usePrev = True

        # if both segments are on the same side, do nothing - either is safe
        # to select as a rightmost segment
        if usePrev:
            self.minIndex = self.minIndex - 1

    def checkForRightmostCoordinate(self, de) -> None:
        edge = de.edge
        coords = edge.coords
        ncoords = len(coords) - 1
        # only check vertices which are the starting point of
        # a non-horizontal segment
        for i in range(ncoords):
            coord = coords[i]
            if self.coord is None or coord.x > self.coord.x:
                self.minDe = de
                self.minIndex = i
                self.coord = coord

    def getRightmostSide(self, de, index: int) -> int:

        side = self.getRightmostSideOfSegment(de, index)

        if side < 0:
            side = self.getRightmostSideOfSegment(de, index - 1)

        if side < 0:
            self.coord = None
            self.checkForRightmostCoordinate(de)

        return side

    def getRightmostSideOfSegment(self, de, i: int) -> int:
        edge = de.edge
        coords = edge.coords

        if i < 0 or i + 1 >= len(coords):
            return -1

        # indicates edge is parallel to x-axis
        if coords[i].y == coords[i + 1].y:
            return -1

        pos = Position.LEFT

        if coords[i].y < coords[i + 1].y:
            pos = Position.RIGHT

        return pos

    def findEdge(self, edgeEnds: list) -> None:

        checked = 0

        # Check all forward DirectedEdges only.  This is still general,
        # because each edge has a forward DirectedEdge.

        for de in edgeEnds:
            if not de.isForward:
                continue
            self.checkForRightmostCoordinate(de)
            checked += 1

        if self.minDe is None:
            raise TopologyException("No forward edges found in buffer subgraph")

        # If the rightmost point is a node, we need to identify which of
        # the incident edges is rightmost.

        if self.minIndex == 0:
            self.findRightmostEdgeAtNode()
        else:
            self.findRightmostEdgeAtVertex()

        # now check that the extreme side is the R side.
        # If not, use the sym instead.

        self.edge = self.minDe
        rightmostSide = self.getRightmostSide(self.minDe, self.minIndex)
        if rightmostSide == Position.LEFT:
            self.edge = self.minDe.sym


class BufferSubGraph():
    """
     * A connected subset of the graph of DirectedEdge and geomgraph.Node.
     *
     * Its edges will generate either
     * - a single polygon in the complete buffer, with zero or more interiors, or
     * -  ne or more connected interiors
    """
    def __init__(self):

        self.finder = RightmostEdgeFinder()
        # geomgraph.DirectedEdge
        self._edgeEnds = []
        # geomgraph.Node
        self.nodes = []
        # geom.Coordinate
        self.rightMostCoord = None
        # geom.Envelope
        self._env = None

    def addReachable(self, startNode) -> None:
        """
         * Adds all nodes and edges reachable from this node to the subgraph.
         *
         * Uses an explicit stack to avoid a large depth of recursion.
         *
         * @param node a node known to be in the subgraph
        """
        nodeStack = []
        nodeStack.append(startNode)
        while (len(nodeStack) > 0):
            node = nodeStack.pop()
            self.add(node, nodeStack)

    def add(self, node, nodeStack: list) -> None:
        """
         * Adds the argument node and all its out edges to the subgraph
         * @param node the node to add
         * @param nodeStack the current set of nodes being traversed
        """
        node.isVisited = True
        self.nodes.append(node)
        star = node.star
        for de in star.edges:
            self._edgeEnds.append(de)
            sym = de.sym
            symNode = sym.node
            """
             * NOTE: this is a depth-first traversal of the graph.
             * This will cause a large depth of recursion.
             * It might be better to do a breadth-first traversal.
            """
            if not symNode.isVisited:
                nodeStack.append(symNode)

    def clearVisitedEdges(self) -> None:
        for de in self._edgeEnds:
            de.isVisited = False

    def computeDepth(self, outsideDepth: int) -> None:
        self.clearVisitedEdges()

        # find an outside edge to assign depth to
        de = self.finder.edge

        logger.debug("BufferSubGraph.computeDepth(outside depth:%s)", outsideDepth)

        # right side of line returned by finder is on the outside
        de.setEdgeDepths(Position.RIGHT, outsideDepth)
        self.copySymDepths(de)
        self.computeDepths(de)

    def computeDepths(self, startEdge) -> None:
        """
         * Compute depths for all dirEdges via breadth-first traversal
         * of nodes in graph
         *
         * @param startEdge edge to start processing with
        """
        # Node
        nodesVisited = {}
        nodeQueue = []
        startNode = startEdge.node
        nodeQueue.append(startNode)
        startEdge.isVisited = True

        while (len(nodeQueue) > 0):
            node = nodeQueue.pop(0)

            nodesVisited[id(node)] = node

            # compute depths around node, starting at this edge since it has depths assigned
            self.computeNodeDepth(node)

            # add all adjacent nodes to process queue,
            # unless the node has been visited already
            star = node.star
            for de in star.edges:
                sym = de.sym
                if sym.isVisited:
                    continue
                adjNode = sym.node
                if nodesVisited.get(id(adjNode)) is None:
                    nodeQueue.append(adjNode)

    def computeNodeDepth(self, node) -> None:
        # find a visited dirEdge to start at
        # DirectedEdge
        startEdge = None
        # DirectedEdgeStar
        star = node.star
        for de in star.edges:
            if de.isVisited or de.sym.isVisited:
                startEdge = de
                break

        # only compute string append if assertion would fail
        if startEdge is None:
            raise TopologyException(
                "unable to find edge to compute depths at",
                node.coord
                )

        star.computeDepths(startEdge)

        # copy depths to sym edges
        for de in star.edges:
            de.isVisited = True
            self.copySymDepths(de)

    def copySymDepths(self, de) -> None:
        # logger.debug("BufferSubGraph.copySymDepths() %s, %s", de.getDepth(Position.LEFT), de.getDepth(Position.RIGHT))
        sym = de.sym
        sym.setDepth(Position.LEFT, de.getDepth(Position.RIGHT))
        sym.setDepth(Position.RIGHT, de.getDepth(Position.LEFT))

    def create(self, node) -> None:
        """
         * Creates the subgraph consisting of all edges reachable from
         * this node.
         *
         * Finds the edges in the graph and the rightmost coordinate.
         *
         * @param node a node to start the graph traversal from
        """
        self.addReachable(node)
        # We are assuming that _edgeEnds
        # contains *at leas* ONE forward DirectedEdge
        self.finder.findEdge(self._edgeEnds)
        self.rightMostCoord = self.finder.coord

    def findResultEdges(self) -> None:
        """
         * Find all edges whose depths indicates that they are in the
         * result area(s).
         *
         * Since we want polygon exteriors to be
         * oriented CW, choose dirEdges with the interiors of the result
         * on the RHS.
         * Mark them as being in the result.
         * Interior Area edges are the result of dimensional collapses.
         * They do not form part of the result area boundary.
        """
        for de in self._edgeEnds:
            """
             * Select edges which have an interiors depth on the RHS
             * and an exterior depth on the LHS.
             * Note that because of weird rounding effects there may be
             * edges which have negative depths!  Negative depths
             * count as "outside".
            """

            """
            logger.debug("BufferSubGraph.findResultEdges() isInterior:%s dirEdge:%s\ndepth left:%s\ndepth right:%s",
                de.isInteriorAreaEdge,
                de,
                de.getDepth(Position.LEFT),
                de.getDepth(Position.RIGHT))
            """

            if (de.getDepth(Position.RIGHT) >= 1 and
                    de.getDepth(Position.LEFT) <= 0 and
                    not de.isInteriorAreaEdge):
                de.isInResult = True

    def compareTo(self, other) -> int:
        """
         * BufferSubgraphs are compared on the x-value of their rightmost
         * Coordinate.
         *
         * This defines a partial ordering on the graphs such that:
         *
         * g1 >= g2 <=>= Ring(g2) does not contain Ring(g1)
         *
         * where Polygon(g) is the buffer polygon that is built from g.
         *
         * This relationship is used to sort the BufferSubgraphs so
         * that exteriors are guaranteed to
         * be built before interiors.
        """
        if self.rightMostCoord.x < other.rightMostCoord.x:
            return -1
        if self.rightMostCoord.x > other.rightMostCoord.x:
            return 1
        return 0

    @property
    def envelope(self):
        """
         * Computes the envelope of the edges in the subgraph.
         * The envelope is cached after being computed.
         *
         * @return the envelope of the graph.
        """
        if self._env is None:
            self._env = Envelope()
            for de in self._edgeEnds:
                coords = de.edge.coords
                for co in coords:
                    self._env.expandToInclude(co)
        return self._env


class DepthSegment():
    """
     * A segment from a directed edge which has been assigned a depth value
     * for its sides.
    """
    def __init__(self, seg, depth: int):

        self.upwardSeg = seg
        self.leftDepth = depth

    @staticmethod
    def compareX(seg0, seg1) -> int:
        """
         * Compare two collinear segments for left-most ordering.
         * If segs are vertical, use vertical ordering for comparison.
         * If segs are equal, return 0.
         * Segments are assumed to be directed so that the second
         * coordinate is >= to the first
         * (e.g. up and to the right).
         *
         * @param seg0 a segment to compare
         * @param seg1 a segment to compare
         * @return
        """
        compare0 = seg0.p0.compareTo(seg1.p0)
        if compare0 != 0:
            return compare0
        return seg0.p1.compareTo(seg1.p1)

    def compareTo(self, other) -> int:
        """
         * Defines a comparision operation on DepthSegments
         * which orders them left to right
         *
         * <pre>
         * DS1 < DS2   if   DS1.seg is left of DS2.seg
         * DS1 > DS2   if   DS1.seg is right of DS2.seg
         * </pre>
         *
         * @param obj
         * @return
        """
        """
         * try and compute a determinate orientation for the segments.
         * Test returns 1 if other is left of this (i.e. this > other)
        """
        orientIndex = self.upwardSeg.orientationIndex(other.upwardSeg)

        """
         * If comparison between this and other is indeterminate,
         * try the opposite call order.
         * orientationIndex value is 1 if this is left of other,
         * so have to flip sign to get proper comparison value of
         * -1 if this is leftmost
        """
        if orientIndex == 0:
            orientIndex = -1 * other.upwardSeg.orientationIndex(self.upwardSeg)

        # if orientation is determinate, return it
        if orientIndex != 0:
            return orientIndex

        # otherwise, segs must be collinear - sort based on minimum X value
        return DepthSegment.compareX(self.upwardSeg, other.upwardSeg)


def depthSegmentLessThen(first, second) -> bool:
    return first.compareTo(second) < 0


class SubgraphDepthLocater():
    """
     * Locates a subgraph inside a set of subgraphs,
     * in order to determine the outside depth of the subgraph.
     *
     * The input subgraphs are assumed to have had depths
     * already calculated for their edges.
    """
    def __init__(self, subgraphs):

        # BufferSubgraph
        self.subgraphs = subgraphs
        # geom.LineSegment
        self.seg = LineSegment()

    def getDepth(self, coord) -> int:
        stabbedSegments = []
        self.findStabbedSegments(coord, stabbedSegments)

        # if no segments on stabbing line subgraph must be outside all others
        if len(stabbedSegments) == 0:
            return 0

        quicksort(stabbedSegments, depthSegmentLessThen)
        ds = stabbedSegments[0]
        return ds.leftDepth

    def findStabbedSegments(self, stabbingRayLeftPt, stabbedSegments: list) -> None:
        """
         * Finds all non-horizontal segments intersecting the stabbing line.
         * The stabbing line is the ray to the right of stabbingRayLeftPt.
         *
         * @param stabbingRayLeftPt the left-hand origin of the stabbing line
         * @param stabbedSegments a vector to which DepthSegments intersecting
         *        the stabbing line will be added.
        """
        for bsg in self.subgraphs:
            env = bsg.envelope
            if not env.contains(stabbingRayLeftPt):
                continue
            self._findStabbedSegments(stabbingRayLeftPt, bsg._edgeEnds, stabbedSegments)

    def _findStabbedSegments(self, stabbingRayLeftPt, dirEdges: list, stabbedSegments: list) -> None:
        """
         * Finds all non-horizontal segments intersecting the stabbing line
         * in the list of dirEdges.
         * The stabbing line is the ray to the right of stabbingRayLeftPt.
         *
         * @param stabbingRayLeftPt the left-hand origin of the stabbing line
         * @param stabbedSegments the current vector of DepthSegments
         *        intersecting the stabbing line will be added.
        """
        # Check all forward DirectedEdges only. This is still general,
        # because each Edge has a forward DirectedEdge.
        for de in dirEdges:
            if not de.isForward:
                self._findStabbedSegment(stabbingRayLeftPt, de, stabbedSegments)

    def _findStabbedSegment(self, stabbingRayLeftPt, dirEdge, stabbedSegments: list) -> None:
        """
         * Finds all non-horizontal segments intersecting the stabbing line
         * in the input dirEdge.
         * The stabbing line is the ray to the right of stabbingRayLeftPt.
         *
         * @param stabbingRayLeftPt the left-hand origin of the stabbing line
         * @param stabbedSegments the current list of DepthSegments intersecting
         *        the stabbing line
        """
        coords = dirEdge.edge.coords
        for i in range(len(coords) - 1):
            low = coords[i]
            high = coords[i + 1]
            swap = False
            if low.y > high.y:
                swap = True
                high, low = low, high

            maxx = max(low.x, high.x)

            if maxx < stabbingRayLeftPt.x:
                continue

            if low.y == high.y:
                continue

            if stabbingRayLeftPt.y < low.y or stabbingRayLeftPt.y > high.y:
                continue

            if CGAlgorithms.computeOrientation(low, high, stabbingRayLeftPt) == CGAlgorithms.RIGHT:
                continue

            if swap:
                depth = dirEdge.getDepth(Position.RIGHT)
            else:
                depth = dirEdge.getDepth(Position.LEFT)

            self.seg.p0 = low
            self.seg.p1 = high

            ds = DepthSegment(self.seg, depth)
            stabbedSegments.append(ds)


class OffsetSegmentString():
    """
     * A dynamic list of the vertices in a constructed offset curve.
     *
     * Automatically removes close vertices
     * which are closer than a given tolerance.
     *
     * @author Martin Davis
    """
    def __init__(self):
        self.coords = CoordinateSequence()
        self.precisionModel = None
        """
         * The distance below which two adjacent points on the curve
         * are considered to be coincident.
         *
         * This is chosen to be a small fraction of the offset distance.
        """
        self.minimumVertexDistance = 0.0

    def reset(self) -> None:
        if self.coords is not None:
            self.coords.clear()
        else:
            self.coords = CoordinateSequence()

        self.precisionModel = None
        self.minimumVertexDistance = 0.0

    def addPt(self, coord) -> None:
        pt = Coordinate(coord.x, coord.y)
        self.precisionModel.makePrecise(pt)
        if self.isRedundant(pt):
            return
        # allow repeated as we checked this ourself
        self.coords.append(pt)

    def addPts(self, coords, isForward) -> None:
        if isForward:
            for coord in coords:
                self.addPt(coord)
        else:
            for coord in reversed(coords):
                self.addPt(coord)

    def closeRing(self) -> None:
        if len(self.coords) < 1:
            return
        if self.coords[0] == self.coords[-1]:
            return
        self.coords.append(self.coords[0])

    def isRedundant(self, coord) -> bool:
        """
         * Tests whether the given point is redundant relative to the previous
         * point in the list (up to tolerance)
         *
         * @param pt
         * @return true if the point is redundant
        """
        if len(self.coords) < 1:
            return False
        lastPt = self.coords[-1]
        dist = coord.distance(lastPt)
        return dist < self.minimumVertexDistance


class BufferInputLineSimplifier():
    """
     * Simplifies a buffer input line to
     * remove concavities with shallow depth.
     *
     * The most important benefit of doing this
     * is to reduce the number of points and the complexity of
     * shape which will be buffered.
     * It also reduces the risk of gores created by
     * the quantized fillet arcs (although this issue
     * should be eliminated in any case by the
     * offset curve generation logic).
     *
     * A key aspect of the simplification is that it
     * affects inside (concave or inward) corners only.
     * Convex (outward) corners are preserved, since they
     * are required to ensure that the generated buffer curve
     * lies at the correct distance from the input geometry.
     *
     * Another important heuristic used is that the end segments
     * of the input are never simplified.  This ensures that
     * the client buffer code is able to generate end caps faithfully.
     *
     * No attempt is made to avoid self-intersections in the output.
     * This is acceptable for use for generating a buffer offset curve,
     * since the buffer algorithm is insensitive to invalid polygonal
     * geometry.  However,
     * this means that this algorithm
     * cannot be used as a general-purpose polygon simplification technique.
     *
     * @author Martin Davis
    """
    NUM_PTS_TO_CHECK = 10
    INIT = 0
    DELETE = 1
    KEEP = 1

    def __init__(self, coords):
        self.coords = coords
        self.angleOrientation = CGAlgorithms.COUNTERCLOCKWISE
        self.distance = 0.0
        self.isDeleted = None

    @staticmethod
    def simplify(coords, distance):
        """
         * Simplify the input coordinate list.
         *
         * If the distance tolerance is positive,
         * concavities on the LEFT side of the line are simplified.
         * If the supplied distance tolerance is negative,
         * concavities on the RIGHT side of the line are simplified.
         *
         * @param inputLine the coordinate sequence to simplify
         * @param distanceTol simplification distance tolerance to use
         * @return a simplified version of the coordinate sequence
        """
        bs = BufferInputLineSimplifier(coords)
        return bs._simplify(distance)

    def _simplify(self, distance):
        self.distance = abs(distance)

        if distance < 0:
            self.angleOrientation = CGAlgorithms.CLOCKWISE

        startValue = BufferInputLineSimplifier.INIT
        self.isDeleted = [startValue for i in range(len(self.coords))]

        isChanged = False
        while (True):
            isChanged = self.deleteShallowConcavities()
            if not isChanged:
                break

        return self.collapseLine()

    def deleteShallowConcavities(self) -> bool:
        """
         * Uses a sliding window containing 3 vertices to detect shallow angles
         * in which the middle vertex can be deleted, since it does not
         * affect the shape of the resulting buffer in a significant way.
         * @return
        """
        # Do not simplify end line segments of the line string.
        # This ensures that end caps are generated consistently.
        index = 1
        midIndex = self.findNextNonDeletedIndex(index)
        lastIndex = self.findNextNonDeletedIndex(midIndex)

        isChanged = False

        while lastIndex < len(self.coords):
            isMiddleVertexDeleted = False

            if self.isDeletable(index, midIndex, lastIndex, self.distance):
                self.isDeleted[midIndex] = BufferInputLineSimplifier.DELETE
                isMiddleVertexDeleted = True
                isChanged = True

            if isMiddleVertexDeleted:
                index = lastIndex
            else:
                index = midIndex

            midIndex = self.findNextNonDeletedIndex(index)
            lastIndex = self.findNextNonDeletedIndex(midIndex)

        return isChanged

    def findNextNonDeletedIndex(self, index: int)-> int:
        """
         * Finds the next non-deleted index,
         * or the end of the point array if none
         *
         * @param index
         * @return the next non-deleted index, if any
         * @return inputLine.size() if there are no more non-deleted indices
        """
        next = index + 1
        length = len(self.coords)
        while next < length and self.isDeleted[next] == BufferInputLineSimplifier.DELETE:
            next += 1
        return next

    def collapseLine(self):
        coords = [co.clone() for i, co in enumerate(self.coords)
            if self.isDeleted[i] != BufferInputLineSimplifier.DELETE]
        return CoordinateSequence(coords)

    def isDeletable(self, i0: int, i1: int, i2: int, distanceTol: float) -> bool:
        p0 = self.coords[i0]
        p1 = self.coords[i1]
        p2 = self.coords[i2]

        if not self.isConcave(p0, p1, p2):
            return False

        if not self.isShallow(p0, p1, p2, distanceTol):
            return False

        return self.isShallowSampled(p0, p1, i0, i2, distanceTol)

    def isShallowConcavity(self, p0, p1, p2, distanceTol: float) -> bool:
        isConcave = self.isConcave(p0, p1, p2)
        if not isConcave:
            return False
        return self.isShallow(p0, p1, p2, distanceTol)

    def isShallowSampled(self, p0, p2, i0: int, i2: int, distanceTol: float)-> bool:
        """
         * Checks for shallowness over a sample of points in the given section.
         *
         * This helps prevents the siplification from incrementally
         * "skipping" over points which are in fact non-shallow.
         *
         * @param p0 start coordinate of section
         * @param p2 end coordinate of section
         * @param i0 start index of section
         * @param i2 end index of section
         * @param distanceTol distance tolerance
         * @return
        """
        # check every n'th point to see if it is within tolerance
        inc = i2 - i0 / BufferInputLineSimplifier.NUM_PTS_TO_CHECK
        if inc <= 0:
            inc = 1

        for i in range(i0, i2):
            if not self.isShallow(p0, p2, self.coords[i], distanceTol):
                return False

        return True

    def isShallow(self, p0, p1, p2, distanceTol: float) -> bool:
        dist = CGAlgorithms.distancePointLine(p1, p0, p2)
        return dist < distanceTol

    def isConcave(self, p0, p1, p2) -> bool:
        orientation = CGAlgorithms.computeOrientation(p0, p1, p2)
        return orientation == self.angleOrientation


class OffsetSegmentGenerator():
    """
     * Generates segments which form an offset curve.
     * Supports all end cap and join options
     * provided for buffering.
     * Implements various heuristics to
     * produce smoother, simpler curves which are
     * still within a reasonable tolerance of the
     * true curve.
     *
     * @author Martin Davis
    """

    """
     * Factor which controls how close offset segments can be to
     * skip adding a filler or mitre.
    """
    OFFSET_SEGMENT_SEPARATION_FACTOR = 1.0e-3

    """
     * Factor which controls how close curve vertices on inside turns
     * can be to be snapped
    """
    INSIDE_TURN_VERTEX_SNAP_DISTANCE_FACTOR = 1.0e-3

    """
     * Factor which controls how close curve vertices can be to be snapped
    """
    CURVE_VERTEX_SNAP_DISTANCE_FACTOR = 1.0e-6

    """
     * Factor which determines how short closing segs can be for round buffers
    """
    MAX_CLOSING_SEG_LEN_FACTOR = 80

    """
     * Use a value which results in a potential distance error which is
     * significantly less than the error due to
     * the quadrant segment discretization.
     * For QS = 8 a value of 100 is reasonable.
     * This should produce a maximum of 1% distance error.
    """
    SIMPLIFY_FACTOR = 100.0

    def __init__(self, precisionModel, bufParams, distance: float):
        """
         * @param nBufParams buffer parameters
        """

        """
         * the max error of approximation (distance) between a quad segment and
         * the true fillet curve
        """
        self.maxCurveSegmentError = 0.0
        """
         * The Closing Segment Factor controls how long "closing
         * segments" are.    Closing segments are added at the middle of
         * inside corners to ensure a smoother boundary for the buffer
         * offset curve.    In some cases (particularly for round joins
         * with default-or-better quantization) the closing segments
         * can be made quite short.    This substantially improves
         * performance (due to fewer intersections being created).
         * A closingSegFactor of 0 results in lines to the corner vertex.
         * A closingSegFactor of 1 results in lines halfway
         * to the corner vertex.
         * A closingSegFactor of 80 results in lines 1/81 of the way
         * to the corner vertex (this option is reasonable for the very
         * common default situation of round joins and quadrantSegs >= 8).
         * The default is 1.
        """
        self.closingSegLengthFactor = 1
        """
         * The angle quantum with which to approximate a fillet curve
         * (based on the input # of quadrant segments)
        """
        self.filletAngleQuantum = pi / 2.0 / bufParams.quadrantSegments
        """
         * Non-round joins cause issues with short closing segments,
         * so don't use them.  In any case, non-round joins
         * only really make sense for relatively small buffer distances.
        """
        if bufParams.quadrantSegments >= 8 and bufParams.joinStyle == JOIN_STYLE.round:
            self.closingSegLengthFactor = OffsetSegmentGenerator.MAX_CLOSING_SEG_LEN_FACTOR

        """
         * Owned by this object, destroyed by dtor
         * This actually gets created multiple times
         * and each of the old versions is pushed
         * to the ptLists std.vector to ensure all
         * created CoordinateSequences are properly
         * destroyed.
        """
        # OffsetSegmentString
        self.segList = OffsetSegmentString()

        self.distance = distance

        # geom.PrecisionModel
        self.precisionModel = precisionModel

        # BufferParameters
        self.bufParams = bufParams

        # algorithm.LineIntersector
        self.li = LineIntersector()

        # geom.Coordinate
        self.s0 = Coordinate()
        self.s1 = Coordinate()
        self.s2 = Coordinate()

        # geom.LineSegment
        self.seg0 = LineSegment()
        self.seg1 = LineSegment()
        self.offset0 = LineSegment()
        self.offset1 = LineSegment()

        self.side = 0

        """
         * Tests whether the input has a narrow concave angle
         * (relative to the offset distance).
         * In this case the generated offset curve will contain self-intersections
         * and heuristic closing segments.
         * This is expected behaviour in the case of buffer curves.
         * For pure offset curves,
         * the output needs to be further treated
         * before it can be used.
         *
         * @return true if the input has a narrow concave angle
        """
        self.hasNarrowConcaveAngle = False

        # Not in JTS, used for single-sided buffers
        self.endCapIndex = 0

        self.init(distance)

    def initSideSegments(self, s1, s2, side: int) -> None:
        self.s1 = s1
        self.s2 = s2
        self.side = side
        self.seg1.setCoordinates(s1, s2)
        self.computeOffsetSegment(self.seg1, side, self.distance, self.offset1)

    def getCoordinates(self, coordList: list) -> None:
        """
         * Get coordinates by taking ownership of them
        """
        # segList : OffsetSegmentString
        coordList.append(self.segList.coords)

    def closeRing(self) -> None:
        self.segList.closeRing()

    def createCircle(self, coord, distance: float) -> None:
        # Adds a CW circle around a point
        pt = Coordinate(coord.x + distance, coord.y)
        self.segList.addPt(pt)
        self._addFillet(coord, 0.0, 2.0 * pi, -1, distance)
        self.segList.closeRing()

    def createSquare(self, coord, distance: float) -> None:
        # Adds a CW square around a point
        self.segList.addPt(Coordinate(coord.x + distance, coord.y + distance))
        self.segList.addPt(Coordinate(coord.x + distance, coord.y - distance))
        self.segList.addPt(Coordinate(coord.x - distance, coord.y - distance))
        self.segList.addPt(Coordinate(coord.x - distance, coord.y + distance))
        self.segList.closeRing()

    def addFirstSegment(self) -> None:
        # Add first offset point
        self.segList.addPt(self.offset1.p0)

    def addLastSegment(self) -> None:
        # Add last offset point
        self.segList.addPt(self.offset1.p1)

    def addNextSegment(self, coord, addStartPoint: bool) -> None:
        # do nothing if points are equal
        if self.s2 == coord:
            return

        # s0-s1-s2 are the coordinates of the previous segment
        # and the current one

        self.s0 = self.s1
        self.s1 = self.s2
        self.s2 = coord

        self.seg0.setCoordinates(self.s0, self.s1)
        self.computeOffsetSegment(self.seg0, self.side, self.distance, self.offset0)

        self.seg1.setCoordinates(self.s1, self.s2)
        self.computeOffsetSegment(self.seg1, self.side, self.distance, self.offset1)

        orientation = CGAlgorithms.computeOrientation(self.s0, self.s1, self.s2)

        outsideTurn = ((orientation == CGAlgorithms.CLOCKWISE and self.side == Position.LEFT) or
            (orientation == CGAlgorithms.COUNTERCLOCKWISE and self.side == Position.RIGHT))

        self.index += 1
        logger.debug("segment: %s outsideTurn:%s orientation:%s", self.index, outsideTurn, orientation)

        if orientation == 0:
            self.addCollinear(addStartPoint)
        elif outsideTurn:
            self.addOutsideTurn(orientation, addStartPoint)
        else:
            self.addInsideTurn(orientation, addStartPoint)

    def addLineEndCap(self, p0, p1) -> None:
        """
         * Add an end cap around point p1, terminating a line segment
         * coming from p0
        """
        seg = LineSegment(p0, p1)
        offsetL = LineSegment()
        offsetR = LineSegment()
        self.computeOffsetSegment(seg, Position.LEFT, self.distance, offsetL)
        self.computeOffsetSegment(seg, Position.RIGHT, self.distance, offsetR)
        dx = p1.x - p0.x
        dy = p1.y - p0.y
        angle = atan2(dy, dx)

        if self.bufParams.endCapStyle == CAP_STYLE.round:
            # add offset seg points with a fillet between them
            self.segList.addPt(offsetL.p1)
            self._addFillet(p1, angle + pi / 2.0, angle - pi / 2.0, CGAlgorithms.CLOCKWISE, self.distance)
            self.segList.addPt(offsetR.p1)

        elif self.bufParams.endCapStyle == CAP_STYLE.flat:
            # only offset segment points are added
            self.segList.addPt(offsetL.p1)
            self.segList.addPt(offsetR.p1)

        elif self.bufParams.endCapStyle == CAP_STYLE.square:
            # add a square defined by extensions of the offset
            # segment endpoints
            x = abs(self.distance) * cos(angle)
            y = abs(self.distance) * sin(angle)

            squareCapLOffset = Coordinate(
                offsetL.p1.x + x,
                offsetL.p1.y + y
                )
            squareCapROffset = Coordinate(
                offsetR.p1.x + x,
                offsetR.p1.y + y
                )

            self.segList.addPt(squareCapLOffset)
            self.segList.addPt(squareCapROffset)

    def addSegments(self, coords, isForward: bool) -> None:
        self.segList.addPts(coords, isForward)

    def addCollinear(self, addStartPoint: bool) -> None:
        """
         * This test could probably be done more efficiently,
         * but the situation of exact collinearity should be fairly rare.
        """
        self.li.computeLinesIntersection(self.s0, self.s1, self.s1, self.s2)
        numInt = self.li.intersections
        """
         * if numInt is<2, the lines are parallel and in the same direction.
         * In this case the point can be ignored, since the offset lines
         * will also be parallel.
        """
        if numInt > 1:
            """
             * Segments are collinear but reversing.
             * Add an "end-cap" fillet
             * all the way around to other direction
             *
             * This case should ONLY happen for LineStrings,
             * so the orientation is always CW (Polygons can never
             * have two consecutive segments which are parallel but
             * reversed, because that would be a self intersection).
            """
            if (self.bufParams.joinStyle == JOIN_STYLE.bevel or
                    self.bufParams.joinStyle == JOIN_STYLE.mitre):

                if addStartPoint:
                    self.segList.addPt(self.offset0.p1)

                self.segList.addPt(self.offset1.p0)
            else:

                self.addFillet(self.s1,
                    self.offset0.p1,
                    self.offset1.p0,
                    CGAlgorithms.CLOCKWISE,
                    self.distance)

    def addMitreJoin(self, coord, offset0, offset1, distance: float) -> None:
        """
         * The mitre will be beveled if it exceeds the mitre ratio limit.
         *
         * @param offset0 the first offset segment
         * @param offset1 the second offset segment
         * @param distance the offset distance
        """
        isMitreWithinLimit = True
        intPt = Coordinate()

        """
         * This computation is unstable if the offset segments
         * are nearly collinear.
         * Howver, this situation should have been eliminated earlier
         * by the check for whether the offset segment endpoints are
         * almost coincident
        """
        try:
            HCoordinate.intersection(offset0.p0, offset0.p1,
                offset1.p0, offset1.p1,
                intPt)

            if distance <= 0.0:
                mitreRatio = 1.0
            else:
                mitreRatio = intPt.distance(coord) / abs(distance)

            if mitreRatio > self.bufParams.mitreLimit:
                isMitreWithinLimit = False

        except:
            isMitreWithinLimit = False
            pass

        if isMitreWithinLimit:
            self.segList.addPt(intPt)
        else:
            self.addLimitedMitreJoin(offset0, offset1, distance, self.bufParams.mitreLimit)

    def addLimitedMitreJoin(self, offset0, offset1, distance: float, mitreLimit: float) -> None:
        """
         * Adds a limited mitre join connecting the two reflex offset segments.
         *
         * A limited mitre is a mitre which is beveled at the distance
         * determined by the mitre ratio limit.
         *
         * @param offset0 the first offset segment
         * @param offset1 the second offset segment
         * @param distance the offset distance
         * @param mitreLimit the mitre limit ratio
        """
        basePt = self.seg0.p1
        ang0 = Angle.angle(basePt, self.seg0.p0)
        # oriented angle between segments
        angDiff = Angle.angleBetweenOriented(self.seg0.p0, basePt, self.seg1.p1)
        # half of the interiors angle
        angDiffHalf = angDiff / 2.0

        # angle for bisector of the interiors angle between the segments
        midAng = Angle.normalize(ang0 + angDiffHalf)
        # rotating this by PI gives the bisector of the reflex angle
        mitreMidAng = Angle.normalize(midAng + pi)

        # the miterLimit determines the distance to the mitre bevel
        mitreDist = mitreLimit * distance
        # the bevel delta is the difference between the buffer distance
        # and half of the length of the bevel segment
        bevelDelta = mitreDist * abs(sin(angDiffHalf))
        bevelHalfLen = distance - bevelDelta

        # compute the midpoint of the bevel segment
        bevelMidX = basePt.x + mitreDist * cos(mitreMidAng)
        bevelMidY = basePt.y + mitreDist * sin(mitreMidAng)
        bevelMidPt = Coordinate(bevelMidX, bevelMidY)

        # compute the mitre midline segment from the corner point to
        # the bevel segment midpoint
        mitreMidLine = LineSegment(basePt, bevelMidPt)

        # finally the bevel segment endpoints are computed as offsets from
        # the mitre midline
        bevelEndLeft = Coordinate()
        mitreMidLine.pointAlongOffset(1.0, bevelHalfLen, bevelEndLeft)
        bevelEndRight = Coordinate()
        mitreMidLine.pointAlongOffset(1.0, -bevelHalfLen, bevelEndRight)

        if self.side == Position.LEFT:
            self.segList.addPt(bevelEndLeft)
            self.segList.addPt(bevelEndRight)
        else:
            self.segList.addPt(bevelEndRight)
            self.segList.addPt(bevelEndLeft)

    def addBevelJoin(self, offset0, offset1)-> None:
        """
         * Adds a bevel join connecting the two offset segments
         * around a reflex corner.
         *
         * @param offset0 the first offset segment
         * @param offset1 the second offset segment
        """
        self.segList.addPt(offset0.p1)
        self.segList.addPt(offset1.p0)

    def init(self, distance: float) -> None:
        self.distance = distance
        self.maxCurveSegmentError = distance * (1 - cos(self.filletAngleQuantum / 2.0))
        self.segList.reset()
        self.segList.precisionModel = self.precisionModel
        # debug
        self.index = 0

        """
         * Choose the min vertex separation as a small fraction of
         * the offset distance.
        """
        self.segList.minimumVertexDistance = \
            distance * OffsetSegmentGenerator.CURVE_VERTEX_SNAP_DISTANCE_FACTOR

    def addOutsideTurn(self, orientation: int, addStartPoint: bool)-> None:
        """
         * Adds the offset points for an outside (convex) turn
         * @param orientation
         * @param addStartPoint
         *
         * Heuristic: If offset endpoints are very close together,
         * just use one of them as the corner vertex.
         * This avoids problems with computing mitre corners in the case
         * where the two segments are almost parallel
         * (which is hard to compute a robust intersection for).
        """
        if (self.offset0.p1.distance(self.offset1.p0) <
                self.distance * OffsetSegmentGenerator.OFFSET_SEGMENT_SEPARATION_FACTOR):
            self.segList.addPt(self.offset0.p1)
            return

        if self.bufParams.joinStyle == JOIN_STYLE.mitre:
            self.addMitreJoin(self.s1, self.offset0, self.offset1, self.distance)

        elif self.bufParams.joinStyle == JOIN_STYLE.bevel:
            self.addBevelJoin(self.offset0, self.offset1)

        else:
            # add a circular fillet connecting the endpoints
            # of the offset segments
            if addStartPoint:
                self.segList.addPt(self.offset0.p1)
            # TESTING - comment out to produce beveled joins
            self.addFillet(self.s1, self.offset0.p1, self.offset1.p0, orientation, self.distance)
            self.segList.addPt(self.offset1.p0)

    def addInsideTurn(self, orientation: int, addStartPoint: bool)-> None:
        """
         * Adds the offset points for an inside (concave) turn
         * @param orientation
         * @param addStartPoint
        """
        # add intersection point of offset segments (if any)
        self.li.computeLinesIntersection(self.offset0.p0, self.offset0.p1, self.offset1.p0, self.offset1.p1)
        if self.li.hasIntersection:
            logger.debug("has Intersection")
            self.segList.addPt(self.li.intersectionPts[0])
            return
        logger.debug("no intersection found %s %s", self.offset0, self.offset1)

        """
         * If no intersection is detected, it means the angle is so small
         * and/or the offset so large that the offsets segments don't
         * intersect. In this case we must add a "closing segment" to make
         * sure the buffer curve is continuous,
         * fairly smooth (e.g. no sharp reversals in direction)
         * and tracks the buffer correctly around the corner.
         * The curve connects the endpoints of the segment offsets to points
         * which lie toward the centre point of the corner.
         * The joining curve will not appear in the final buffer outline,
         * since it is completely internal to the buffer polygon.

         * In complex buffer cases the closing segment may cut across many
         * other segments in the generated offset curve.
         * In order to improve the performance of the noding, the closing
         * segment should be kept as short as possible.
         * (But not too short, since that would defeat it's purpose).
         * This is the purpose of the closingSegLengthFactor heuristic value.

         * The intersection test above is vulnerable to robustness errors
         * i.e. it may be that the offsets should intersect very close to
         * their endpoints, but aren't reported as such due to rounding.
         * To handle this situation appropriately, we use the following test:
         * If the offset points are very close, don't add closing segments
         * but simply use one of the offset points
        """
        self.hasNarrowConcaveAngle = True
        if (self.offset0.p1.distance(self.offset1.p0) <
                self.distance * OffsetSegmentGenerator.INSIDE_TURN_VERTEX_SNAP_DISTANCE_FACTOR):
            self.segList.addPt(self.offset0.p1)
        else:
            # add endpoint of this segment offset
            self.segList.addPt(self.offset0.p1)
            # Add "closing segment" of required length.

            if self.closingSegLengthFactor > 0:

                mid0 = Coordinate(
                    (self.closingSegLengthFactor * self.offset0.p1.x + self.s1.x) / (self.closingSegLengthFactor + 1),
                    (self.closingSegLengthFactor * self.offset0.p1.y + self.s1.y) / (self.closingSegLengthFactor + 1)
                )
                self.segList.addPt(mid0)

                mid1 = Coordinate(
                    (self.closingSegLengthFactor * self.offset1.p0.x + self.s1.x) / (self.closingSegLengthFactor + 1),
                    (self.closingSegLengthFactor * self.offset1.p0.y + self.s1.y) / (self.closingSegLengthFactor + 1)
                )
                self.segList.addPt(mid1)

            else:

                # This branch is not expected to be used
                self.segList.addPt(self.s1)

            # add start point of next segment offset
            self.segList.addPt(self.offset1.p0)

    def computeOffsetSegment(self, seg, side: int, distance: float, offset) -> None:
        """
         * Compute an offset segment for an input segment on a given
         * side and at a given distance.
         *
         * The offset points are computed in full double precision,
         * for accuracy.
         *
         * @param seg the segment to offset
         * @param side the side of the segment the offset lies on
         * @param distance the offset distance
         * @param offset the points computed for the offset segment
        """
        if side == Position.LEFT:
            sideSign = 1
        else:
            sideSign = -1

        dx = seg.p1.x - seg.p0.x
        dy = seg.p1.y - seg.p0.y

        length = sqrt(dx * dx + dy * dy)

        # u is the vector that is the length of the offset
        # in the direction of the segment
        ux = sideSign * distance * dx / length
        uy = sideSign * distance * dy / length

        offset.p0.x = seg.p0.x - uy
        offset.p0.y = seg.p0.y + ux
        offset.p1.x = seg.p1.x - uy
        offset.p1.y = seg.p1.y + ux

    def addFillet(self, coord, p0, p1, direction: int, radius: float) -> None:
        """
         * Adds points for a circular fillet around a reflex corner.
         *
         * Adds the start and end points
         *
         * @param p base point of curve
         * @param p0 start point of fillet curve
         * @param p1 endpoint of fillet curve
         * @param direction the orientation of the fillet
         * @param radius the radius of the fillet
        """
        dx0 = p0.x - coord.x
        dy0 = p0.y - coord.y
        startAngle = atan2(dy0, dx0)

        dx1 = p1.x - coord.x
        dy1 = p1.y - coord.y
        endAngle = atan2(dy1, dx1)

        if direction == CGAlgorithms.CLOCKWISE:
            if startAngle <= endAngle:
                startAngle += 2.0 * pi

        elif startAngle >= endAngle:
            # direction==COUNTERCLOCKWISE
            startAngle -= 2.0 * pi

        self.segList.addPt(p0)
        self._addFillet(coord, startAngle, endAngle, direction, radius)
        self.segList.addPt(p1)

    def _addFillet(self, coord, startAngle: float, endAngle: float, direction: int, radius: float)-> None:
        """
         * Adds points for a circular fillet arc between two specified angles.
         *
         * The start and end point for the fillet are not added -
         * the caller must add them if required.
         *
         * @param direction is -1 for a CW angle, 1 for a CCW angle
         * @param radius the radius of the fillet
        """

        if direction == CGAlgorithms.CLOCKWISE:
            directionFactor = -1
        else:
            directionFactor = 1

        totalAngle = abs(startAngle - endAngle)
        nSegs = int(totalAngle / self.filletAngleQuantum + 0.5)

        # no segments because angle is less than increment-nothing to do!
        if nSegs < 1:
            return

        # choose angle increment so that each segment has equal length
        currAngleInc = totalAngle / nSegs

        currAngle = 0.0
        pt = Coordinate()
        while currAngle < totalAngle:
            angle = startAngle + directionFactor * currAngle
            pt.x = coord.x + (radius * cos(angle))
            pt.y = coord.y + (radius * sin(angle))
            self.segList.addPt(pt)
            currAngle += currAngleInc


class OffsetCurveBuilder():
    """
     * Computes the raw offset curve for a
     * single Geometry component (ring, line or point).
     *
     * A raw offset curve line is not noded -
     * it may contain self-intersections (and usually will).
     * The final buffer polygon is computed by forming a topological graph
     * of all the noded raw curves and tracing outside contours.
     * The points in the raw curve are rounded to a given geom.PrecisionModel.
    """

    """
     * Use a value which results in a potential distance error which is
     * significantly less than the error due to
     * the quadrant segment discretization.
     * For QS = 8 a value of 100 is reasonable.
     * This should produce a maximum of 1% distance error.
    """
    SIMPLIFY_FACTOR = 100.0

    def __init__(self, precisionModel, bufParams):
        """
         * @param nBufParams buffer parameters
        """
        self.distance = 0.0
        # PrecisionModel
        self.precisionModel = precisionModel
        # BufferParameters
        self.bufParams = bufParams

    def getLineCurve(self, coords, distance: float, lineList: list) -> None:
        """
         * This method handles single points as well as lines.
         * Lines are assumed to <b>not</b> be closed (the function will not
         * fail for closed lines, but will generate superfluous line caps).
         *
         * @param lineList the std.vector to which the newly created
         *                 CoordinateSequences will be pushed_back.
         *                 Caller is responsible to delete these new elements.
        """
        self.distance = distance

        # a zero or (non-singlesided) negative width buffer of a line/point is empty
        if distance == 0.0:
            return

        if distance < 0.0 and not self.bufParams.isSingleSided:
            return

        posDistance = abs(distance)

        # OffsetSegmentGenerator
        segGen = self.getSegGen(posDistance)

        if len(coords) < 2:
            self.computePointCurve(coords[0], segGen)
        else:
            if self.bufParams.isSingleSided:
                isRightSide = distance < 0.0
                self.computeSingleSidedBufferCurve(coords, isRightSide, segGen)
            else:
                self.computeLineBufferCurve(coords, segGen)

        segGen.getCoordinates(lineList)

    def getSingleSidedLineCurve(self, coords, distance: float, lineList: list, leftSide: bool, rightSide: bool) -> None:
        """
         * This method handles single points as well as lines.
         *
         * Lines are assumed to <b>not</b> be closed (the function will not
         * fail for closed lines, but will generate superfluous line caps).
         *
         * @param lineList the std.vector to which newly created
         *                 CoordinateSequences will be pushed_back.
         *                 Caller will be responsible to delete them.
         * @param leftSide indicates that the left side buffer will be
         *                 obtained/skipped
         * @param rightSide indicates that the right side buffer will
         *                  be obtained/skipped
         *
         * NOTE: this is a GEOS extension
        """
        # A zero or negative width buffer of a line/point is empty.
        if distance <= 0.0:
            return

        if len(coords) < 2:
            # No cap
            return

        distTol = self.simplifyTolerance(distance)
        segGen = self.getSegGen(distance)

        if leftSide:
            # --------- compute points for left side of line
            # Simplify the appropriate side of the line before generating
            simp1 = BufferInputLineSimplifier.simplify(coords, distTol)
            n1 = len(simp1)
            if n1 < 2:
                raise ValueError("Cannot get offset of single-vertex line")

            segGen.initSideSegments(simp1[0], simp1[1], Position.LEFT)
            segGen.addFirstSegment()

            for i in range(2, n1):
                segGen.addNextSegment(simp1[i], True)

            segGen.addLastSegment()

        if rightSide:
            # --------- compute points for right side of line
            # Simplify the appropriate side of the line before generating
            simp2 = BufferInputLineSimplifier.simplify(coords, -distTol)
            n2 = len(simp2)
            if n2 < 2:
                raise ValueError("Cannot get offset of single-vertex line")

            segGen.initSideSegments(simp2[-1], simp2[-2], Position.LEFT)
            segGen.addFirstSegment()

            for i in range(n2 - 3, -1, -1):
                segGen.addNextSegment(simp2[i], True)

            segGen.addLastSegment()

        segGen.getCoordinates(lineList)

    def getRingCurve(self, coords, side: int, distance: float, lineList: list) -> None:
        """
         * This method handles the degenerate cases of single points and lines,
         * as well as rings.
         *
         * @param lineList the std.vector to which CoordinateSequences will
         *                 be pushed_back
        """
        self.distance = distance

        if distance == 0.0:
            lineList.append(coords.clone())
            return

        if len(coords) < 3:
            self.getLineCurve(coords, distance, lineList)
            return

        segGen = self.getSegGen(abs(distance))
        self.computeRingBufferCurve(coords, side, segGen)
        segGen.getCoordinates(lineList)

    def simplifyTolerance(self, bufDistance: float) -> float:
        """
         * Computes the distance tolerance to use during input
         * line simplification.
         *
         * @param distance the buffer distance
         * @return the simplification tolerance
        """
        return bufDistance / OffsetCurveBuilder.SIMPLIFY_FACTOR

    def computeLineBufferCurve(self, coords, segGen) -> None:
        distTol = self.simplifyTolerance(self.distance)

        # --------- compute points for left side of line
        # Simplify the appropriate side of the line before generating
        # CoordinateSequence
        simp1 = BufferInputLineSimplifier.simplify(coords, distTol)

        n1 = len(simp1)
        segGen.initSideSegments(simp1[0], simp1[1], Position.LEFT)

        for i in range(2, n1):
            segGen.addNextSegment(simp1[i], True)

        segGen.addLastSegment()

        # add line cap for end of line
        segGen.addLineEndCap(simp1[-2], simp1[-1])

        # --------- compute points for right side of line
        # Simplify the appropriate side of the line before generating
        # CoordinateSequence
        simp2 = BufferInputLineSimplifier.simplify(coords, -distTol)

        n2 = len(simp2)
        segGen.initSideSegments(simp2[-1], simp2[-2], Position.LEFT)

        for i in range(n2 - 3, -1, -1):
            segGen.addNextSegment(simp2[i], True)

        segGen.addLastSegment()

        # add line cap for end of line
        segGen.addLineEndCap(simp2[1], simp2[0])

        segGen.closeRing()

    def computeSingleSidedBufferCurve(self, coords, isRightSide: bool, segGen) -> None:
        distTol = self.simplifyTolerance(self.distance)

        if isRightSide:
            # add original line
            segGen.addSegments(coords, True)

            # ---------- compute points for right side of line
            # Simplify the appropriate side of the line before generating
            simp2 = BufferInputLineSimplifier.simplify(coords, -distTol)

            n2 = len(simp2)
            segGen.initSideSegments(simp2[-1], simp2[-2], Position.LEFT)
            segGen.addFirstSegment()

            for i in range(n2 - 3, -1, -1):
                segGen.addNextSegment(simp2[i], True)

        else:
            # add original line
            segGen.addSegments(coords, False)

            # ---------- compute points for left side of line
            # Simplify the appropriate side of the line before generating
            simp1 = BufferInputLineSimplifier.simplify(coords, distTol)

            n1 = len(simp1)
            segGen.initSideSegments(simp1[0], simp1[1], Position.LEFT)
            segGen.addFirstSegment()

            for i in range(2, n1):
                segGen.addNextSegment(simp1[i], True)

        segGen.addLastSegment()
        segGen.closeRing()

    def computeRingBufferCurve(self, coords, side: int, segGen) -> None:
        distTol = self.simplifyTolerance(self.distance)

        if side == Position.RIGHT:
            distTol = -distTol

        simp = BufferInputLineSimplifier.simplify(coords, distTol)

        n = len(simp)
        segGen.initSideSegments(simp[-2], simp[0], side)

        for i in range(1, n):
            addStartPoint = i != 1
            segGen.addNextSegment(simp[i], addStartPoint)

        segGen.closeRing()

    def getSegGen(self, dist: float):
        # OffsetSegmentGenerator
        return OffsetSegmentGenerator(self.precisionModel, self.bufParams, dist)

    def computePointCurve(self, coord, segGen) -> None:

        if self.bufParams.endCapStyle == CAP_STYLE.round:
            segGen.createCircle(coord, self.distance)

        elif self.bufParams.endCapStyle == CAP_STYLE.square:
            segGen.createSquare(coord, self.distance)


class OffsetCurveSetBuilder():
    """
     * Creates all the raw offset curves for a buffer of a Geometry.
     *
     * Raw curves need to be noded together and polygonized to form the
     * final buffer area.
    """
    def __init__(self, geom, distance: float, curveBuilder):

        self.geom = geom

        self.distance = distance

        # OffsetCurveBuilder
        self.curveBuilder = curveBuilder

        # The raw offset curves computed.
        # noding.SegmentString
        self.curveList = []

    def addCurve(self, coords, leftLoc: int, rightLoc: int) -> None:
        """
         * Creates a noding.SegmentString for a coordinate list which is a raw
         * offset curve, and adds it to the list of buffer curves.
         * The noding.SegmentString is tagged with a geomgraph.Label
         * giving the topology of the curve.
         * The curve may be oriented in either direction.
         * If the curve is oriented CW, the locations will be:
         * - Left: Location.EXTERIOR
         * - Right: Location.INTERIOR
         *
         * @param coord is raw offset curve, ownership transferred here
        """
        if len(coords) < 2:
            return

        # add the edge for a coordinate list which is a raw offset curve
        label = Label(0, Location.BOUNDARY, leftLoc, rightLoc)

        ss = NodedSegmentString(coords, label)

        self.curveList.append(ss)

    def add(self, geom) -> None:

        if geom.is_empty:
            return

        if geom.type_id == GeomTypeId.GEOS_POINT:
            self.addPoint(geom)

        elif geom.type_id in (
                GeomTypeId.GEOS_LINESTRING,
                GeomTypeId.GEOS_LINEARRING):
            self.addLineString(geom)

        elif geom.type_id == GeomTypeId.GEOS_POLYGON:
            self.addPolygon(geom)

        elif geom.type_id in (
                GeomTypeId.GEOS_MULTIPOINT,
                GeomTypeId.GEOS_MULTILINESTRING,
                GeomTypeId.GEOS_MULTIPOLYGON,
                GeomTypeId.GEOS_GEOMETRYCOLLECTION):
            self.addCollection(geom)

        else:
            raise ValueError("GeometryGraph.add(Geometry): unknown geometry type: {}".format(type(geom).__name))

    def addCollection(self, geoms) -> None:
        for geom in geoms.geoms:
            self.add(geom)

    def addPoint(self, p) -> None:
        """
         * Add a Point to the graph.
        """
        if self.distance <= 0.0:
            return
        # CoordinateSequence
        coords = p.coords
        # CoordinateSequence
        lineList = []
        self.curveBuilder.getLineCurve(coords, self.distance, lineList)
        self.addCurves(lineList, Location.EXTERIOR, Location.INTERIOR)

    def addLineString(self, line) -> None:
        if self.distance <= 0.0 and not self.curveBuilder.bufParams.isSingleSided:
            return
        coords = CoordinateSequence.removeRepeatedPoints(line.coords)
        lineList = []
        self.curveBuilder.getLineCurve(coords, self.distance, lineList)
        self.addCurves(lineList, Location.EXTERIOR, Location.INTERIOR)

    def addPolygon(self, p) -> None:
        offsetDistance = self.distance
        offsetSide = Position.LEFT

        if self.distance < 0.0:
            offsetDistance = -self.distance
            offsetSide = Position.RIGHT

        exterior = p.exterior

        # optimization - don't bother computing buffer
        # if the polygon would be completely eroded
        if self.distance < 0.0 and self.isErodedCompletely(exterior, self.distance):
            return

        # don't attempt to buffer a polygon
        # with too few distinct vertices
        exteriorCoord = CoordinateSequence.removeRepeatedPoints(exterior.coords)
        if self.distance < 0.0 and len(exteriorCoord) < 3:
            return

        self.addPolygonRing(exteriorCoord,
            offsetDistance,
            offsetSide,
            Location.EXTERIOR,
            Location.INTERIOR)

        for hole in p.interiors:

            if self.distance > 0.0 and self.isErodedCompletely(hole, -self.distance):
                continue

            holeCoord = CoordinateSequence.removeRepeatedPoints(hole.coords)

            self.addPolygonRing(holeCoord,
                offsetDistance,
                Position.opposite(offsetSide),
                Location.INTERIOR,
                Location.EXTERIOR)

    def addPolygonRing(self, coords, offsetDistance: float, side: int, cwLeftLoc: int, cwRightLoc: int) -> None:
        """
         * Add an offset curve for a polygon ring.
         * The side and left and right topological location arguments
         * assume that the ring is oriented CW.
         * If the ring is in the opposite orientation,
         * the left and right locations must be interchanged and the side
         * flipped.
         *
         * @param coord the coordinates of the ring (must not contain
         * repeated points)
         * @param offsetDistance the distance at which to create the buffer
         * @param side the side of the ring on which to construct the buffer
         *             line
         * @param cwLeftLoc the location on the L side of the ring
         *                  (if it is CW)
         * @param cwRightLoc the location on the R side of the ring
         *                   (if it is CW)
        """
        if offsetDistance == 0.0 and len(coords) < 4:
            return

        leftLoc = cwLeftLoc
        rightLoc = cwRightLoc

        if len(coords) > 3 and CGAlgorithms.isCCW(coords):
            leftLoc, rightLoc = rightLoc, leftLoc
            side = Position.opposite(side)

        lineList = []
        self.curveBuilder.getRingCurve(coords, side, offsetDistance, lineList)
        self.addCurves(lineList, leftLoc, rightLoc)

    def isErodedCompletely(self, ring, bufferDistance: float) -> bool:
        """
         * The ringCoord is assumed to contain no repeated points.
         * It may be degenerate (i.e. contain only 1, 2, or 3 points).
         * In this case it has no area, and hence has a minimum diameter of 0.
         *
         * @param ring
         * @param offsetDistance
         * @return
        """
        coords = ring.coords

        # degenerate ring has no area
        if len(coords) < 4:
            return bufferDistance < 0

        # important test to eliminate inverted triangle bug
        # also optimizes erosion test for triangles
        if len(coords) == 4:
            return self.isTriangleErodedCompletely(coords, bufferDistance)

        env = ring.envelope
        envMinDimension = min(env.height, env.width)
        if bufferDistance < 0.0 and 2 * abs(bufferDistance) > envMinDimension:
            return True
        """
         * The following is a heuristic test to determine whether an
         * inside buffer will be eroded completely->
         * It is based on the fact that the minimum diameter of the ring
         * pointset
         * provides an upper bound on the buffer distance which would erode the
         * ring->
         * If the buffer distance is less than the minimum diameter, the ring
         * may still be eroded, but this will be determined by
         * a full topological computation->
        """
        # There's an unknown bug so disable this for now
        """
        md = MinimumDiameter(ring)
        minDiam = md.length
        return minDiam < 2 * abs(bufferDistance)
        """
        return False

    def isTriangleErodedCompletely(self, coords, bufferDistance: float) -> bool:
        """
         * Tests whether a triangular ring would be eroded completely by
         * the given buffer distance.
         * This is a precise test.  It uses the fact that the inner buffer
         * of a triangle converges on the inCentre of the triangle (the
         * point equidistant from all sides).  If the buffer distance is
         * greater than the distance of the inCentre from a side, the
         * triangle will be eroded completely.
         *
         * This test is important, since it removes a problematic case where
         * the buffer distance is slightly larger than the inCentre distance.
         * In this case the triangle buffer curve "inverts" with incorrect
         * topology, producing an incorrect hole in the buffer.
         *
         * @param triCoord
         * @param bufferDistance
         * @return
        """
        tri = Triangle(coords[0], coords[1], coords[2])
        inCentre = Coordinate()
        tri.inCentre(inCentre)
        distToCentre = CGAlgorithms.distancePointLine(inCentre, tri.p0, tri.p1)
        return distToCentre < abs(bufferDistance)

    def getCurves(self) -> list:
        """
         * Computes the set of raw offset curves for the buffer.
         *
         * Each offset curve has an attached {@link geomgraph.Label} indicating
         * its left and right location.
         *
         * @return a Collection of SegmentStrings representing the raw
         * buffer curves
        """
        self.add(self.geom)
        return self.curveList

    def addCurves(self, lineList: list, leftLoc: int, rightLoc: int) -> None:
        """
         * Add raw curves for a set of CoordinateSequences
         *
         * @param lineList is a list of CoordinateSequence, ownership
         *       of which is transferred here.
        """
        # logger.debug("OffsetCurveSetBuilder.addCurves(%s)", len(lineList))
        for coords in lineList:
            self.addCurve(coords, leftLoc, rightLoc)


class BufferBuilder():
    """
     * Builds the buffer geometry for a given input geometry and precision model.
     *
     * Allows setting the level of approximation for circular arcs,
     * and the precision model in which to carry out the computation.
     *
     * When computing buffers in floating point double-precision
     * it can happen that the process of iterated noding can fail to converge
     * (terminate).
     *
     * In this case a TopologyException will be thrown.
     * Retrying the computation in a fixed precision
     * can produce more robust results.
    """
    def __init__(self, bufParams):
        """
         * Creates a new BufferBuilder
         *
         * @param nBufParams buffer parameters, this object will
         *                   keep a reference to the passed parameters
         *                   so caller must make sure the object is
         *                   kept alive for the whole lifetime of
         *                   the buffer builder.
        """
        # BufferParameters
        self.bufParams = bufParams
        # PrecisionModel
        self.workingPrecisionModel = None
        # LineIntersector
        self.li = None
        # IntersectionNodeAdder
        self.si = None
        # Noder
        self.workingNoder = None
        # GeometryFactory
        self._factory = None
        # geomgraph.EdgeList
        self.edgeList = EdgeList()
        # geomgraph.Label
        self.newLabels = []
        
    def buffer(self, geom, distance: float):
        precisionModel = self.workingPrecisionModel
        if precisionModel is None:
            precisionModel = geom.precisionModel

        # factory must be the same as the one used by the input
        self._factory = geom._factory

        curveBuilder = OffsetCurveBuilder(precisionModel, self.bufParams)
        curveSetBuilder = OffsetCurveSetBuilder(geom, distance, curveBuilder)
        # SegmentString
        bufferSegStrList = curveSetBuilder.getCurves()


        # blines = self._factory.buildGeometry([self._factory.createLineString(ns.coords) for ns in bufferSegStrList])
        # self._factory.output(blines, name="bufferSegStrList")

        logger.debug("OffsetCurveSetBuilder got %s curves", len(bufferSegStrList))

        # short circuit tester
        if len(bufferSegStrList) <= 0:
            return self.createEmptyResultGeometry()

        self.computeNodeEdges(bufferSegStrList, precisionModel)

        # Geometry
        resultGeom = None
        # Geomety
        resultPolyList = []
        # BufferSubGraph
        subGraphList = []

        logger.debug("BufferBuilder.edgeList %s", self.edgeList)

        graph = PlanarGraph(OverlayNodeFactory())

        graph.addEdges(self.edgeList)

        self.createSubGraphs(graph, subGraphList)
        logger.debug("BufferBuilder.subGraphList %s", len(subGraphList))

        polyBuilder = PolygonBuilder(self._factory)
        self.buildSubGraphs(subGraphList, polyBuilder)
        resultPolyList.extend(polyBuilder.getPolygons())

        if len(resultPolyList) == 0:
            return self.createEmptyResultGeometry()

        resultGeom = self._factory.buildGeometry(resultPolyList)

        return resultGeom

    def bufferLineSingleSided(self, geom, distance: float, leftSide: bool):
        """
         * Generates offset curve for linear geometry.
         *
         * @param g non-areal geometry object
         * @param distance width of offset
         * @param leftSide controls on which side of the input geometry
         *        offset curve is generated.
         *
         * @note For left-side offset curve, the offset will be at the left side
         *       of the input line and retain the same direction.
         *       For right-side offset curve, it'll be at the right side
         *       and in the opposite direction.
         *
         * @note BufferParameters.setSingleSided parameter, which is specific to
         *       areal geometries only, is ignored by this routine.
        """

        # Returns the line used to create a single-sided buffer.
        # Input requirement: Must be a LineString.
        if geom.type_id != GeomTypeId.GEOS_LINESTRING:
            raise ValueError("BufferBuilder.bufferLineSingleSided only accept linestrings")

        if distance == 0:
            # Nothing to do for a distance of zero
            return geom.clone()

        precisionModel = self.workingPrecisionModel
        if precisionModel is None:
            precisionModel = geom.precisionModel

        self._factory = geom._factory

        # BufferParameters
        modParams = BufferParameters(self.bufParams)
        modParams.endCapStyle = CAP_STYLE.flat

        # ignore parameter for areal-only geometries
        modParams.isSingleSided = False
        # """
        tmp = BufferBuilder(modParams)
        buf = tmp.buffer(geom, distance)
        # self._factory.output(buf, name="buffer")

        # Create MultiLineStrings from this polygon.
        bufLineString = buf.boundary
        # self._factory.output(bufLineString, name="boundary")

        # Then, get the raw (i.e. unnoded) single sided offset curve.
        curveBuilder = OffsetCurveBuilder(precisionModel, modParams)

        # CoordinateSequence
        lineList = []
        coords = geom.coords
        curveBuilder.getSingleSidedLineCurve(coords, distance, lineList, leftSide, not leftSide)

        # SegmentString
        curveList = [NodedSegmentString(line, None) for line in lineList]
        # lineList.clear()

        # Noder
        noder = self.getNoder(precisionModel)
        noder.computeNodes(curveList)

        # SegmentString
        nodedEdges = noder.getNodedSubStrings()

        # Geometry
        singleSidedNodedEdges = [self._factory.createLineString(ss.coords.clone()) for ss in nodedEdges]
        singleSided = self._factory.createMultiLineString(singleSidedNodedEdges)
        # self._factory.output(singleSided, name="singleSided")

        # Use the boolean operation intersect to obtain the line segments lying
        # on both the butt-cap buffer and this multi-line.
        # Geometry* intersectedLines = singleSided->intersection( bufLineString )
        # NOTE: we use Snapped overlay because the actual buffer boundary might
        #       diverge from original offset curves due to the addition of
        #       intersections with caps and joins curves
        intersectedLines = SnapOverlayOp.intersection(singleSided, bufLineString)
        # self._factory.output(intersectedLines, name="intersectedLines")

        lineMerge = LineMerger()
        lineMerge.add(intersectedLines)

        # LineString
        mergedLines = lineMerge.getMergedLineStrings()
        merged = self._factory.createMultiLineString(mergedLines)
        # self._factory.output(merged, name="merged")

        # Geometry
        mergedLinesGeom = []
        startPoint = geom.coords[0]
        endPoint = geom.coords[-1]
        while (len(mergedLines) > 0):
            # Remove end points if they are a part of the original line to be buffered
            # CoordinateSequence
            coords = mergedLines[-1].coords
            if coords is not None:
                # Use 98% of the buffer width as the point-distance requirement - this
                # is to ensure that the point that is "distance" +/- epsilon is not
                # included.
                #
                # Let's try and estimate a more accurate bound instead of just assuming
                # 98%. With 98%, the episilon grows as the buffer distance grows,
                # so that at large distance, artifacts may skip through this filter
                # Let the length of the line play a factor in the distance, which is still
                # going to be bounded by 98%. Take 10% of the length of the line  from the buffer distance
                # to try and minimize any artifacts.

                ptDistAllowance = max(distance - geom.length * 0.1, distance * 0.98)
                # Use 102% of the buffer width as the line-length requirement - this
                # is to ensure that line segments that is length "distance" +/-
                # epsilon is removed.

                segLengthAllowance = 1.02 * distance

                # Clean up the front of the list.
                # Loop until the line's end is not inside the buffer width from
                # the startPoint.

                while len(coords) > 1 and coords[0].distance(startPoint) < ptDistAllowance:
                    # Record the end segment length.
                    segLength = coords[0].distance(coords[1])
                    # Stop looping if there are no more points, or if the segment
                    # length is larger than the buffer width.
                    if len(coords) <= 1 or segLength > segLengthAllowance:
                        break
                    # If the first point is less than buffer width away from the
                    # reference point, then delete the point.
                    coords.pop(0)

                while len(coords) > 1 and coords[0].distance(endPoint) < ptDistAllowance:
                    segLength = coords[0].distance(coords[1])
                    if len(coords) <= 1 or segLength > segLengthAllowance:
                        break
                    coords.pop(0)

                # Clean up the back of the list.
                while len(coords) > 1 and coords[-1].distance(startPoint) < ptDistAllowance:
                    segLength = coords[-1].distance(coords[-2])
                    if len(coords) <= 1 or segLength > segLengthAllowance:
                        break
                    coords.pop()

                while len(coords) > 1 and coords[-1].distance(endPoint) < ptDistAllowance:
                    segLength = coords[-1].distance(coords[-2])
                    if len(coords) <= 1 or segLength > segLengthAllowance:
                        break
                    coords.pop()

                # Add the coordinates to the resultant line string.
                if len(coords) > 1:
                    mergedLinesGeom.append(
                        self._factory.createLineString(coords)
                        )

                mergedLines.pop()
        
        if len(mergedLinesGeom) > 1:
            return self._factory.createMultiLineString(mergedLinesGeom)
        elif len(mergedLinesGeom) == 1:
            return mergedLinesGeom[0]
        else:
            return self._factory.createLineString()

    @staticmethod
    def depthDelta(label) -> int:
        """
         * Compute the change in depth as an edge is crossed from R to L
        """
        lLoc = label.getLocation(0, Position.LEFT)
        rLoc = label.getLocation(0, Position.RIGHT)
        if lLoc == Location.INTERIOR and rLoc == Location.EXTERIOR:
            return 1
        elif lLoc == Location.EXTERIOR and rLoc == Location.INTERIOR:
            return -1
        return 0

    def computeNodeEdges(self, bufferSegStrList: list, precisionModel) -> None:
        noder = self.getNoder(precisionModel)

        noder.computeNodes(bufferSegStrList)

        # SegmentString
        nodedSegStrings = noder.getNodedSubStrings()
        logger.debug("buffer.computeNodeEdges noder:%s nodedSegStrings:%s", id(noder), len(nodedSegStrings))
        for segStr in nodedSegStrings:

            # Label
            label = Label(segStr.context)
            # CoordinateSequence
            cs = CoordinateSequence.removeRepeatedPoints(segStr.coords)
            if len(cs) < 2:
                logger.debug("buffer.computeNodeEdges cs:%s", cs)
                continue

            edge = Edge(cs, label)
            self.insertUniqueEdge(edge)

    def insertUniqueEdge(self, edge) -> None:
        """
         * Inserted edges are checked to see if an identical edge already
         * exists.
         * If so, the edge is not inserted, but its label is merged
         * with the existing edge.
        """
        # logger.debug("buffer.insertUniqueEdge()")

        existingEdge = self.edgeList.findEqualEdge(edge)
        if existingEdge is not None:
            existingLabel = existingEdge.label
            labelToMerge = Label(edge.label)

            # check if new edge is in reverse direction to existing edge
            # if so, must flip the label before merging it
            if not existingEdge.isPointwiseEqual(edge):
                labelToMerge.flip()

            existingLabel.merge(labelToMerge)
            # compute new depth delta of sum of edges
            mergeDelta = BufferBuilder.depthDelta(labelToMerge)
            existingDelta = BufferBuilder.depthDelta(existingLabel)
            newDelta = mergeDelta + existingDelta
            existingEdge.depthDelta = newDelta
        else:
            # add this new edge to the list of edges in this graph
            self.edgeList.add(edge)
            edge.depthDelta = BufferBuilder.depthDelta(edge.label)

    def createSubGraphs(self, graph, subGraphList: list) -> None:
        nodes = graph.nodes
        for node in nodes:
            if not node.isVisited:
                subGraph = BufferSubGraph()
                subGraph.create(node)
                subGraphList.append(subGraph)
        """
         * Sort the subgraphs in descending order of their rightmost coordinate
         * This ensures that when the Polygons for the subgraphs are built,
         * subgraphs for exteriors will have been built before the subgraphs for
         * any interiors they contain
        """
        quicksort(subGraphList, bufferSubgraphGT)

    def buildSubGraphs(self, subGraphList: list, polyBuilder) -> None:
        """
         * Completes the building of the input subgraphs by
         * depth-labelling them,
         * and adds them to the PolygonBuilder.
         * The subgraph list must be sorted in rightmost-coordinate order.
         *
         * @param subgraphList the subgraphs to build
         * @param polyBuilder the overlay.PolygonBuilder which will build
         *        the final polygons
        """
        # BufferSubgraph
        processedGraphs = []
        for i, subgraph in enumerate(subGraphList):
            # Coordinate
            p = subgraph.rightMostCoord

            # SubgraphDepthLocater
            locater = SubgraphDepthLocater(processedGraphs)
            outsideDepth = locater.getDepth(p)
            
            subgraph.computeDepth(outsideDepth)
            subgraph.findResultEdges()

            processedGraphs.append(subgraph)

            polyBuilder.add(subgraph)

    def getNoder(self, precisionModel):
        """
         * Return the externally-set noding.Noder OR a newly created
         * one using the given precisionModel.
         * NOTE: if an externally-set noding.Noder is available no
         * check is performed to ensure it will use the
         * given PrecisionModel
        """
        # this doesn't change workingNoder precisionModel!
        if self.workingNoder is not None:
            return self.workingNoder

        # otherwise use a fast (but non-robust) noder

        if self.li is not None:
            # reuse existing IntersectionAdder and LineIntersector
            self.li.precisionModel = precisionModel

        else:
            self.li = LineIntersector(precisionModel)
            self.si = IntersectionAdder(self.li)

        noder = MCIndexNoder(self.si)
        return noder

    def createEmptyResultGeometry(self):
        """
         * Gets the standard result for an empty buffer.
         * Since buffer always returns a polygonal result,
         * this is chosen to be an empty polygon.
         *
         * @return the empty result geometry.
        """
        # Geometry
        return self._factory.createPolygon(None, None)
