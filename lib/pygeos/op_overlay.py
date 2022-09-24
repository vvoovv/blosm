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


from .shared import (
    logger,
    TopologyException,
    Location,
    Position,
    Envelope,
    Coordinate,
    GeometryTransformer,
    GeomTypeId,
    PrecisionModel
    )
from .geomgraph import (
    Label,
    PlanarGraph,
    GeometryGraphOperation,
    Node,
    DirectedEdgeStar,
    EdgeRing,
    EdgeList,
    EdgeNodingValidator
    )
from .algorithms import (
    CGAlgorithms,
    PointLocator,
    LineSegment,
    UniqueCoordinateArrayFilter
    )
from .precision import CommonBitsRemover


class GeomPtrPair():
    # geom.GeomPtrPair
    def __init__(self):
        self.first = None
        self.second = None


class SnapTransformer(GeometryTransformer):

    def __init__(self, snapTol, coords):
        GeometryTransformer.__init__(self)
        self.snapTol = snapTol
        self.coords = coords

    def snapLine(self, coords):
        snapper = LineStrigSnapper(coords, self.snapTol)
        snapped = snapper.snapTo(self.coords)
        return self._factory.coordinateSequenceFactory.create(snapped)

    def transformCoordinates(self, coords, parent):
        return self.snapLine(coords)


class LineStrigSnapper():
    """
     * Snaps the vertices and segments of a LineString to a set
     * of target snap vertices.
     *
     * A snapping distance tolerance is used to control where snapping is performed.
    """
    def __init__(self, coords, snapTolerance: float):
        """
         * Creates a new snapper using the given points
         * as source points to be snapped.
         *
         * @param nSrcPts the points to snap
         * @param nSnapTolerance the snap tolerance to use
         """
        self.coords = coords
        self.snapTolerance = snapTolerance
        self.allowSnappingToSourceVertices = False
        self.isClosed = len(coords) > 1 and coords[0] == coords[-1]

    def snapTo(self, coords):
        # Snap points are assumed to be all distinct points (a set would be better, uh ?)
        coordList = list(self.coords)
        self.snapVertices(coordList, coords)
        self.snapSegments(coordList, coords)
        return coordList

    def snapVertices(self, srcCoords, snapPts) -> None:

        # Modifies first arg
        end = len(srcCoords)

        # nothing to do if there are no source coords..
        if end == 0:
            return

        if self.isClosed:
            end -= 1

        for snapPt in snapPts:

            vertpos = self.findVertexToSnap(snapPt, srcCoords, 0, end)

            if vertpos == end:
                continue

            srcCoords[vertpos] = snapPt

            # keep final closing point in synch (rings only)
            if vertpos == 0 and self.isClosed:
                srcCoords[-1] = snapPt

    def findSnapForVertex(self, pt, snapPts):
        # not used internally
        end = len(snapPts)
        candidate = end
        minDist = self.snapTolerance
        for i, coord in enumerate(snapPts):

            if coord == pt:
                return end

            dist = coord.distance(pt)
            if dist < minDist:
                minDist = dist
                candidate = i

        return candidate

    def snapSegments(self, srcCoords, snapPts)-> None:
        """
         * Snap segments of the source to nearby snap vertices.
         *
         * Source segments are "cracked" at a snap vertex.
         * A single input segment may be snapped several times
         * to different snap vertices.
         *
         * For each distinct snap vertex, at most one source segment
         * is snapped to.  This prevents "cracking" multiple segments
         * at the same point, which would likely cause
         * topology collapse when being used on polygonal linework.
         *
         * @param srcCoords the coordinates of the source linestring to be snapped
         *                  the object will be modified (coords snapped)
         * @param snapPts the target snap vertices
        """
        if len(srcCoords) == 0:
            return

        for i, coord in enumerate(snapPts):

            end = len(srcCoords) - 1

            segpos = self.findSegmentToSnap(coord, srcCoords, 0, end)

            if segpos == end:
                continue
            # Check if the snap point falls outside of the segment
            # If the snap point is outside, this means that an endpoint
            # was not snap where it should have been
            # so what we should do is re-snap the endpoint to this
            # snapPt and then snap the closest between this and
            # previous (for pf < 0.0) or next (for pf > 1.0) segment
            # to the old endpoint.
            #     --strk May 2013
            to = segpos + 1
            seg = LineSegment(srcCoords[segpos], srcCoords[to])
            pf = seg.projectionFactor(coord)
            if pf >= 1.0:

                newSnapPt = Coordinate(seg.p1.x, seg.p1.y)
                seg.p1 = coord
                srcCoords[to] = coord

                # now snap from-to (segpos) or to-next (segpos++) to newSnapPt
                if to == end:
                    if self.isClosed:
                        srcCoords[0] = coord
                        to = 0
                    else:
                        srcCoords.insert(to, newSnapPt)
                        continue

                to += 1
                nextSeg = LineSegment(seg.p1, srcCoords[to])

                if nextSeg.distance(newSnapPt) < seg.distance(newSnapPt):
                    srcCoords.insert(to, newSnapPt)
                else:
                    segpos += 1
                    srcCoords.insert(segpos, newSnapPt)

            elif pf <= 0.0:

                newSnapPt = Coordinate(seg.p0.x, seg.p0.y)
                seg.p0 = coord
                srcCoords[segpos] = coord

                # now snap prev-from (--segpos) or from-to (segpos) to newSnapPt
                if segpos == 0:
                    if self.isClosed:
                        segpos = len(srcCoords) - 1
                        # sync to end point
                        srcCoords[segpos] = coord
                    else:
                        segpos += 1
                        srcCoords.insert(segpos, newSnapPt)
                        continue

                segpos -= 1
                prevSeg = LineSegment(srcCoords[segpos], seg.p0)

                if prevSeg.distance(newSnapPt) < seg.distance(newSnapPt):
                    # insert into prev segment
                    segpos += 1
                    srcCoords.insert(segpos, newSnapPt)
                else:
                    # insert must happen one-past first point (before next point)
                    srcCoords.insert(to, newSnapPt)
            else:
                # insert must happen one-past first point (before next point)
                segpos += 1
                srcCoords.insert(segpos, coord)

    def findSegmentToSnap(self, snapPt, srcCoords: list, start: int, end: int) -> int:
        """
         * Finds a src segment which snaps to (is close to) the given snap
         * point.
         * Only a single segment is selected for snapping.
         * This prevents multiple segments snapping to the same snap vertex,
         * which would almost certainly cause invalid geometry
         * to be created.
         * (The heuristic approach to snapping used here
         * is really only appropriate when
         * snap pts snap to a unique spot on the src geometry.)
         *
         * Also, if the snap vertex occurs as a vertex in the src
         * coordinate list, no snapping is performed (may be changed
         * using setAllowSnappingToSourceVertices).
         *
         * @param from
         *        an iterator to first point of first segment to be checked
         *
         * @param too_far
         *        an iterator to last point of last segment to be checked
         *
         * @returns an iterator to the snapped segment or
         *          too_far if no segment needs snapping
         *          (either none within snapTol distance,
         *           or one found on the snapPt)
        """
        seg = LineSegment()
        minDist = self.snapTolerance
        # make sure the first closer then
        # snapTolerance is accepted
        match = end
        for i in range(start, end):
            seg.p0 = srcCoords[i]
            to = i + 1
            seg.p1 = srcCoords[to]
            """
             * Check if the snap pt is equal to one of
             * the segment endpoints.
             *
             * If the snap pt is already in the src list,
             * don't snap at all (unless allowSnappingToSourceVertices
             * is set to true)
            """
            if seg.p0 == snapPt or seg.p1 == snapPt:
                if self.allowSnappingToSourceVertices:
                    continue
                else:
                    return end

            dist = seg.distance(snapPt)
            if dist >= minDist:
                continue

            if dist == 0.0:
                return start

            match = start
            minDist = dist

        return match

    def findVertexToSnap(self, snapPt, srcCoords: list, start: int, end: int) -> int:
        match = end
        minDist = self.snapTolerance
        for i in range(start, end):
            c0 = srcCoords[i]
            dist = c0.distance(snapPt)
            if dist >= minDist:
                continue
            if dist == 0.0:
                return i
            match = i
            minDist = dist
        return match


class GeometrySnapper():
    """
     * Snaps the vertices and segments of a {@link Geometry}
     * to another Geometry's vertices.
     *
     * A snap distance tolerance is used to control where snapping is performed.
     * Snapping one geometry to another can improve
     * robustness for overlay operations by eliminating
     * nearly-coincident edges
     * (which cause problems during noding and intersection calculation).
     * Too much snapping can result in invalid topology
     * being created, so the number and location of snapped vertices
     * is decided using heuristics to determine when it
     * is safe to snap.
     * This can result in some potential snaps being omitted, however.
    """
    # eventually this will be determined from the geometry topology
    snapPrecisionFactor = 10e-9

    def __init__(self, geom):
        """
         * Creates a new snapper acting on the given geometry
         *
         * @param g the geometry to snap
        """
        self.geom = geom

    @staticmethod
    def snap(g0, g1, snapTolerance: float, ret) -> None:
        """
         * Snaps two geometries together with a given tolerance.
         *
         * @param g0 a geometry to snap
         * @param g1 a geometry to snap
         * @param snapTolerance the tolerance to use
         * @param ret the snapped geometries as a pair of smart pointers
         *            (output parameter)
        """
        snapper0 = GeometrySnapper(g0)
        ret.first = snapper0.snapTo(g1, snapTolerance)
        snapper1 = GeometrySnapper(g1)
        ret.second = snapper1.snapTo(ret.first, snapTolerance)

    @staticmethod
    def snapToSelf(geom, snapTolerance: float, cleanResult: bool) -> None:
        snapper = GeometrySnapper(geom)
        return snapper._snapToSelf(geom, snapTolerance)

    def snapTo(self, geom, snapTolerance: float):
        """
         * Snaps the vertices in the component {@link LineString}s
         * of the source geometry to the vertices of the given snap geometry
         * with a given snap tolerance
         *
         * @param g a geometry to snap the source to
         * @param snapTolerance
         * @return a new snapped Geometry
        """
        snapPts = self.extractTargetCoordinates(geom)
        snapTrans = SnapTransformer(snapTolerance, snapPts)
        return snapTrans.transform(self.geom)

    def _snapToSelf(self, snapTolerance: float, cleanResult: bool=False):
        """
         * Snaps the vertices in the component {@link LineString}s
         * of the source geometry to the vertices of itself
         * with a given snap tolerance and optionally cleaning the result.
         *
         * @param snapTolerance
         * @param cleanResult clean the result
         * @return a new snapped Geometry
        """
        snapPts = self.extractTargetCoordinates(self.geom)
        snapTrans = SnapTransformer(snapTolerance, snapPts)
        result = snapTrans.transform(self.geom)
        if (cleanResult and
                self.geom.type_id in [
                    GeomTypeId.GEOS_POLYGON,
                    GeomTypeId.GEOS_MULTIPOLYGON
                    ]):

            result = None
        return result

    @staticmethod
    def computeOverlaySnapTolerance(g1, g2=None) -> float:
        """
         * Estimates the snap tolerance for a Geometry, taking into account
         * its precision model.
         *
         * @param g a Geometry
         * @return the estimated snap tolerance
        """
        if g2 is None:
            snapTolerance = GeometrySnapper.computeSizeBasedSnapTolerance(g1)
            """
             * Overlay is carried out in the precision model
             * of the two inputs.
             * If this precision model is of type FIXED, then the snap tolerance
             * must reflect the precision grid size.
             * Specifically, the snap tolerance should be at least
             * the distance from a corner of a precision grid cell
             * to the centre point of the cell.
            """
            pm = g1.precisionModel
            if pm.modelType == PrecisionModel.FIXED:
                fixedSnapTol = (1.0 / pm.scale) * 2.0 / 1.415
                if fixedSnapTol > snapTolerance:
                    snapTolerance = fixedSnapTol
            return snapTolerance

        else:
            return min(GeometrySnapper.computeOverlaySnapTolerance(g1),
                    GeometrySnapper.computeOverlaySnapTolerance(g2))

    @staticmethod
    def computeSizeBasedSnapTolerance(geom) -> float:
        """
        """
        env = geom.envelope
        minDimension = min(env.width, env.height)
        snapTol = minDimension * GeometrySnapper.snapPrecisionFactor
        return snapTol

    def extractTargetCoordinates(self, geom) -> list:
        """
         * Extract target (unique) coordinates
        """
        snapPts = []
        filter = UniqueCoordinateArrayFilter(snapPts)
        geom.apply_ro(filter)
        return snapPts


class SnapOverlayOp():

    def __init__(self, g0, g1):
        self.geom0 = g0
        self.geom1 = g1
        self.snapTolerance = 0.0
        # CommonBitsRemover
        self.cbr = None
        self.computeSnapTolerance()

    @staticmethod
    def overlayOp(g0, g1, opCode: int):
        op = SnapOverlayOp(g0, g1)
        return op.getResultGeometry(opCode)

    @staticmethod
    def intersection(g0, g1):
        return SnapOverlayOp.overlayOp(g0, g1, OverlayOp.opINTERSECTION)

    @staticmethod
    def union(g0, g1):
        return SnapOverlayOp.overlayOp(g0, g1, OverlayOp.opUNION)

    @staticmethod
    def difference(g0, g1):
        return SnapOverlayOp.overlayOp(g0, g1, OverlayOp.opDIFFERENCE)

    @staticmethod
    def symDifference(g0, g1):
        return SnapOverlayOp.overlayOp(g0, g1, OverlayOp.opSYMDIFFERENCE)

    def getResultGeometry(self, opCode: int):
        # geom.GeomPtrPair
        prepGeom = GeomPtrPair()
        self.snap(prepGeom)
        result = OverlayOp.overlayOp(prepGeom.first, prepGeom.second, opCode)
        self.prepareResult(result)
        return result

    def computeSnapTolerance(self) -> None:
        self.snapTolerance = GeometrySnapper.computeOverlaySnapTolerance(self.geom0, self.geom1)

    def snap(self, ret) -> None:
        # geom.GeomPtrPair
        remGeom = GeomPtrPair()
        # clone geometry
        self.removeCommonBits(self.geom0, self.geom1, remGeom)
        GeometrySnapper.snap(remGeom.first, remGeom.second, self.snapTolerance, ret)

    def removeCommonBits(self, geom0, geom1, ret) -> None:
        self.cbr = CommonBitsRemover()
        self.cbr.add(geom0)
        self.cbr.add(geom1)
        ret.first = self.cbr.removeCommonBits(geom0.clone())
        ret.second = self.cbr.removeCommonBits(geom1.clone())

    def prepareResult(self, geom) -> None:
        # re-adds common bits to the given geom
        self.cbr.addCommonBits(geom)


class SnapIfNeededOverlayOp():
    """
     * Performs an overlay operation using snapping and enhanced precision
     * to improve the robustness of the result.
     * This class only uses snapping
     * if an error is detected when running the standard JTS overlay code.
     * Errors detected include thrown exceptions
     * (in particular, {@link TopologyException})
     * and invalid overlay computations.
     *
     * @author Martin Davis
     * @version 1.7
    """
    def __init__(self, g0, g1) -> None:
        self.geom0 = g0
        self.geom1 = g1

    @staticmethod
    def overlayOp(g0, g1, opCode: int):
        op = SnapIfNeededOverlayOp(g0, g1)
        return op.getResultGeometry(opCode)

    @staticmethod
    def intersection(g0, g1):
        return SnapIfNeededOverlayOp.overlayOp(g0, g1, OverlayOp.opINTERSECTION)

    @staticmethod
    def union(g0, g1):
        return SnapIfNeededOverlayOp.overlayOp(g0, g1, OverlayOp.opUNION)

    @staticmethod
    def difference(g0, g1):
        return SnapIfNeededOverlayOp.overlayOp(g0, g1, OverlayOp.opDIFFERENCE)

    @staticmethod
    def symDifference(g0, g1):
        return SnapIfNeededOverlayOp.overlayOp(g0, g1, OverlayOp.opSYMDIFFERENCE)

    def getResultGeometry(self, opCode: int):
        result = None
        isSuccess = False
        savedException = None
        try:
            # try basic operation with input geometries
            result = OverlayOp.overlayOp(self.geom0, self.geom1, opCode)
            # isValid = True
            # not needed if noding validation is used
            #  boolean isValid = OverlayResultValidator.isValid(geom[0], geom[1], OverlayOp.INTERSECTION, result);
            isSuccess = True
        except TopologyException as ex:
            savedException = ex
            # ignore this exception, since the operation will be rerun

        if not isSuccess:
            # this may still throw an exception
            # if so, throw the original exception since it has the input coordinates
            try:
                result = SnapOverlayOp.overlayOp(self.geom0, self.geom1, opCode)
            except:
                raise savedException

        return result


class MinimalEdgeRing(EdgeRing):
    """
     * A ring of Edges with the property that no node
     * has degree greater than 2.
     *
     * These are the form of rings required
     * to represent polygons under the OGC SFS spatial data model.
     *
     * @see operation.overlay.MaximalEdgeRing
    """
    def __init__(self, start, factory):
        EdgeRing.__init__(self, start, factory)
        # May raise TopologyException
        self.computePoints(start)
        self.computeRing()

    def getNext(self, de):
        return de.nextMin

    def setEdgeRing(self, de, er) -> None:
        de.minEdgeRing = er


class MaximalEdgeRing(EdgeRing):
    """
     * A ring of Edges which may contain nodes of degree > 2.
     *
     * A MaximalEdgeRing may represent two different spatial entities:
     *
     * - a single polygon possibly containing inversions (if the ring is oriented CW)
     * - a single hole possibly containing exversions (if the ring is oriented CCW)
     *
     * If the MaximalEdgeRing represents a polygon,
     * the interiors of the polygon is strongly connected.
     *
     * These are the form of rings used to define polygons under some spatial data models.
     * However, under the OGC SFS model, MinimalEdgeRings are required.
     * A MaximalEdgeRing can be converted to a list of MinimalEdgeRings using the
     * buildMinimalRings() method.
    """
    def __init__(self, start, factory):
        EdgeRing.__init__(self, start, factory)
        # May raise TopologyException
        self.computePoints(start)
        self.computeRing()

    def getNext(self, de):
        return de.next

    def setEdgeRing(self, de, er) -> None:
        de.edgeRing = er

    def buildMinimalRings(self, minEdgeRings: list) -> None:
        de = self.startDe
        while (True):
            if de.minEdgeRing is None:
                # May raise TopologyException
                minEr = MinimalEdgeRing(de, self._factory)
                minEdgeRings.append(minEr)
            de = de.next
            if de is self.startDe:
                break

    def linkDirectedEdgesForMinimalEdgeRings(self) -> None:
        # DirectedEdge
        de = self.startDe
        while (True):
            # Node
            node = de.node
            # DirectedEdgeStar
            star = node.star
            star.linkMinimalDirectedEdges(self)
            de = de.next
            if de is self.startDe:
                break


class PolygonBuilder():
    """
     * Forms Polygon out of a graph of geomgraph.DirectedEdge.
     *
     * The edges to use are marked as being in the result Area.
    """
    def __init__(self, newFactory):
        self._factory = newFactory
        self._exteriorList = []

    def add(self, graph, nodes=None) -> None:
        """
         * Add a complete graph.
         * or Add a set of edges and nodes, which form a graph.
         * The graph is assumed to contain one or more polygons,
         * possibly with interiors.
        """
        if nodes is None:
            # DirectedEdge
            dirEdges = list(graph._edgeEnds)

            # Node
            nodes = list(graph.nodes)

            logger.debug("PolygonBuilder.add() PlanarGraph has %s EdgeEnds and %s Nodes", len(dirEdges), len(nodes))

            self.add(dirEdges, nodes)

        else:
            # direEdges + nodes
            dirEdges = graph
            PlanarGraph.static_linkResultDirectedEdges(nodes)

            # MaximalEdgeRing
            maxEdgeRings = []
            self.buildMaximalEdgeRings(dirEdges, maxEdgeRings)

            # EdgeRing
            freeHoleList = []

            # MaximalEdgeRing
            edgeRings = []

            self.buildMinimalEdgeRings(maxEdgeRings,
                    self._exteriorList,
                    freeHoleList,
                    edgeRings)

            self.sortShellsAndHoles(edgeRings, self._exteriorList, freeHoleList)
            self.placeFreeHoles(self._exteriorList, freeHoleList)
            """
            logger.debug("PolygonBuilder.add() dirEdges: %s nodes:%s maxEdgeRings:%s _exteriorList:%s freeholeList:%s",
                len(dirEdges),
                len(nodes),
                len(maxEdgeRings),
                len(self._exteriorList),
                len(freeHoleList),
                )
            """

    def getPolygons(self) -> list:
        return self.computePolygons(self._exteriorList)

    def containsPoint(self, coord) -> bool:
        """
         * Checks the current set of exteriors (with their associated interiors) to
         * see if any of them contain the point.
        """
        for er in self._exteriorList:
            if er.containsPoint(coord):
                return True
        return False

    def buildMaximalEdgeRings(self, dirEdges: list, maxEdgeRings: list) -> None:
        """
         * For all DirectedEdges in result, form them into MaximalEdgeRings
         *
         * @param maxEdgeRings
         *   Formed MaximalEdgeRings will be pushed to this vector.
         *   Ownership of the elements is transferred to caller.
        """
        logger.debug("PolygonBuilder.buildMaximalEdgeRings got %s dirEdges", len(dirEdges))
        # oldSize = len(maxEdgeRings)

        for i, de in enumerate(dirEdges):
            """
            logger.debug("%s %s inResult:%s isArea:%s label:%s %s",
                type(de).__name__,
                i,
                de.isInResult,
                de.label.isArea(),
                de.label,
                de.printEdge())
            """
            if de.isInResult and de.label.isArea():
                # if this edge has not yet been processed
                if de.edgeRing is None:
                    # may rise TopologyException
                    er = MaximalEdgeRing(de, self._factory)
                    maxEdgeRings.append(er)
                    er.isInResult = True

        # logger.debug("pushed %s maxEdgeRings", len(maxEdgeRings) - oldSize)

    def buildMinimalEdgeRings(self,
            maxEdgeRings: list,
            newShellList: list,
            freeHoleList: list,
            edgeRings: list) -> None:

        """

        """
        for i, er in enumerate(maxEdgeRings):

            logger.debug("PolygonBuilder.buildMinimalEdgeRings(): maxEdgeRing %s has maxNodeDegree %s",
                i,
                er.maxNodeDegree)

            if er.maxNodeDegree > 2:
                er.linkDirectedEdgesForMinimalEdgeRings()

                # MinimalEdgeRing
                minEdgeRings = []
                er.buildMinimalRings(minEdgeRings)

                # EdgeRing
                exterior = self.findShell(minEdgeRings)

                if exterior is not None:
                    self.placePolygonHoles(exterior, minEdgeRings)
                    newShellList.append(exterior)
                else:
                    freeHoleList.extend(minEdgeRings)

            else:
                edgeRings.append(er)

    def findShell(self, minEdgeRings: list):
        """
         * This method takes a list of MinimalEdgeRings derived from a
         * MaximalEdgeRing, and tests whether they form a Polygon.
         * This is the case if there is a single exterior
         * in the list.  In this case the exterior is returned.
         * The other possibility is that they are a series of connected
         * interiors, in which case no exterior is returned.
         *
         * @return the exterior geomgraph.EdgeRing, if there is one
         * @return NULL, if all the rings are interiors
        """
        logger.debug("PolygonBuilder.findShell got %s minEdgeRings\n", len(minEdgeRings))
        exteriorCount = 0
        exterior = None

        for er in minEdgeRings:
            if not er.isHole:
                exterior = er
                exteriorCount += 1

        if exteriorCount > 1:
            raise TopologyException("found two exteriors in MinimalEdgeRing list")

        return exterior

    def placePolygonHoles(self, exterior, minEdgeRings: list) -> None:
        """
         * This method assigns the interiors for a Polygon (formed from a list of
         * MinimalEdgeRings) to its exterior.
         * Determining the interiors for a MinimalEdgeRing polygon serves two
         * purposes:
         *
         *  - it is faster than using a point-in-polygon check later on.
         *  - it ensures correctness, since if the PIP test was used the point
         *    chosen might lie on the exterior, which might return an incorrect
         *    result from the PIP test
        """
        for er in minEdgeRings:
            if er.isHole:
                er.setShell(exterior)

    def sortShellsAndHoles(self, edgeRings: list, newShellList: list, freeHoleList: list) -> None:
        """
         * For all rings in the input list,
         * determine whether the ring is a exterior or a hole
         * and add it to the appropriate list.
         * Due to the way the DirectedEdges were linked,
         * a ring is a exterior if it is oriented CW, a hole otherwise.
        """
        logger.debug("PolygonBuilder.sortShellsAndHoles() edgeRings:%s", len(edgeRings))

        for i, er in enumerate(edgeRings):

            logger.debug("EdgeRing %s isHole:%s", i, er.isHole)

            if er.isHole:
                freeHoleList.append(er)
            else:
                newShellList.append(er)

    def placeFreeHoles(self, newShellList: list, freeHoleList: list) -> None:
        """
         * This method determines finds a containing exterior for all interiors
         * which have not yet been assigned to a exterior.
         *
         * These "free" interiors should all be <b>properly</b> contained in
         * their parent exteriors, so it is safe to use the
         * findEdgeRingContaining method.
         * This is the case because any interiors which are NOT
         * properly contained (i.e. are connected to their
         * parent exterior) would have formed part of a MaximalEdgeRing
         * and been handled in a previous step.
         *
         * @throws TopologyException if a hole cannot be assigned to a exterior
        """
        for hole in freeHoleList:

            if hole._exterior is None:

                exterior = self.findEdgeRingContaining(hole, newShellList)

                if exterior is None:
                    """
                    for rIt in newShellList:
                        rIt.toPolygon(self._factory)
                    hole.toPolygon(self._factory)
                    """
                    raise TopologyException("PolygonBuilder.placeFreeHoles() unable to assign hole to a exterior")

                hole.setShell(exterior)

    def findEdgeRingContaining(self, testEr, newShellList: list):
        """
         * Find the innermost enclosing exterior geomgraph.EdgeRing containing the
         * argument geomgraph.EdgeRing, if any.
         *
         * The innermost enclosing ring is the <i>smallest</i> enclosing ring.
         * The algorithm used depends on the fact that:
         *
         * ring A contains ring B iff envelope(ring A)
         * contains envelope(ring B)
         *
         * This routine is only safe to use if the chosen point of the hole
         * is known to be properly contained in a exterior
         * (which is guaranteed to be the case if the hole does not touch
         * its exterior)
         *
         * @return containing geomgraph.EdgeRing, if there is one
         * @return NULL if no containing geomgraph.EdgeRing is found
        """
        # LinearRing
        testRing = testEr.getLinearRing()
        # Envelope
        testEnv = testRing.envelope
        # Coordinate
        coord = testRing.coords[0]
        # EdgeRing
        minShell = None
        # Envelope
        minEnv = None

        for tryShell in newShellList:
            # LinearRing
            lr = None
            tryRing = tryShell.getLinearRing()
            # Envelope
            tryEnv = tryRing.envelope

            if minShell is not None:
                # LinearRing
                lr = minShell.getLinearRing()
                # Envelope
                minEnv = lr.envelope

            isContained = False

            # CoordinateSequence
            coords = tryRing.coords

            if tryEnv.contains(testEnv) and CGAlgorithms.isPointInRing(coord, coords):
                isContained = True

            if isContained:
                if minShell is None or minEnv.contains(tryEnv):
                    minShell = tryShell

        return minShell

    def computePolygons(self, newShellList: list) -> list:

        logger.debug("PolygonBuilder.computePolygons: got %s exteriors", len(newShellList))

        # Geometry
        resultPolyList = []

        for er in newShellList:
            poly = er.toPolygon(self._factory)
            resultPolyList.append(poly)

        return resultPolyList


class LineBuilder():
    """
     * Forms LineStrings out of a the graph of geomgraph.DirectedEdge
     * created by an OverlayOp.
    """
    def __init__(self, newOp, newFactory, newPtLocator):
        # OverlayOp
        self._op = newOp
        # PointLocator
        self._ptLocator = newPtLocator
        # GeometryFactory
        self._factory = newFactory
        # Edge
        self._lineEdgesList = []
        # LineString
        self._resultLineList = []

    def build(self, opCode: int) -> list:
        """
         * @return a list of the LineStrings in the result of the specified overlay operation
        """
        self.findCorevedLineEdges()
        self.collectLines(opCode)
        self.buildLines(opCode)
        return self._resultLineList

    def collectLineEdge(self, de, opCode: int, edges: list) -> None:
        """
         * Collect line edges which are in the result.
         *
         * Line edges are in the result if they are not part of
         * an area boundary, if they are in the result of the overlay operation,
         * and if they are not covered by a result area.
         *
         * @param de the directed edge to test.
         * @param opCode the overlap operation
         * @param edges the list of included line edges.
        """
        # DirectedEdge
        # include L edges which are in the result
        if de.isLineEdge:
            # Label
            label = de.label
            # Edge
            ed = de.edge
            if ((not de.isVisited) and
                    OverlayOp.isResultOfOp(label, opCode) and
                    (not ed.isCovered)):
                edges.append(ed)
                de.setVisitedEdge(True)

    def findCorevedLineEdges(self) -> None:
        """
         * Find and mark L edges which are "covered" by the result area (if any).
         * L edges at nodes which also have A edges can be checked by checking
         * their depth at that node.
         * L edges at nodes which do not have A edges can be checked by doing a
         * point-in-polygon test with the previously computed result areas
        """
        # first set covered for all L edges at nodes which have A edges too
        nodes = self._op._graph.nodes
        for node in nodes:
            # DirectedEdgeStar
            star = node.star
            star.findCoveredLineEdges()
        """
         * For all L edges which weren't handled by the above,
         * use a point-in-poly test to determine whether they are covered
        """
        ee = self._op._graph._edgeEnds
        for de in ee:
            # Edge
            ed = de.edge
            if de.isLineEdge and not ed.isCoveredSet:
                ed.isCovered = self._op.isCoveredByA(de.coord)

    def collectLines(self, opCode: int) -> None:
        ee = self._op._graph._edgeEnds
        for de in ee:
            self.collectLineEdge(de, opCode, self._lineEdgesList)
            self.collectBoundaryTouchEdge(de, opCode, self._lineEdgesList)

    def buildLines(self, opCode: int) -> None:
        for ed in self._lineEdgesList:
            line = self._factory.createLineString(ed.coords.clone())
            self._resultLineList.append(line)
            ed.isInResult = True

    def labelIsolatedLines(self, edgesList: list) -> None:
        for ed in edgesList:
            label = ed.label
            if ed.isIsolated:
                if label.isNull(0):
                    self.labelIsolatedLine(ed, 0)
                else:
                    self.labelIsolatedLine(ed, 1)

    def collectBoundaryTouchEdge(self, de, opCode: int, edges: list) -> None:
        """
         * Collect edges from Area inputs which should be in the result but
         * which have not been included in a result area.
         * This happens ONLY:
         *
         *  -  during an intersection when the boundaries of two
         *     areas touch in a line segment
         *  -   OR as a result of a dimensional collapse.
         *
        """
        # only interested in area edges
        if de.isLineEdge:
            return
        # already processed
        if de.isVisited:
            return
        # added to handle dimensional collapses
        if de.isInteriorAreaEdge:
            return
        # if the edge linework is already included, don't include it again
        if de.edge.isInResult:
            return

        # include the linework if it's in the result of the operation
        label = de.label
        if opCode == OverlayOp.opINTERSECTION and OverlayOp.isResultOfOp(label, opCode):
            edges.append(de.edge)
            de.setVisitedEdge(True)

    def labelIsolatedLine(self, edge, targetIndex: int) -> None:
        """
         * Label an isolated node with its relationship to the target geometry.
        """
        loc = self._ptLocator.locate(edge.coord, self._op.arg[targetIndex].geom)
        edge.label.setLocation(targetIndex, loc)

    def propagateZ(self, coords):
        """
         * If the given CoordinateSequence has mixed 3d/2d vertexes
         * set Z for all vertexes missing it.
         * The Z value is interpolated between 3d vertexes and copied
         * from a 3d vertex to the end.
        """
        pass


class PointBuilder():
    """
    """
    def __init__(self, newOp, newFactory, newPtLocator):
        # OverlayOp
        self._op = newOp
        # PointLocator
        self._ptLocator = newPtLocator
        # GeometryFactory
        self._factory = newFactory

    def build(self, opCode: int) -> list:
        return []


class OverlayNodeFactory():
    """
     * Creates nodes for use in the geomgraph.PlanarGraph constructed during
     * overlay operations. NOTE: also used by operation.valid
    """
    def createNode(self, coord):
        return Node(coord, DirectedEdgeStar())


class OverlayOp(GeometryGraphOperation):

    opINTERSECTION = 1
    opUNION = 2
    opDIFFERENCE = 3
    opSYMDIFFERENCE = 4

    def toOperationName(self, opCode: int) -> str:
        return (
            'INTERSECTION',
            'UNION',
            'DIFFERENCE',
            'SYMDIFFERENCE'
            )[opCode - 1]

    def __init__(self, geom1, geom2):

        GeometryGraphOperation.__init__(self, geom1, geom2)

        self._ptLocator = PointLocator()
        # GeometryFactory
        self._factory = geom1._factory
        # Geometry
        self._resultGeom = None

        # geomgraph.PlanarGraph
        self._graph = PlanarGraph(OverlayNodeFactory())

        # EdgeList of Edges
        self._edgeList = EdgeList()

        # Polygon
        self._resultPolyList = None

        # LineString
        self._resultLineList = None

        # Point
        self._resultPointList = None

        self._env = Envelope(geom1.envelope)
        self._env.expandToInclude(geom2.envelope)

    @staticmethod
    def overlayOp(geom0, geom1, opCode: int):
        """
         * Computes an overlay operation for the given geometry arguments.
         *
         * @param geom0 the first geometry argument
         * @param geom1 the second geometry argument
         * @param opCode the code for the desired overlay operation
         * @return the result of the overlay operation
         * @throws TopologyException if a robustness problem is encountered
        """
        op = OverlayOp(geom0, geom1)
        logger.debug("******************************\n")
        logger.debug("OverlayOp.overlayOp(%s)\n", op.toOperationName(opCode))
        logger.debug("******************************")
        geom = op.getResultGeometry(opCode)
        return geom

    @staticmethod
    def _isResultOfOp(loc0: int, loc1: int, opCode: int) -> bool:

        if loc0 == Location.BOUNDARY:
            loc0 = Location.INTERIOR

        if loc1 == Location.BOUNDARY:
            loc1 = Location.INTERIOR

        if opCode == OverlayOp.opINTERSECTION:
            return loc0 == Location.INTERIOR and loc1 == Location.INTERIOR

        elif opCode == OverlayOp.opUNION:
            return loc0 == Location.INTERIOR or loc1 == Location.INTERIOR

        elif opCode == OverlayOp.opDIFFERENCE:
            return loc0 == Location.INTERIOR and loc1 != Location.INTERIOR

        elif opCode == OverlayOp.opSYMDIFFERENCE:
            return ((loc0 == Location.INTERIOR and loc1 != Location.INTERIOR) or
                (loc0 != Location.INTERIOR and loc1 == Location.INTERIOR))

        return False

    @staticmethod
    def isResultOfOp(label, opCode: int) -> bool:
        """
         * Tests whether a point with a given topological {Label}
         * relative to two geometries is contained in
         * the result of overlaying the geometries using
         * a given overlay operation.
         *
         * The method handles arguments of {Location#NONE} correctly
         *
         * @param label the topological label of the point
         * @param opCode the code for the overlay operation to test
         * @return true if the label locations correspond to the overlayOpCode
        """
        loc0 = label.getLocation(0)
        loc1 = label.getLocation(1)
        return OverlayOp._isResultOfOp(loc0, loc1, opCode)

    def getResultGeometry(self, opCode: int):
        """
         * Gets the result of the overlay for a given overlay operation.
         *
         * Note: this method can be called once only.
         *
         * @param overlayOpCode the overlay operation to perform
         * @return the compute result geometry
         * @throws TopologyException if a robustness problem is encountered
        """
        self.computeOverlay(opCode)
        return self._resultGeom

    def isCoveredByLA(self, coord) -> bool:
        """
         * This method is used to decide if a point node should be included
         * in the result or not.
         *
         * @return true if the coord point is covered by a result Line
         * or Area geometry
        """
        if self.isCovered(coord, self._resultLineList):
            return True
        if self.isCovered(coord, self._resultPolyList):
            return True
        return False

    def isCoveredByA(self, coord) -> bool:
        """
         * This method is used to decide if an L edge should be included
         * in the result or not.
         *
         * @return true if the coord point is covered by a result Area geometry
        """
        return self.isCovered(coord, self._resultPolyList)

    def insertUniqueEdge(self, edge) -> None:
        """
         * Insert an edge from one of the noded input graphs.
         *
         * Checks edges that are inserted to see if an
         * identical edge already exists.
         * If so, the edge is not inserted, but its label is merged
         * with the existing edge.
        """
        # Edge
        existingEdge = self._edgeList.findEqualEdge(edge)

        if existingEdge is not None:

            # If an identical edge already exists, simply update its label
            # logger.debug("OverlayOp.insertUniqueEdge() found identical edge\n%s", edge)

            label = existingEdge.label
            labelToMerge = edge.label

            if not existingEdge.isPointwiseEqual(edge):
                labelToMerge = Label(edge.label)
                labelToMerge.flip()

            depth = existingEdge.depth

            if depth.isNull():
                depth.add(label)

            depth.add(labelToMerge)
            label.merge(labelToMerge)
        else:
            # logger.debug("OverlayOp.insertUniqueEdge() no matching existing edge\n%s", edge)
            self._edgeList.add(edge)

    def computeOverlay(self, opCode: int) -> None:
        env = None
        env0 = self.arg[0].geom.envelope
        env1 = self.arg[1].geom.envelope

        # Envelope-based optimization only works in floating precision
        if opCode == OverlayOp.opINTERSECTION:
            env = Envelope()
            env0.intersection(env1, env)

        elif opCode == OverlayOp.opDIFFERENCE:
            env = Envelope(env0)

        # copy points from input Geometries.
        # This ensures that any Point geometries
        # in the input are considered for inclusion in the result set
        self.copyPoints(0, env)
        self.copyPoints(1, env)

        # node the input Geometries
        self.arg[0].computeSelfNodes(self._li, False, env)
        self.arg[1].computeSelfNodes(self._li, False, env)

        logger.debug("OverlayOp.computeOverlay: computed SelfNodes")

        # compute intersections between edges of the two input geometries
        self.arg[0].computeEdgeIntersections(self.arg[1], self._li, True, env)

        logger.debug("OverlayOp.computeOverlay: computed EdgeIntersections")
        # logger.debug("OverlayOp.computeOverlay: li: %s", self._li)

        # Edge
        baseSplitEdges = []
        self.arg[0].computeSplitEdges(baseSplitEdges)
        self.arg[1].computeSplitEdges(baseSplitEdges)

        # add the noded edges to this result graph
        # logger.debug("OverlayOp.insertUniqueEdges() at call time:\n%s", "\n".join(
        #     [str(edge) for edge in baseSplitEdges]))
        self.insertUniqueEdges(baseSplitEdges, env)
        logger.debug("OverlayOp.insertUniqueEdges() result:\n%s", "\n".join(
             [str(edge) for edge in self._edgeList]))

        logger.debug("OverlayOp.computeLabelsFromDepths()")
        self.computeLabelsFromDepths()

        logger.debug("OverlayOp.replaceCollapsedEdges()")
        self.replaceCollapsedEdges()

        """
         * Check that the noding completed correctly.
         *
         * This test is slow, but necessary in order to catch
         * robustness failure situations.
         * If an exception is thrown because of a noding failure,
         * then snapping will be performed, which will hopefully avoid
         * the problem.
         * In the future hopefully a faster check can be developed.
        """
        logger.debug("OverlayOp EdgeNodingValidator.checkValid()")
        EdgeNodingValidator.checkValid(self._edgeList)

        # logger.debug("OverlayOp._graph.addEdges() at call time:\n%s", "\n".join(
        #     [str(edge) for edge in self._edgeList]))
        self._graph.addEdges(self._edgeList)
        # logger.debug("OverlayOp._graph.addEdges() after:\n%s", "\n".join(
        #     [str(edge) for edge in self._edgeList]))

        # this can throw TopologyException
        logger.debug("OverlayOp.computeLabelling()")
        self.computeLabelling()
        self.labelIncompleteNodes()
        """
         * The ordering of building the result Geometries is important.
         * Areas must be built before lines, which must be built
         * before points.
         * This is so that lines which are covered by areas are not
         * included explicitly, and similarly for points.
        """
        self.findResultAreaEdges(opCode)
        logger.debug("OverlayOp.findResultAreaEdges() after:\n%s", "\n".join(
             [str(edge) for edge in self._edgeList]))

        self.cancelDuplicateResultEdges()
        logger.debug("OverlayOp.cancelDuplicateResultEdges() after:\n%s", "\n".join(
             [str(edge) for edge in self._edgeList]))

        polyBuilder = PolygonBuilder(self._factory)
        polyBuilder.add(self._graph)

        # Geometry
        self._resultPolyList = polyBuilder.getPolygons()

        lineBuilder = LineBuilder(self, self._factory, self._ptLocator)
        # Geometry
        self._resultLineList = lineBuilder.build(opCode)

        pointBuilder = PointBuilder(self, self._factory, self._ptLocator)
        # Geometry
        self._resultPointList = pointBuilder.build(opCode)

        # Geometry
        self._resultGeom = self.computeGeometry(
            self._resultPointList,
            self._resultLineList,
            self._resultPolyList)

        self.checkObviouslyWrongResult(opCode)

    def insertUniqueEdges(self, edges, env=None) -> None:
        # Edge
        for edge in edges:
            if (env is not None) and not env.intersects(edge.envelope):
                continue
            self.insertUniqueEdge(edge)

    def computeLabelsFromDepths(self) -> None:
        """
         * Update the labels for edges according to their depths.
         *
         * For each edge, the depths are first normalized.
         * Then, if the depths for the edge are equal,
         * this edge must have collapsed into a line edge.
         * If the depths are not equal, update the label
         * with the locations corresponding to the depths
         * (i.e. a depth of 0 corresponds to a Location of EXTERIOR,
         * a depth of 1 corresponds to INTERIOR)
        """
        edges = self._edgeList
        for edge in edges:
            # Edge
            label = edge.label
            depth = edge.depth

            # logger.debug("OverlayOp.computeLabelsFromDepths() before Label:%s depth:%s", label, depth)
            """
             * Only check edges for which there were duplicates,
             * since these are the only ones which might
             * be the result of dimensional collapses.
            """
            if depth.isNull():
                continue

            depth.normalize()
            for i in range(2):
                if (not label.isNull(i)) and label.isArea() and (not depth.isNull(i)):
                    """
                     * if the depths are equal, this edge is the result of
                     * the dimensional collapse of two or more edges.
                     * It has the same location on both sides of the edge,
                     * so it has collapsed to a line.
                    """
                    if depth.getDelta(i) == 0:
                        label.toLine(i)
                    else:
                        """
                         * This edge may be the result of a dimensional collapse,
                         * but it still has different locations on both sides.  The
                         * label of the edge must be updated to reflect the resultant
                         * side locations indicated by the depth values.
                        """
                        assert(not depth.isNull(i, Position.LEFT)), "depth of LEFT side has not been initialized"
                        label.setLocation(i, Position.LEFT, depth.getLocation(i, Position.LEFT))

                        assert(not depth.isNull(i, Position.RIGHT)), "depth of RIGHT side has not been initialized"
                        label.setLocation(i, Position.RIGHT, depth.getLocation(i, Position.RIGHT))

            logger.debug("OverlayOp.computeLabelsFromDepths() result Label:%s depth:%s", label, depth)

    def replaceCollapsedEdges(self) -> None:
        """
         * If edges which have undergone dimensional collapse are found,
         * replace them with a new edge which is a L edge
        """
        edges = self._edgeList
        for i, edge in enumerate(edges):
            if edge.isCollapsed:
                logger.debug(" replacing collapsed edge %s", i)
                edges[i] = edge.getCollapsedEdge()

    def copyPoints(self, geomIndex: int, env=None) -> None:
        """
         * Copy all nodes from an arg geometry into this graph.
         *
         * The node label in the arg geometry overrides any previously
         * computed label for that geomIndex.
         * (E.g. a node may be an intersection node with
         * a previously computed label of BOUNDARY,
         * but in the original arg Geometry it is actually
         * in the interiors due to the Boundary Determination Rule)
        """
        copied = 0
        nodes = self.arg[geomIndex].nodes
        for node in nodes:
            # Node
            coord = node.coord
            # not in JTS
            if (env is not None) and not env.covers(coord):
                continue

            copied += 1
            # Node
            newNode = self._graph.addNode(coord)
            newNode.setLabel(geomIndex, node.label.getLocation(geomIndex))
        """
        logger.debug("Source nodes for geom %s \n%s",
            geomIndex,
            "\n".join([str(node) for node in nodes]))

        logger.debug("Copied %s nodes out of %s for geom %s \n%s", copied,
            len(nodes),
            geomIndex,
            "\n".join([str(node) for node in self._graph.nodes]))
        """

    def computeLabelling(self) -> None:
        """
         * Compute initial labelling for all DirectedEdges at each node.
         *
         * In this step, DirectedEdges will acquire a complete labelling
         * (i.e. one with labels for both Geometries)
         * only if they
         * are incident on a node which has edges for both Geometries
        """
        nodes = self._graph.nodes

        # logger.debug("OverlayOp.computeLabelling(): at call time:\n%s", self._edgeList)
        logger.debug("OverlayOp.computeLabelling() scanning %s nodes from map:", len(nodes))

        for node in nodes:
            # logger.debug(" %s has %s edgeEnds", node, len(node.star))
            node.star.computeLabelling(self.arg)

        self.mergeSymLabels()
        self.updateNodeLabelling()

    def mergeSymLabels(self) -> None:
        """
         * For nodes which have edges from only one Geometry incident on them,
         * the previous step will have left their dirEdges with no
         * labelling for the other Geometry.
         * However, the sym dirEdge may have a labelling for the other
         * Geometry, so merge the two labels.
        """
        nodes = self._graph.nodes
        logger.debug("OverlayOp.mergeSymLabels() scanning %s nodes from map:", len(nodes))
        for node in nodes:
            # DirectedEdgeStar
            node.star.mergeSymLabels()

    def updateNodeLabelling(self) -> None:
        """
         * update the labels for nodes
         * The label for a node is updated from the edges incident on it
         * (Note that a node may have already been labelled
         * because it is a point in one of the input geometries)
        """
        nodes = self._graph.nodes
        logger.debug("OverlayOp.updateNodeLabelling() scanning %s nodes from map:", len(nodes))
        for node in nodes:
            # DirectedEdgeStar
            node.label.merge(node.star.label)
            logger.debug("%s", node)

    def labelIncompleteNodes(self) -> None:
        """
         * Incomplete nodes are nodes whose labels are incomplete.
         *
         * (e.g. the location for one Geometry is NULL).
         * These are either isolated nodes,
         * or nodes which have edges from only a single Geometry incident
         * on them.
         *
         * Isolated nodes are found because nodes in one graph which
         * don't intersect nodes in the other are not completely
         * labelled by the initial process of adding nodes to the nodeList.
         * To complete the labelling we need to check for nodes that
         * lie in the interiors of edges, and in the interiors of areas.
         *
         * When each node labelling is completed, the labelling of the
         * incident edges is updated, to complete their labelling as well.
        """
        nodes = self._graph.nodes
        logger.debug("OverlayOp.labelIncompleteNodes() scanning %s nodes from map:", len(nodes))
        for node in nodes:
            # Label
            label = node.label
            # has only one geometry
            if node.isIsolated:
                if label.isNull(0):
                    self.labelIncompleteNode(node, 0)
                else:
                    self.labelIncompleteNode(node, 1)

            # logger.debug("OverlayOp.labelIncompleteNodes() Node[%s].label:%s", id(node), label)

            # now update the labelling for the DirectedEdges incident on this node
            # DirectedEdgeStar
            node.star.updateLabelling(label)

    def labelIncompleteNode(self, node, geomIndex: int) -> None:
        """
         * Label an isolated node with its relationship to the target geometry.
        """
        logger.debug("OverlayOp.labelIncompleteNode() geomIndex:%s\n%s", geomIndex, node)

        geom = self.arg[geomIndex].geom
        loc = self._ptLocator.locate(node.coord, geom)
        node.label.setLocation(geomIndex, loc)

    def findResultAreaEdges(self, opCode: int) -> None:
        """
         * Find all edges whose label indicates that they are in the result
         * area(s), according to the operation being performed.
         *
         * Since we want polygon exteriors to be
         * oriented CW, choose dirEdges with the interiors of the result
         * on the RHS.
         * Mark them as being in the result.
         * Interior Area edges are the result of dimensional collapses.
         * They do not form part of the result area boundary.
        """
        # EdgeEnd
        ee = self._graph._edgeEnds
        logger.debug("OverlayOp.findResultAreaEdges EdgeEnds: %s", len(ee))

        for i, de in enumerate(ee):
            # mark all dirEdges with the appropriate label
            label = de.label

            if label.isArea() and (not de.isInteriorAreaEdge) and OverlayOp._isResultOfOp(
                    label.getLocation(0, Position.RIGHT),
                    label.getLocation(1, Position.RIGHT),
                    opCode
                    ):
                de.isInResult = True

            logger.debug("%s: isArea:%s inResult:%s %s", i, label.isArea(), de.isInResult, de)

    def cancelDuplicateResultEdges(self) -> None:
        """
         * If both a dirEdge and its sym are marked as being in the result,
         * cancel them out.
        """
        ee = self._graph._edgeEnds
        for de in ee:
            sym = de.sym
            if de.isInResult and sym.isInResult:
                de.isInResult = False
                sym.isInResult = False

    def isCovered(self, coord, geomList: list) -> bool:
        """
         * @return true if the coord is located in the interiors or boundary of
         * a geometry in the list.
        """
        for geom in geomList:
            loc = self._ptLocator.locate(coord, geom)
            if loc != Location.EXTERIOR:
                return True
        return False

    def computeGeometry(self, nResultPointList: list, nResultLineList: list, nResultPolyList: list):
        """
         * Build a Geometry containing all Geometries in the given vectors.
         * Takes element's ownership, vector control is left to caller.
        """
        geomList = []

        # element geometries of the result are always in the order P,L,A
        geomList.extend(nResultPointList)
        geomList.extend(nResultLineList)
        geomList.extend(nResultPolyList)

        # build the most specific geometry possible
        return self._factory.buildGeometry(geomList)

    def checkObviouslyWrongResult(self, opCode: int) -> None:
        pass


class overlayOp():
    """
     * OverlayOp.overlayOp Adapter for use with geom.BinaryOp
    """
    def __init__(self, opCode: int):
        self.opCode = opCode

    def execute(self, geom0, geom1):
        return OverlayOp.overlayOp(geom0, geom1, self.opCode)
