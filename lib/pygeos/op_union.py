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


from .algorithms import (
    PointLocator
    )
from .shared import (
    logger,
    GeomTypeId,
    Location,
    Envelope,
    GeometryExtracter,
    GeometryCombiner,
    PolygonExtracter
    )
from .index_strtree import (
    STRtree,
    ItemsListItem
    )
from .op_binary import BinaryOp, check_valid
from .op_overlay import (
    OverlayOp,
    overlayOp
    )

    
class CascadedUnion():
    """
     * Provides an efficient method of unioning a collection of Geometries
     *
     * This algorithm is more robust than the simple iterated approach
     * of repeatedly unioning each geometry to a result geometry.
    """

    """
     * The effectiveness of the index is somewhat sensitive
     * to the node capacity.
     * Testing indicates that a smaller capacity is better.
     * For an STRtree, 4 is probably a good number (since
     * this produces 2x2 "squares").
    """
    STRTREE_NODE_CAPACITY = 4

    def __init__(self, geoms=None):
        """
         * Creates a new instance to union
         * the given collection of {@link Geometry}s.
         *
         * @param geoms a collection of {@link Geometryal} {@link Geometry}s
         *        ownership of elements _and_ vector are left to caller.
        """
        self.geoms = geoms
        self._factory = None

    @staticmethod
    def union(polys):
        return CascadedUnion(polys)._union()

    def _union(self, geoms=None, start=None, end=None):
        """
         * Computes the union of a collection of {@link Geometry}s.
         *
         * @param geoms a collection of {@link Geometry}s.
         *        ownership of elements _and_ vector are left to caller.
         * @return the union of the input geometries
         * @return null if no input geometries were provided
        """
        if geoms is None:
            geoms = self.geoms

        if geoms is None:
            return None

        if start is not None and end is not None:
            geoms = [geoms[i] for i in range(start, end)]

        if len(geoms) == 0:
            return None

        self._factory = geoms[0]._factory

        index = STRtree(CascadedUnion.STRTREE_NODE_CAPACITY)

        for geom in geoms:
            index.insert(geom.envelope, geom)

        itemTree = index.itemsTree()
        return self.unionTree(itemTree)

    def unionTree(self, geomTree):
        """
         * Recursively unions all subtrees in the list into single geometries.
         * The result is a list of Geometry's only
        """
        geoms = self.reduceToGeometries(geomTree)
        return self.binaryUnion(geoms)

    def binaryUnion(self, geoms):
        """
         * Unions a list of geometries
         * by treating the list as a flattened binary tree,
         * and performing a cascaded union on the tree.
         * @param geom GeometryListHolder
        """
        return self._binaryUnion(geoms, 0, len(geoms))

    def _binaryUnion(self, geoms, start, end):
        """
         * Unions a section of a list using a recursive binary union on each half
         * of the section.
         *
         * @param geoms GeometryListHolder
         * @param start
         * @param end
         * @return the union of the list section
        """
        if end - start < 2:
            return self.unionSafe(geoms[start], None)
        elif end - start == 2:
            return self.unionSafe(geoms[start], geoms[start + 1])
        else:
            mid = int((end + start) / 2)
            geom0 = self._binaryUnion(geoms, start, mid)
            geom1 = self._binaryUnion(geoms, mid, end)
            return self.unionSafe(geom0, geom1)

    def reduceToGeometries(self, geomTree):
        """
         * Reduces a tree of geometries to a list of geometries
         * by recursively unioning the subtrees in the list.
         *
         * @param geomTree a tree-structured list of geometries
         * @return a list of Geometrys
        """
        # logger.debug("%s.reduceToGeometries:\n%s", type(self).__name__, geomTree)
        geoms = []
        for item in geomTree:
            if item.t == ItemsListItem.item_is_list:
                geom = self.unionTree(item.l)
                geoms.append(geom)

            elif item.t == ItemsListItem.item_is_geometry:
                geoms.append(item.g)

            else:
                assert(0), "should never be reached"
        return geoms

    def unionSafe(self, geom0, geom1):
        """
         * Computes the union of two geometries,
         * either of both of which may be null.
         *
         * @param g0 a Geometry
         * @param g1 a Geometry
         * @return the union of the input(s)
         * @return null if both inputs are null
        """
        if geom0 is None and geom1 is None:
            return None
        if geom0 is None:
            return geom1.clone()
        if geom1 is None:
            return geom0.clone()

        return self.unionOptimized(geom0, geom1)

    def unionOptimized(self, geom0, geom1):
        env0 = geom0.envelope
        env1 = geom1.envelope

        if not env0.intersects(env1):
            return GeometryCombiner.combine(geom0, geom1)

        if geom0.numgeoms < 2 and geom1.numgeoms < 2:
            return self.unionActual(geom0, geom1)

        commonEnv = Envelope()
        env0.intersection(env1, commonEnv)

        return self.unionUsingEnvelopeIntersection(geom0, geom1, commonEnv)

    def unionUsingEnvelopeIntersection(self, geom0, geom1, env):
        """
         * Unions two geometries.
         * The case of multi geometries is optimized to union only
         * the components which lie in the intersection of the two geometry's
         * envelopes.
         * Geometrys outside this region can simply be combined with the union
         * result, which is potentially much faster.
         * This case is likely to occur often during cascaded union, and may also
         * occur in real world data (such as unioning data for parcels on
         * different street blocks).
         *
         * @param g0 a geometry
         * @param g1 a geometry
         * @param common the intersection of the envelopes of the inputs
         * @return the union of the inputs
        """
        disjointGeoms = []
        g0Int = self.extractByEnvelope(env, geom0, disjointGeoms)
        g1Int = self.extractByEnvelope(env, geom1, disjointGeoms)
        u = self.unionActual(g0Int, g1Int)
        disjointGeoms.append(u)

        return GeometryCombiner.combine(disjointGeoms)

    def _extractByEnvelope(self, env, geom, intersectingGeoms: list, disjointGeoms: list) -> None:
        for i in range(geom.numgeoms):
            g = geom.getGeometryN(i)
            if g.envelope.intersects(env):
                intersectingGeoms.append(g)
            else:
                disjointGeoms.append(g)

    def extractByEnvelope(self, env, geom, disjointGeoms: list):
        intersectingGeoms = []
        self._extractByEnvelope(env, geom, intersectingGeoms, disjointGeoms)
        return self._factory.buildGeometry(intersectingGeoms)

    def unionActual(self, geom0, geom1):
        """
         * Encapsulates the actual unioning of two polygonal geometries.
         *
         * @param g0
         * @param g1
         * @return
        """
        return geom0.union(geom1)


class CascadedPolygonUnion(CascadedUnion):
    """
     * Provides an efficient method of unioning a collection of
     * {@link Polygonal} geometries.
     * This algorithm is faster and likely more robust than
     * the simple iterated approach of
     * repeatedly unioning each polygon to a result geometry.
     *
     * The <tt>buffer(0)</tt> trick is sometimes faster, but can be less robust and
     * can sometimes take an exceptionally long time to complete.
     * This is particularly the case where there is a high degree of overlap
     * between the polygons.  In this case, <tt>buffer(0)</tt> is forced to compute
     * with <i>all</i> line segments from the outset,
     * whereas cascading can eliminate many segments
     * at each stage of processing.
     * The best case for buffer(0) is the trivial case
     * where there is <i>no</i> overlap between the input geometries.
     * However, this case is likely rare in practice.
    """

    """
     * The effectiveness of the index is somewhat sensitive
     * to the node capacity.
     * Testing indicates that a smaller capacity is better.
     * For an STRtree, 4 is probably a good number (since
     * this produces 2x2 "squares").
    """
    STRTREE_NODE_CAPACITY = 4

    def __init__(self, geoms):
        """
         * Creates a new instance to union
         * the given collection of {@link Geometry}s.
         *
         * @param geoms a collection of {@link Polygonal} {@link Geometry}s
         *        ownership of elements _and_ vector are left to caller.
        """
        CascadedUnion.__init__(self, geoms)

    def restrictToPolygons(self, geom):
        """
         * Computes a {@link Geometry} containing only {@link Polygonal} components.
         *
         * Extracts the {@link Polygon}s from the input
         * and returns them as an appropriate {@link Polygonal} geometry.
         *
         * If the input is already <tt>Polygonal</tt>, it is returned unchanged.
         *
         * A particular use case is to filter out non-polygonal components
         * returned from an overlay operation.
         *
         * @param g the geometry to filter
         * @return a Polygonal geometry
        """
        if geom.type_id == GeomTypeId.GEOS_POLYGON:
            return geom

        polys = []
        PolygonExtracter.getPolygons(geom, polys)
        if len(polys) == 1:
            return polys[0].clone()

        newPolys = [poly.clone() for poly in polys]
        return self._factory.createMultiPolygon(newPolys)

    @staticmethod
    def union(geoms):
        """
         * Computes the union of
         * a collection of {@link Polygonal} {@link Geometry}s.
         *
         * @param polys a collection of {@link Polygonal} {@link Geometry}s.
        """
        polys = []

        try:
            iter(geoms)
            polys = geoms
        except TypeError:
            pass

        try:
            if geoms.type_id == GeomTypeId.GEOS_MULTIPOLYGON:
                polys = [poly for poly in geoms.geoms]
        except:
            pass

        logger.debug("CascadedPolygonUnion.union() polygons:%s", len(polys))

        return CascadedPolygonUnion(polys)._union()

    def unionUsingEnvelopeIntersection(self, g0, g1, env):
        """
         * Unions two polygonal geometries, restricting computation
         * to the envelope intersection where possible.
         *
         * The case of MultiPolygons is optimized to union only
         * the polygons which lie in the intersection of the two geometry's
         * envelopes.
         * Polygons outside this region can simply be combined with the union
         * result, which is potentially much faster.
         * This case is likely to occur often during cascaded union, and may also
         * occur in real world data (such as unioning data for parcels on
         * different street blocks).
         *
         * @param g0 a polygonal geometry
         * @param g1 a polygonal geometry
         * @param env the intersection of the envelopes of the inputs
         * @return the union of the inputs
        """
        disjointPolys = []
        
        check_valid(g0, "unionUsingEnvelopeIntersection g0")
        check_valid(g1, "unionUsingEnvelopeIntersection g1")
        
        g0Int = self.extractByEnvelope(env, g0, disjointPolys)
        g1Int = self.extractByEnvelope(env, g1, disjointPolys)
        
        check_valid(g0Int, "unionUsingEnvelopeIntersection g0Int")
        check_valid(g1Int, "unionUsingEnvelopeIntersection g1Int")
        
        u = self.unionActual(g0Int, g1Int)

        check_valid(u, "unionUsingEnvelopeIntersection unionActual return")
        
        if len(disjointPolys) == 0:
            return u
        
        for i, poly in enumerate(disjointPolys):
            check_valid(poly, "disjoint poly {}".format(i))
        
        polysOn = []
        polysOff = []
        self._extractGeomListByEnvelope(u.envelope, disjointPolys, polysOn, polysOff)

        if len(polysOn) == 0:
            disjointPolys.append(u)
            ret = GeometryCombiner.combine(disjointPolys)
        else:
            ret = GeometryCombiner.combine(disjointPolys)
            ret = self.unionActual(ret, u)
        
        check_valid(ret, "unionUsingEnvelopeIntersection returned geom")
        
        return ret

    def _extractGeomListByEnvelope(self, env, geomList: list, intersectingPolys: list, disjointPolys: list) -> None:
        for g in geomList:
            if g.envelope.intersects(env):
                intersectingPolys.append(g)
            else:
                disjointPolys.append(g)

    def unionActual(self, g0, g1):
        """
         * Encapsulates the actual unioning of two polygonal geometries.
         *
         * @param g0
         * @param g1
         * @return
        """
        return self.restrictToPolygons(g0.union(g1))


class PointGeometryUnion():
    """
     * Computes the union of a {@link Puntal} geometry with
     * another arbitrary {@link Geometry}.
     *
     * Does not copy any component geometries.
     *
    """
    def __init__(self, pt, geom):
        self._factory = geom._factory
        self.pt = pt
        self.geom = geom

    @staticmethod
    def union(pt, geom):
        op = PointGeometryUnion(pt, geom)
        return op._union()

    def _union(self):
        locater = PointLocator()
        exteriorCoords = set()

        for i in range(self.pt.numgeoms):
            pt = self.pt.getGeometryN(i)
            coord = pt.coord
            loc = locater.locate(coord, self.geom)
            if loc == Location.EXTERIOR:
                exteriorCoords.insert(coord)

        if len(exteriorCoords) == 0:
            return self.geom.clone()

        ptComp = None
        if len(exteriorCoords) == 1:
            ptComp = self._factory.createPoint(exteriorCoords[0])
        else:
            coords = list(exteriorCoords)
            ptComp = self._factory.createMultiPoint(coords)

        return GeometryCombiner.combine(ptComp, self.geom)


class UnaryUnionOp():
    """
     * Unions a collection of Geometry or a single Geometry
     * (which may be a collection) together.
     * By using this special-purpose operation over a collection of
     * geometries it is possible to take advantage of various optimizations
     * to improve performance.
     * Heterogeneous {@link GeometryCollection}s are fully supported.
     *
     * The result obeys the following contract:
     *
     * - Unioning a set of overlapping {@link Polygons}s has the effect of
     *   merging the areas (i.e. the same effect as
     *   iteratively unioning all individual polygons together).
     * - Unioning a set of {@link LineString}s has the effect of
     *   <b>fully noding</b> and <b>dissolving</b> the input linework.
     *   In this context "fully noded" means that there will be a node or
     *   endpoint in the output for every endpoint or line segment crossing
     *   in the input.
     *   "Dissolved" means that any duplicate (e.g. coincident) line segments
     *   or portions of line segments will be reduced to a single line segment
     *   in the output.  *   This is consistent with the semantics of the
     *   {@link Geometry#union(Geometry)} operation.
     *   If <b>merged</b> linework is required, the {@link LineMerger} class
     *   can be used.
     * - Unioning a set of {@link Points}s has the effect of merging
     *   al identical points (producing a set with no duplicates).
     *
     * <tt>UnaryUnion</tt> always operates on the individual components of
     * MultiGeometries.
     * So it is possible to use it to "clean" invalid self-intersecting
     * MultiPolygons (although the polygon components must all still be
     * individually valid.)
    """
    def __init__(self, geoms, factory=None):

        self._factory = factory
        self.polygons = []
        self.lines = []
        self.points = []
        self.empty = None

        try:
            iter(geoms)
            self.extractGeoms(geoms)

        except:
            self.extract(geoms)
            pass

    @staticmethod
    def union(geoms, factory=None):
        op = UnaryUnionOp(geoms, factory)
        logger.debug("******************************\n")
        logger.debug("UnaryUnionOp.union()\n")
        logger.debug("******************************")
        return op._union()

    def _union(self):
        """
         * Gets the union of the input geometries.
         *
         * If no input geometries were provided, a POINT EMPTY is returned.
         *
         * @return a Geometry containing the union
         * @return an empty GEOMETRYCOLLECTION if no geometries were provided
         *         in the input
        """
        if self._factory is None:
            return None

        """
         * For points and lines, only a single union operation is
         * required, since the OGC model allowings self-intersecting
         * MultiPoint and MultiLineStrings.
         * This is not the case for polygons, so Cascaded Union is required.
        """
        unionPoints = None
        if len(self.points) > 0:
            logger.debug("UnaryUnionOp._union() points:%s", len(self.points))
            geom = self._factory.buildGeometry(self.points)
            unionPoints = self.unionNoOpt(geom)

        unionLines = None
        if len(self.lines) > 0:
            """
             * we use cascaded here for robustness [1]
             * but also add a final unionNoOpt step to deal with
             * self-intersecting lines [2]
            """
            logger.debug("UnaryUnionOp._union() lines:%s", len(self.lines))
            unionLines = CascadedUnion.union(self.lines)
            unionLines = self.unionNoOpt(unionLines)

        unionPolygons = None
        if len(self.polygons) > 0:
            logger.debug("UnaryUnionOp._union() polygons:%s", len(self.polygons))
            unionPolygons = CascadedPolygonUnion.union(self.polygons)

        """
         * Performing two unions is somewhat inefficient,
         * but is mitigated by unioning lines and points first
        """
        unionLA = self.unionWithNull(unionLines, unionPolygons)

        if unionPoints is None:
            ret = unionLA
        elif unionLA is None:
            ret = unionPoints
        else:
            ret = PointGeometryUnion.union(unionPoints, unionLA)

        if ret is None:
            logger.debug("UnaryUnionOp._union() result empty")
            return self._factory.createGeometryCollection()

        return ret

    def extractGeoms(self, geoms) -> None:
        for geom in geoms:
            self.extract(geom)

    def extract(self, geom) -> None:
        if self._factory is None:
            self._factory = geom._factory

        GeometryExtracter.extract(GeomTypeId.GEOS_POLYGON, geom, self.polygons)
        GeometryExtracter.extract(GeomTypeId.GEOS_LINESTRING, geom, self.lines)
        GeometryExtracter.extract(GeomTypeId.GEOS_POINT, geom, self.points)

    def unionNoOpt(self, geom):
        """
         * Computes a unary union with no extra optimization,
         * and no short-circuiting.
         * Due to the way the overlay operations
         * are implemented, this is still efficient in the case of linear
         * and puntal geometries.
         * Uses robust version of overlay operation
         * to ensure identical behaviour to the <tt>union(Geometry)</tt> operation.
         *
         * @param g0 a geometry
         * @return the union of the input geometry
        """
        if self.empty is not None:
            self.empty = self._factory.createEmptyGeometry()
        return BinaryOp(geom, self.empty, overlayOp(OverlayOp.opUNION))

    def unionWithNull(self, g0, g1):
        """
         * Computes the union of two geometries,
         * either of both of which may be null.
         *
         * @param g0 a Geometry (ownership transferred)
         * @param g1 a Geometry (ownership transferred)
         * @return the union of the input(s)
         * @return null if both inputs are null
        """
        if g0 is None and g1 is None:
            return None

        if g0 is None:
            return g1
        if g1 is None:
            return g0

        return g0.union(g1)
