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


from itertools import islice
from math import sqrt
from .shared import (
    GeomTypeId,
    Dimension,
    Envelope,
    PrecisionModel,
    Coordinate,
    CoordinateSequence,
    CoordinateSequenceFilter,
    CoordinateFilter,
    GeometryFilter,
    CoordinateOperation,
    GeometryEditor,
    GeometryComponentFilter,
    CAP_STYLE,
    JOIN_STYLE
    )
from .algorithms import (
    CGAlgorithms,
    ConvexHull
    )
from .op_valid import IsValidOp
from .op_simple import IsSimpleOp
from .op_linemerge import LineMerger
from .op_overlay import OverlayOp, overlayOp
from .op_binary import BinaryOp
from .op_union import UnaryUnionOp
from .op_relate import RelateOp
from .op_buffer import BufferOp
from .simplify import (
    TopologyPreservingSimplifier,
    DouglasPeukerSimplifier
)
from .affine import affine_transform
from .geomgraph import GeometryGraph


class Lineal():
    """
     * Identifies {Geometry} subclasses which
     * are 1-dimensional and with components which are {LineString}s.
    """


class Puntal():
    """
     * Identifies {Geometry} subclasses which
     * are 0-dimensional and with components which are {Point}s.
    """


class Polygonal():
    """
     * Identifies {Geometry} subclasses which
     * are 2-dimensional and with components which are {Polygon}s.
    """


class GeometryChangedFilter(GeometryComponentFilter):

    def filter_rw(self, geom):
        geom.geometryChangedAction()


class Geometry():
    """
     * Geometry
     *
     * Basic implementation of Geometry, constructed and
     * destructed by GeometryFactory.
     *
     *  clone returns a deep copy of the object.
     *  Use GeometryFactory to construct.
     *
     *  Binary Predicates
     * Because it is not clear at this time
     * what semantics for spatial
     *  analysis methods involving GeometryCollections would be useful,
     *  GeometryCollections are not supported as arguments to binary
     *  predicates (other than convexHull) or the relate
     *  method.
     *
     *  Set-Theoretic Methods
     *
     *  The spatial analysis methods will
     *  return the most specific class possible to represent the result. If the
     *  result is homogeneous, a Point, LineString, or
     *  Polygon will be returned if the result contains a single
     *  element otherwise, a MultiPoint, MultiLineString,
     *  or MultiPolygon will be returned. If the result is
     *  heterogeneous a GeometryCollection will be returned. <P>
     *
     *  Because it is not clear at this time what semantics for set-theoretic
     *  methods involving GeometryCollections would be useful,
     * GeometryCollections
     *  are not supported as arguments to the set-theoretic methods.
     *
     *  Representation of Computed Geometries
     *
     *  The SFS states that the result
     *  of a set-theoretic method is the "point-set" result of the usual
     *  set-theoretic definition of the operation (SFS 3.2.21.1). However, there are
     *  sometimes many ways of representing a point set as a Geometry.
     *  <P>
     *
     *  The SFS does not specify an unambiguous representation of a given point set
     *  returned from a spatial analysis method. One goal of JTS is to make this
     *  specification precise and unambiguous. JTS will use a canonical form for
     *  Geometrys returned from spatial analysis methods. The canonical
     *  form is a Geometry which is simple and noded:
     *  <UL>
     *    <LI> Simple means that the Geometry returned will be simple according to
     *    the JTS definition of is_simple.
     *    <LI> Noded applies only to overlays involving LineStrings. It
     *    means that all intersection points on LineStrings will be
     *    present as endpoints of LineStrings in the result.
     *  </UL>
     *  This definition implies that non-simple geometries which are arguments to
     *  spatial analysis methods must be subjected to a line-dissolve process to
     *  ensure that the results are simple.
     *
     *   Constructed Points And The Precision Model
     *
     *  The results computed by the set-theoretic methods may
     *  contain constructed points which are not present in the input Geometry.
     *  These new points arise from intersections between line segments in the
     *  edges of the input Geometry. In the general case it is not
     *  possible to represent constructed points exactly. This is due to the fact
     *  that the coordinates of an intersection point may contain twice as many bits
     *  of precision as the coordinates of the input line segments. In order to
     *  represent these constructed points explicitly, JTS must truncate them to fit
     *  the PrecisionModel.
     *
     *  Unfortunately, truncating coordinates moves them slightly. Line segments
     *  which would not be coincident in the exact result may become coincident in
     *  the truncated representation. This in turn leads to "topology collapses" --
     *  situations where a computed element has a lower dimension than it would in
     *  the exact result.
     *
     *  When JTS detects topology collapses during the computation of spatial
     *  analysis methods, it will throw an exception. If possible the exception will
     *  report the location of the collapse.
     *
     *  equals(Object) and hashCode are not overridden, so that when two
     *  topologically equal Geometries are added to HashMaps and HashSets, they
     *  remain distinct. This behaviour is desired in many cases.
     *
    """

    def __init__(self, newFactory):
        self._factory = newFactory
        self.geometryChangedFilter = GeometryChangedFilter()
        self.has_z = False

    @property
    def numpoints(self):
        """
         * Returns the count of this Geometrys vertices.
        """
        return 0

    @property
    def dimension(self):
        # Returns the dimension of this Geometry (0=point, 1=line, 2=surface)
        return Dimension.P

    @property
    def boundaryDimension(self):
        return Dimension.P

    @property
    def geom_type(self):
        """
         * Return a string representation of this Geometry type
        """
        return (
            'Point',
            'LineString',
            'LinearRing',
            'Polygon',
            'MultiPoint',
            'MultiLineString',
            'MultiLinearRing',
            'MultiPolygon',
            'GeometryCollection'
            )[self.type_id]

    @property
    def type_id(self):
        """
         * Return an integer representation of this Geometry type
        """
        return -1

    @property
    def numgeoms(self):
        """
         * Returns the number of geometries in this collection
         * (or 1 if this is not a collection)
        """
        return 1

    @property
    def envelope(self):
        """
         * Returns the minimum and maximum x and y values in this Geometry,
         * or a null Envelope if this Geometry is empty.
        """
        return self.computeEnvelope()

    @property
    def is_valid(self):
        """
         * Tests the validity of this <code>Geometry</code>.
         * Subclasses provide their own definition of "valid".
         * @return <code>true</code> if this <code>Geometry</code> is valid
         * @see IsValidOp
        """
        return IsValidOp(self).is_valid()

    @property
    def is_simple(self):
        """
         * Returns false if the Geometry not simple.
        """
        if self.type_id == GeomTypeId.GEOS_GEOMETRYCOLLECTION:
            raise ValueError("This method does not support GeometryCollection")

        return IsSimpleOp(self).is_simple()

    @property
    def is_empty(self):
        return False

    @property
    def is_rectangle(self):
        return False

    @property
    def is_ccw(self):
        return False

    def getEnvelope(self):
        """
         * Returns envelope as geometry
        """
        return self._factory.toGeometry(self.envelope)

    @property
    def area(self):
        """
         *  Returns the area of this Geometry.
         *  Areal Geometries have a non-zero area.
         *  They override this function to compute the area.
         *  Others return 0.0
         *
         * @return the area of the Geometry
        """
        return 0.0

    @property
    def length(self):
        """
         *  Returns the length of this Geometry.
         *  Linear geometries return their length.
         *  Areal geometries return their perimeter.
         *  They override this function to compute the area.
         *  Others return 0.0
         *
         * @return the length of the Geometry
        """
        return 0.0

    @property
    def precisionModel(self):
        return self._factory.precisionModel

    def compareTo(self, geom):

        if geom is self:
            return 0

        classSortIndex = self.classSortIndex - geom.classSortIndex
        if classSortIndex != 0:
            return classSortIndex

        if self.is_empty and geom.is_empty:
            return 0

        if self.is_empty:
            return -1

        if geom.is_empty:
            return 1

        return self.compareToSameClass(geom)

    def compareToSameClass(self, geom):
        raise NotImplementedError()

    def isEquivalentClass(self, other) -> bool:
        return self.type_id == other.type_id

    def geometryChanged(self):
        self.apply_rw(self.geometryChangedFilter)

    def geometryChangedAction(self):
        self._env = None

    @property
    def classSortIndex(self):
        cls = type(self).__name__
        try:
            index = {
                'Point': 0,
                'MultiPoint': 1,
                'LineString': 2,
                'LinearRing': 3,
                'MultiLineString': 4,
                'Polygon': 5,
                'MultiPolygon': 6,
                'GeometryCollection': 7
                }[cls]
        except:
            index = -1
            raise ValueError("Class not supported: {}".format(cls))

        return index

    def hasNonEmptyElements(self, geoms):
        for geom in geoms:
            if not geom.is_empty:
                return True
        return False

    def hasNullElements(self, geoms):
        for geom in geoms:
            if geom is None:
                return True
        return False

    def getGeometryN(self, index):
        return self

    # relationships (relate)
    def disjoint(self, other) -> bool:
        if not self.envelope.intersects(other.envelope):
            return True

        im = self.relate(other)
        return im.isDisjoint

    def touches(self, other) -> bool:
        if not self.envelope.intersects(other.envelope):
            return False
        im = self.relate(other)
        return im.isTouches(self.dimension, other.dimension)

    def intersects(self, other) -> bool:
        if not self.envelope.intersects(other.envelope):
            return False
        # @TODO: optimization for rectangles
        im = self.relate(other)
        return im.isIntersects

    def covers(self, other) -> bool:
        if not self.envelope.intersects(other.envelope):
            return False
        im = self.relate(other)
        return im.isCovers

    def crosses(self, other) -> bool:
        if not self.envelope.intersects(other.envelope):
            return False
        im = self.relate(other)
        return im.isCrosses(self.dimension, other.dimension)

    def within(self, other) -> bool:
        return other.contains(self)

    def contains(self, other) -> bool:
        if not self.envelope.intersects(other.envelope):
            return False
        im = self.relate(other)
        return im.isContains

    def overlaps(self, other) -> bool:
        if not self.envelope.intersects(other.envelope):
            return False

        im = self.relate(other)
        return im.isOverlaps(self.dimension, other.dimension)

    def equals(self, other) -> bool:
        if not self.envelope.intersects(other.envelope):
            return False
        if self.is_empty:
            return other.is_empty
        elif other.is_empty:
            return self.is_empty
        im = self.relate(other)
        return im.isEquals(self.dimension, other.dimension)

    def relate(self, other, intersectionPattern: str=None):
        if intersectionPattern is None:
            # IntersectionMatrix
            return RelateOp.relate(self, other)
        else:
            im = self.relate(other)
            return im.matches(intersectionPattern)

    # Convex hull
    @property
    def convex_hull(self):
        return ConvexHull(self).getConvexHull()

    def computeMinimumRotatedRectangle(self):
        """
         * return transformed rectangle,
         * untransformed rectangle and inverse matrix
        """
        hull = self.convex_hull
        try:
            coords = hull.exterior.coords
        except AttributeError:
            # may be a Point or a LineString
            coords = hull.coords
            pass

        if len(coords) < 2:
            return None, None, None

        # generate the edge vectors between the convex hull's coords
        edges = ((pt2.x - pt1.x, pt2.y - pt1.y) for pt1, pt2 in zip(
            coords, islice(coords, 1, None)))

        def _transformed_rects():
            for dx, dy in edges:
                # compute the normalized direction vector of the edge
                # vector.
                length = sqrt(dx ** 2 + dy ** 2)
                ux, uy = dx / length, dy / length
                # compute the normalized perpendicular vector
                vx, vy = -uy, ux
                # transform hull from the original coordinate system to
                # the coordinate system defined by the edge and compute
                # the axes-parallel bounding rectangle.
                transf_rect = affine_transform(
                    hull, (ux, uy, vx, vy, 0, 0)).getEnvelope()
                # yield the transformed rectangle and a matrix to
                # transform it back to the original coordinate system.
                yield (transf_rect, (ux, vx, uy, vy, 0, 0))

        # check for the minimum area rectangle and return it
        transf_rect, inv_matrix = min(
            _transformed_rects(), key=lambda r: r[0].area)

        return affine_transform(transf_rect, inv_matrix), transf_rect, inv_matrix

    @property
    def minimum_rotated_rectangle(self):
        rect, transf_rect, inv_matrix = self.computeMinimumRotatedRectangle()
        return rect

    def checkNotGeometryCollection(self, geom):
        if geom.type_id == GeomTypeId.GEOS_GEOMETRYCOLLECTION:
            raise ValueError("This method does not support heterogeneous geometry collection")

    # Boolean operations (overlay)
    def intersection(self, other):
        # special case: if one input is empty ==> other input
        if self.is_empty or other.is_empty:
            return self._factory.createGeometryCollection()

        if (self.type_id == GeomTypeId.GEOS_GEOMETRYCOLLECTION or
                self.type_id == GeomTypeId.GEOS_MULTIPOLYGON or
                self.type_id == GeomTypeId.GEOS_MULTILINESTRING or
                self.type_id == GeomTypeId.GEOS_MULTIPOINT):
            geoms = [geom.intersection(other) for geom in self.geoms]
            return self._factory.buildGeometry(geoms)

        # @TODO: Use rectangle intersection optimization
        self.checkNotGeometryCollection(self)
        self.checkNotGeometryCollection(other)

        return BinaryOp(self, other, overlayOp(OverlayOp.opINTERSECTION))

    def union(self, other=None):

        if other is None:
            return UnaryUnionOp.union(self)

        # special case: if one input is empty ==> other input
        if self.is_empty:
            return other.clone()

        if other.is_empty:
            return self.clone()

        # if envelopes are disjoint return a MULTI geom or
        # a geometrycollection
        if not self.envelope.intersects(other.envelope):

            v = []
            if issubclass(type(self), GeometryCollection):
                for geom in self.geoms:
                    v.append(geom.clone())
            else:
                v.append(self.clone())

            if issubclass(type(other), GeometryCollection):
                for geom in other.geoms:
                    v.append(geom.clone())
            else:
                v.append(other.clone())

            return self._factory.buildGeometry(v)

        self.checkNotGeometryCollection(self)
        self.checkNotGeometryCollection(other)

        return BinaryOp(self, other, overlayOp(OverlayOp.opUNION))

    def difference(self, other):
        # special case: if A.is_empty ==> empty; if B.is_empty ==> A
        if self.is_empty:
            return self._factory.createGeometryCollection()

        if other.is_empty:
            return self.clone()

        self.checkNotGeometryCollection(self)
        self.checkNotGeometryCollection(other)

        return BinaryOp(self, other, overlayOp(OverlayOp.opDIFFERENCE))

    def symmetric_difference(self, other):
        # special case: if one input is empty ==> other input
        if self.is_empty:
            return other.clone()

        if other.is_empty:
            return self.clone()

        # if envelopes are disjoint return a MULTI geom or
        # a geometrycollection
        if not self.envelope.intersects(other.envelope):
            v = []
            if issubclass(type(self), GeometryCollection):
                for geom in self.geoms:
                    v.append(geom.clone())
            else:
                v.append(self.clone())

            if issubclass(type(other), GeometryCollection):
                for geom in other.geoms:
                    v.append(geom.clone())
            else:
                v.append(other.clone())

            return self._factory.buildGeometry(v)

        self.checkNotGeometryCollection(self)
        self.checkNotGeometryCollection(other)

        return BinaryOp(self, other, overlayOp(OverlayOp.opSYMDIFFERENCE))

    # buffer
    def buffer(self,
            distance: float=0,
            resolution: int=12,
            cap_style: int=CAP_STYLE.round,
            join_style: int=JOIN_STYLE.round,
            mitre_limit: float=5.0,
            single_sided: bool=False):

        if self.is_empty:
            return

        return BufferOp.bufferOp(self,
                distance=distance,
                quadrantSegments=resolution,
                endCapStyle=cap_style,
                joinStyle=join_style,
                mitreLimit=mitre_limit,
                singleSided=single_sided)

    # line merge
    def line_merge(self):
        return LineMerger.merge(self)

    # simplify
    def simplify(self, tolerance: float, preserve_topology: bool=True):
        if preserve_topology:
            return TopologyPreservingSimplifier.simplify(self, tolerance)
        else:
            return DouglasPeukerSimplifier.simplify(self, tolerance)

    # Filters
    def apply_ro(self, filter) -> None:
        filter.filter_ro(self)

    def apply_rw(self, filter) -> None:
        filter.filter_rw(self)

    def applyComponentFilter(self, filter):
        """
         * Apply a fiter to each component of this geometry.
         * The filter is expected to provide a .filter(const Geometry*)
         * method.
        """
        for i in range(self.numgeoms):
            filter.filter(self.getGeometryN(i))

    def __geo_interface__(self):
        """
        Returns the geojson dictionary representation of the geometry.
        """
        geo = dict()
        coords = []
        # @TODO: apply a component filter here to get coords
        geo["type"] = self.geom_type
        geo["coordinates"] = coords
        return geo


class Point(Geometry, Puntal):
    """
     * Implementation of Point.
     *
     * A Point is valid iff:
     *
     * - the coordinate which defines it is a valid coordinate
     *   (i.e does not have an NaN X or Y ordinate)
     *
    """
    def __init__(self, coord, factory):
        Geometry.__init__(self, factory)
        self._env = None
        self.coord = coord

    def computeEnvelope(self):
        if self._env is None:
            x = self.coord.x
            y = self.coord.y
            self._env = Envelope(x, y, x, y)
        return self._env

    @property
    def type_id(self):
        """
         * Return an integer representation of this Geometry type
        """
        return GeomTypeId.GEOS_POINT

    def clone(self):
        return Point(self.coord.clone(), self._factory)

    @property
    def coords(self):
        return self._factory.coordinateSequenceFactory.create([self.coord])

    def almost_equals(self, other, tolerance):
        if self.type_id != other.type_id:
            return False
        return self.coord.distance(other.coord) < tolerance

    def apply_ro(self, filter) -> None:

        if issubclass(type(filter), CoordinateFilter):
            self.coords.apply_ro(filter)

        elif issubclass(type(filter), CoordinateSequenceFilter):
            filter.filter_ro(self.coords, 0)
        else:
            filter.filter_ro(self)

    def apply_rw(self, filter) -> None:

        if issubclass(type(filter), CoordinateFilter):
            self.coords.apply_rw(filter)

        elif issubclass(type(filter), CoordinateSequenceFilter):
            filter.filter_rw(self.coords, 0)
            if filter.isGeometryChanged:
                self.geometryChanged()
        else:
            filter.filter_rw(self)


class LineString(Geometry, Lineal):
    """
     *  Models an OGC-style LineString.
     *
     *  A LineString consists of a sequence of two or more vertices,
     *  along with all points along the linearly-interpolated curves
     *  (line segments) between each
     *  pair of consecutive vertices.
     *  Consecutive vertices may be equal.
     *  The line segments in the line may intersect each other (in other words,
     *  the linestring may "curl back" in itself and self-intersect.
     *  Linestrings with exactly two identical points are invalid.
     *
     *  A linestring must have either 0 or 2 or more points.
     *  If these conditions are not met, the constructors throw
     *  an ValueError
    """
    def __init__(self, coords, newFactory):

        Geometry.__init__(self, newFactory)
        if coords is None:
            coords = []

        # Envelope internal cache
        self._env = None
        self._coords = CoordinateSequence.removeRepeatedPoints(coords)

        self.validateConstruction()

    def validateConstruction(self):
        if self.numpoints == 1:
            raise ValueError("point array must contain 0 or >1 elements")

    def computeEnvelope(self):

        if self._env is None:
            x = [c.x for c in self._coords]
            y = [c.y for c in self._coords]
            xmin = min(x)
            ymin = min(y)
            xmax = max(x)
            ymax = max(y)
            self._env = Envelope(xmin, ymin, xmax, ymax)

        return self._env

    def compareToSameClass(self, geom):

        mynpts = self.numpoints
        othpts = geom.numpoints

        if mynpts > othpts:
            return 1
        if mynpts < othpts:
            return -1
        _coords = geom._coords
        for i, coord in enumerate(self._coords):
            cmp = coord.compareTo(_coords[i])
            if bool(cmp):
                return cmp
        return 0

    @property
    def coord(self):
        if self.is_empty:
            return None
        return self._coords[0]

    @property
    def coords(self):
        return self._coords

    @property
    def dimension(self):
        return Dimension.L

    @property
    def boundaryDimension(self):
        if self.isClosed:
            return Dimension.FALSE
        return Dimension.P

    @coords.setter
    def coords(self, coords):
        """
         * reset envelope on coords set
        """
        self._env = None
        self._coords = coords

    @property
    def coordsRO(self):
        """
         * Returns immutable coords
        """
        return tuple(self._coords)

    @property
    def numpoints(self):
        return len(self._coords)

    @property
    def boundary(self):
        """
         * return Geometric boundary as LineStrings
        """
        # GeometryFactory
        gf = self._factory

        # using the default OGC_SFS MOD2 rule, the boundary of a
        # closed LineString is empty
        if self.isClosed:
            return gf.createMultiPoint()

        pts = []
        pts.append(self._coords[0].clone())
        pts.append(self._coords[-1].clone())

        return gf.createMultiPoint(pts)

    @property
    def is_empty(self):
        return self.numpoints < 1

    @property
    def is_ring(self):
        return self.isClosed and self.is_simple

    @property
    def is_ccw(self):
        return CGAlgorithms.isCCW(self.coords)

    @property
    def isClosed(self):
        return (not self.is_empty) and self._coords[0] == self._coords[-1]

    @property
    def length(self):
        return CGAlgorithms.length(self._coords)

    def clone(self):
        return LineString(self._coords.clone(), self._factory)

    def reverse(self):
        return self._factory.createLineSting(reversed(self.coords.clone()))

    # Buffer apply to linestring only
    def parallel_offset(self, distance: float, resolution: int, join_style: int, mitre_limit: int):

        if self.is_empty:
            return

        return BufferOp.offsetCurveOp(self, distance, resolution, join_style, mitre_limit)

    @property
    def type_id(self):
        """
         * Return an integer representation of this Geometry type
        """
        return GeomTypeId.GEOS_LINESTRING

    def apply_ro(self, filter) -> None:

        if issubclass(type(filter), CoordinateFilter):
            self.coords.apply_ro(filter)

        elif issubclass(type(filter), CoordinateSequenceFilter):
            if len(self.coords) == 0:
                return
            for i, coord in enumerate(self.coords):
                filter.filter_ro(self.coords, i)
                if filter.isDone:
                    break

        else:
            filter.filter_ro(self)

    def apply_rw(self, filter) -> None:

        if issubclass(type(filter), CoordinateFilter):
            self.coords.apply_rw(filter)

        elif issubclass(type(filter), CoordinateSequenceFilter):
            if len(self.coords) == 0:
                return
            for i, coord in enumerate(self.coords):
                filter.filter_rw(self.coords, i)
                if filter.isDone:
                    break
                if filter.isGeometryChanged:
                    self.geometryChanged()

        else:
            filter.filter_rw(self)

    def __str__(self):
        return "{} {}".format(type(self).__name__, self.coords)

    def almost_equals(self, other, tolerance):
        if self.type_id != other.type_id:
            return False
        if tolerance == 0:
            return self.coords == other.coords
        else:
            return self.coords.almost_equals(other.coords, tolerance)


class LinearRing(LineString):
    """
     * Models an OGC SFS LinearRing.
     *
     * A LinearRing is a LineString which is both closed and simple.
     * In other words,
     * the first and last coordinate in the ring must be equal,
     * and the interiors of the ring must not self-intersect.
     * Either orientation of the ring is allowed.
     *
     * A ring must have either 0 or 4 or more points.
     * The first and last points must be equal (in 2D).
     * If these conditions are not met, the constructors throw
     * an ValueError
    """
    def __init__(self, coords, newFactory):
        if coords is None:
            coords = []
        coords = CoordinateSequence._removeRepeatedPoints(coords)
        LineString.__init__(self, coords, newFactory)

    def clone(self):
        return LinearRing(self._coords.clone(), self._factory)

    def validateConstruction(self):
        if 0 < self.numpoints < 4:
            raise ValueError("point array must contain 0 or >3 elements")
        elif self.numpoints > 0 and self._coords[0] != self._coords[-1]:
            # self._factory.output(self, name="not closed")
            raise ValueError("first and last points must be equal")

    @property
    def boundaryDimension(self):
        return Dimension.FALSE

    @property
    def isClosed(self):
        return self.is_empty or self._coords[0] == self._coords[-1]

    @property
    def is_simple(self):
        # linearRings are simple by definition
        return True

    def reverse(self):
        return self._factory.createLinearRing(reversed(self.coords))

    @property
    def type_id(self):
        """
         * Return an integer representation of this Geometry type
        """
        return GeomTypeId.GEOS_LINEARRING


class Polygon(Geometry, Polygonal):
    """
     * Polygon
     *
     * Represents a linear polygon, which may include interiors.
     *
     * The exterior and interiors of the polygon are represented by {LinearRing}s.
     * In a valid polygon, interiors may touch the exterior or other interiors at a single point.
     * However, no sequence of touching interiors may split the polygon into two pieces.
     * The orientation of the rings in the polygon does not matter.
     * <p>
     *  The exterior and interiors must conform to the assertions specified in the <A
     *  HREF="http://www.opengis.org/techno/specs.htm">OpenGIS Simple Features
     *  Specification for SQL</A> .
    """
    def __init__(self, exterior, interiors, factory):

        Geometry.__init__(self, factory)

        self._env = None
        self._area = None
        self._exteriorArea = None
        
        if exterior is None:
            self.exterior = self._factory.createLinearRing(None)
        else:
            if interiors is not None and exterior.is_empty and self.hasNonEmptyElements(interiors):
                raise ValueError("exterior is empty but interiors are not")

            self.exterior = exterior

        if interiors is None:
            self.interiors = []
        else:
            if self.hasNullElements(interiors):
                raise ValueError("interiors must not contain null elements")
            for hole in interiors:
                if hole.type_id != GeomTypeId.GEOS_LINEARRING:
                    raise ValueError("interiors must be LinearRings")
            self.interiors = interiors

    def clone(self):
        return Polygon(self.exterior.clone(), [hole.clone() for hole in self.interiors], self._factory)

    def computeEnvelope(self):

        if self._env is None:
            self._env = self.exterior.envelope
        return self._env

    def compareToSameClass(self, geom):
        return self.exterior.compareToSameClass(geom.exterior)

    @property
    def dimension(self):
        return Dimension.A

    @property
    def boundaryDimension(self):
        return Dimension.L

    @property
    def coords(self):
        if self.is_empty:
            return self._factory.coordinateSequenceFactory.create()
        coords = list(self.exterior.coords)
        for hole in self.interiors:
            coords.extend(hole.coords)
        return self._factory.coordinateSequenceFactory.create(coords)

    @property
    def coord(self):
        if self.is_empty:
            return None
        return self.exterior.coord

    @property
    def numpoints(self):
        numpoints = self.exterior.numpoints
        for hole in self.interiors:
            numpoints += hole.numpoints
        return numpoints

    @property
    def is_empty(self):
        return self.exterior.is_empty

    @property
    def boundary(self):
        """
         * return Geometric boundary as LineStrings
        """
        # GeometryFactory
        gf = self._factory

        if self.is_empty:
            return gf.createMultiLineString()

        if len(self.interiors) == 0:
            return gf.createLineString(self.exterior.coords.clone())

        rings = [gf.createLineString(self.exterior.coords.clone())] + [gf.createLineString(hole.coords.clone())
                    for hole in self.interiors]

        return gf.createMultiLineString(rings)

    @property
    def area(self):
        if self._area is None:
            self._area = abs(CGAlgorithms.signedArea(self.exterior.coords))
            for hole in self.interiors:
                self._area -= abs(CGAlgorithms.signedArea(hole.coords))
        return self._area

    @property
    def exterior_area(self):
        if self._exteriorArea is None:
            self._exteriorArea = abs(CGAlgorithms.signedArea(self.exterior.coords))
        return self._exteriorArea
        
    @property
    def length(self):
        length = self.exterior.length
        for hole in self.interiors:
            length += hole.length
        return length
    
    @property
    def is_rectangle(self):
        if len(self.interiors) > 0:
            return False
        
        coords = self.exterior.coords
            
        if len(coords) != 5:
            return False
        
        # check vertices have correct values
        env = self.envelope
        for i in range(5):
            x = coords[i].x
            if (not (x == env.minx or x == env.maxx)):
                return False
            y = coords[i].y
            if (not (y == env.miny or y == env.maxy)):
                return False
        
        # check vertices are in right order
        prevX = coords[0].x
        prevY = coords[0].y
        for i in range(1, 5):
            x = coords[i].x
            y = coords[i].y
            xChanged = (x != prevX)
            yChanged = (y != prevY)
            if (xChanged == yChanged):
                return False
            prevX = x
            prevY = y
        return True
    
    def normalize(self):
        self._normalize(self.exterior, True)
        for hole in self.interiors:
            self._normalize(hole, False)

    def _normalize(self, ring, clockwise):

        if ring.is_empty:
            return

        uniqueCoords = list(ring.coords)
        uniqueCoords.pop()
        minCoord = CoordinateSequence.minCoordinate(uniqueCoords)

        CoordinateSequence.scroll(uniqueCoords, minCoord)
        uniqueCoords.append(uniqueCoords[0])

        if CGAlgorithms.isCCW(uniqueCoords) == clockwise:
            CoordinateSequence.reverse(uniqueCoords)

        ring.coords = uniqueCoords

    def geometryChangedAction(self):
        self._env = None    
        self._area = None
        self._exteriorArea = None
        
    @property
    def type_id(self):
        """
         * Return an integer representation of this Geometry type
        """
        return GeomTypeId.GEOS_POLYGON

    def apply_ro(self, filter) -> None:
        if issubclass(type(filter), GeometryFilter):
            filter.filter_ro(self)

        elif issubclass(type(filter), GeometryComponentFilter):
            filter.filter_ro(self)
            self.exterior.apply_ro(filter)
            for hole in self.interiors:
                hole.apply_ro(filter)

        elif issubclass(type(filter), CoordinateFilter):
            self.exterior.apply_ro(filter)
            for hole in self.interiors:
                hole.apply_ro(filter)

        elif issubclass(type(filter), CoordinateSequenceFilter):
            self.exterior.apply_ro(filter)
            if not filter.isDone:
                for hole in self.interiors:
                    hole.apply_ro(filter)

        else:
            raise ValueError("Unknown filter type {}".format(type(filter).__name__))

    def apply_rw(self, filter) -> None:
        if issubclass(type(filter), GeometryFilter):
            filter.filter_rw(self)

        elif issubclass(type(filter), GeometryComponentFilter):
            filter.filter_rw(self)
            self.exterior.apply_rw(filter)
            for hole in self.interiors:
                hole.apply_rw(filter)

        elif issubclass(type(filter), CoordinateFilter):
            self.exterior.apply_rw(filter)
            for hole in self.interiors:
                hole.apply_rw(filter)

        elif issubclass(type(filter), CoordinateSequenceFilter):
            self.exterior.apply_rw(filter)
            if not filter.isDone:
                for hole in self.interiors:
                    hole.apply_rw(filter)

        else:
            raise ValueError("Unknown filter type {}".format(type(filter).__name__))

    @property
    def convex_hull(self):
        return self.exterior.convex_hull

    def almost_equals(self, other, tolerance=0):
        if other is None:
            return False
        if not self.exterior.almost_equals(other.exterior, tolerance):
            return False
        if len(self.interiors) != len(other.interiors):
            return False
        for i, hole in enumerate(self.interiors):
            if not hole.almost_equals(other.interiors[i], tolerance):
                return False
        return True


class GeometryCollection(Geometry):
    """
     * GeometryCollection
     *
     * Represents a collection of heterogeneous Geometry objects.
     *
     * Collections of Geometry of the same type are
     * represented by GeometryCollection subclasses MultiPoint,
     * MultiLineString, MultiPolygon.
    """
    def __init__(self, geoms=[], factory=None):
        Geometry.__init__(self, factory)
        if geoms is None:
            geoms = []
        self._env = None
        self.geoms = geoms

    def clone(self):
        return GeometryCollection([geom.clone() for geom in self.geoms], self._factory)

    @property
    def numgeoms(self):
        return len(self.geoms)

    @property
    def dimension(self):
        """
         * Returns the maximum dimension of geometries in this collection
         * (0=point, 1=line, 2=surface)
         *
        """
        dim = 0
        for geom in self.geoms:
            if geom.dimension > dim:
                dim = geom.dimension
        return dim

    @property
    def type_id(self):
        """
         * Return an integer representation of this Geometry type
        """
        return GeomTypeId.GEOS_GEOMETRYCOLLECTION

    def computeEnvelope(self):
        if self._env is None:
            self._env = Envelope()
            for geom in self.geoms:
                self._env.expandToInclude(geom.envelope)
        return self._env

    def getGeometryN(self, index):
        return self.geoms[index]

    def apply_ro(self, filter) -> None:
        if issubclass(type(filter), GeometryFilter):
            filter.filter_ro(self)
            for geom in self.geoms:
                geom.apply_ro(filter)

        elif issubclass(type(filter), GeometryComponentFilter):
            filter.filter_ro(self)
            for geom in self.geoms:
                geom.apply_ro(filter)

        elif issubclass(type(filter), CoordinateFilter):
            for geom in self.geoms:
                geom.apply_ro(filter)

        elif issubclass(type(filter), CoordinateSequenceFilter):

            for geom in self.geoms:
                geom.apply_ro(filter)
                if filter.isDone:
                    break

        else:
            raise ValueError("Unknown filter type {}".format(type(filter).__name__))

    def apply_rw(self, filter) -> None:
        if issubclass(type(filter), GeometryFilter):
            filter.filter_rw(self)
            for geom in self.geoms:
                geom.apply_rw(filter)

        elif issubclass(type(filter), GeometryComponentFilter):
            filter.filter_rw(self)
            for geom in self.geoms:
                geom.apply_rw(filter)

        elif issubclass(type(filter), CoordinateFilter):
            for geom in self.geoms:
                geom.apply_rw(filter)

        elif issubclass(type(filter), CoordinateSequenceFilter):

            for geom in self.geoms:
                geom.apply_rw(filter)
                if filter.isDone:
                    break
            if filter.isGeometryChanged:
                self.geometryChanged()

        else:
            raise ValueError("Unknown filter type {}".format(type(filter).__name__))

    def almost_equals(self, other, tolerance: float=0) -> bool:
        if len(self.geoms) != len(other.geoms):
            return False
        if self.type_id != other.type_id:
            return False
        for i, geom in enumerate(self.geoms):
            if not geom.almost_equals(other.geoms[i], tolerance):
                return False
        return True


class MultiLineString(GeometryCollection, Lineal):
    """
    """
    def __init__(self, geoms, factory):
        GeometryCollection.__init__(self, geoms, factory)

    def clone(self):
        return MultiLineString([geom.clone() for geom in self.geoms], self._factory)

    @property
    def dimension(self):
        """
         * Returns line dimension (1)
        """
        return Dimension.L

    @property
    def boundaryDimension(self):
        for geom in self.geoms:
            if not geom.isClosed:
                return Dimension.P
        return Dimension.FALSE

    @property
    def type_id(self):
        """
         * Return an integer representation of this Geometry type
        """
        return GeomTypeId.GEOS_MULTILINESTRING

    @property
    def boundary(self):
        if self.is_empty:
            return self._factory.createGeometryCollection(None)
        gg = GeometryGraph(0, self)
        coords = gg.boundaryPoints
        return self._factory.createMultiPoint(coords)


class MultiPoint(GeometryCollection, Puntal):
    """
    """
    def __init__(self, geoms, factory):
        GeometryCollection.__init__(self, geoms, factory)

    def clone(self):
        return MultiPoint([geom.clone() for geom in self.geoms], self._factory)

    @property
    def dimension(self):
        """
         * Returns point dimension (0)
        """
        return Dimension.P

    @property
    def boundaryDimension(self):
        return Dimension.FALSE

    @property
    def type_id(self):
        """
         * Return an integer representation of this Geometry type
        """
        return GeomTypeId.GEOS_MULTIPOINT


class MultiPolygon(GeometryCollection, Polygonal):
    """
    """
    def __init__(self, geoms, factory):
        GeometryCollection.__init__(self, geoms, factory)

    def clone(self):
        return MultiPolygon([geom.clone() for geom in self.geoms], self._factory)

    @property
    def dimension(self):
        """
         * Returns area dimension (2)
        """
        return Dimension.A

    @property
    def boundaryDimension(self):
        return Dimension.L

    @property
    def type_id(self):
        """
         * Return an integer representation of this Geometry type
        """
        return GeomTypeId.GEOS_MULTIPOLYGON

    @property
    def boundary(self):
        if self.is_empty:
            return self._factory.createMultiLineString()
        rings = []
        for geom in self.geoms:
            if geom.type_id == GeomTypeId.GEOS_POLYGON:
                rings.append(geom.boundary)
            elif geom.type_id == GeomTypeId.GEOS_MULTIPOLYGON:
                rings.extend([g.boundary for g in geom.geoms])

        return self._factory.createMultiLineString(rings)


class CoordinateSequenceFactory():
    """
     * A factory to create concrete instances of {CoordinateSequence}s.
     *
     * Used to configure {GeometryFactory}s
     * to provide specific kinds of CoordinateSequences.
    """
    def create(self, coords=None, allowRepeated: bool=True, direction: bool=True):
        return CoordinateSequence(coords, allowRepeated, direction)


class gfCoordinateOperation(CoordinateOperation):

    def __init__(self, gsf):
        self.gsf = gsf

    def _edit(self, coords, geom):
        return self.gsf.create(coords)


class GeometryFactory():
    """
     * Supplies a set of utility methods for building Geometry objects
     * from CoordinateSequence or other Geometry objects.
     *
     * Note that the factory constructor methods do <b>not</b> change the input
     * coordinates in any way.
     * In particular, they are not rounded to the supplied <tt>PrecisionModel</tt>.
     * It is assumed that input Coordinates meet the given precision.
    """
    def __init__(self, coordinateSequenceFactory=None, precisionModel=None, SRID: int=0, outputFactory=None):

        if coordinateSequenceFactory is None:
            coordinateSequenceFactory = CoordinateSequenceFactory()
        self.coordinateSequenceFactory = coordinateSequenceFactory

        if precisionModel is None:
            precisionModel = PrecisionModel()
        self.precisionModel = precisionModel

        self.outputFactory = outputFactory

        self.SRID = SRID

    @staticmethod
    def create(precisionModel=None):
        return GeometryFactory(precisionModel=precisionModel)

    def clone(self, precisionModel=None):
        if precisionModel is None:
            precisionModel = self.precisionModel
        return GeometryFactory(coordinateSequenceFactory=self.coordinateSequenceFactory,
            SRID=self.SRID,
            outputFactory=self.outputFactory,
            precisionModel=precisionModel)

    def createGeometryCollection(self, newGeoms=[]):
        return GeometryCollection(newGeoms, self)

    def createMultiLineString(self, fromLines):
        return MultiLineString(fromLines, self)

    def createMultiPolygon(self, newPolys):
        return MultiPolygon(newPolys, self)

    def createMultiPoint(self, newPoints):
        return MultiPoint(newPoints, self)

    def createPoint(self, coord):
        return Point(coord, self)

    def createLinearRing(self, fromCoords=None):
        return LinearRing(fromCoords, self)

    def createLineString(self, fromCoords=None):
        ls = LineString(fromCoords, self)
        if len(ls.coords) > 1:
            return ls
        return None

    def createPolygon(self, exterior=None, interiors=None):
        return Polygon(exterior, interiors, self)

    def createGeometry(self, geom):
        editor = GeometryEditor(self)
        coordOp = gfCoordinateOperation(self.coordinateSequenceFactory)
        return editor.edit(geom, coordOp)

    def buildGeometry(self, newGeoms):

        isHeterogeneous = False
        hasGeometryCollection = False

        geomClass = None

        for geom in newGeoms:
            type_id = geom.type_id

            if geomClass is None:
                geomClass = type_id
            elif geomClass != type_id:
                isHeterogeneous = True

            if type_id == GeomTypeId.GEOS_GEOMETRYCOLLECTION:
                hasGeometryCollection = True

        # for the empty geometry, return an empty GeometryCollection
        if geomClass is None:
            return self.createGeometryCollection()

        if isHeterogeneous or hasGeometryCollection:
            return self.createGeometryCollection(newGeoms)

        # At this point we know the collection is not hetereogenous.
        # Determine the type of the result from the first Geometry in the
        # list. This should always return a geometry, since otherwise
        # an empty collection would have already been returned
        isCollection = len(newGeoms) > 1

        if isCollection:
            if geomClass == GeomTypeId.GEOS_POLYGON:
                return self.createMultiPolygon(newGeoms)

            elif geomClass == GeomTypeId.GEOS_LINESTRING:
                return self.createMultiLineString(newGeoms)

            elif geomClass == GeomTypeId.GEOS_POINT:
                return self.createMultiPoint(newGeoms)

            else:
                return self.createGeometryCollection(newGeoms)

        return newGeoms[0]

    def toGeometry(self, envelope):

        if envelope.isNull:
            return self.createPoint()

        if envelope.width == 0 and envelope.height == 0:
            coord = Coordinate(envelope.minx, envelope.miny)
            return self.createPoint(coord)

        cl = self.coordinateSequenceFactory.create([
            Coordinate(envelope.minx, envelope.miny),
            Coordinate(envelope.maxx, envelope.miny),
            Coordinate(envelope.maxx, envelope.maxy),
            Coordinate(envelope.minx, envelope.maxy),
            Coordinate(envelope.minx, envelope.miny)
            ])
        return self.createPolygon(self.createLinearRing(cl), None)

    def createCoordinate(self, co):
        x, y, z = 0, 0, 0
        try:
            iter(co)
            if len(co) == 2:
                x, y = co
            if len(co) == 3:
                x, y, z = co
        except:
            pass
        try:
            x, y = co.x, co.y
        except:
            pass
        return Coordinate(x, y, z)

    # output geometry using an outputFactory
    def output(self, geoms, name="output", multiple=False):
        if self.outputFactory is not None:
            return self.outputFactory.output(geoms, name, multiple)

    def outputCoord(self, coord, name):
        if self.outputFactory is not None:
            return self.outputFactory.outputCoord(coord, name)
