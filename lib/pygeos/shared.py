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


from math import sqrt, log, ceil, floor
import logging
logger = logging.getLogger("pygeos")


def quicksort(array, sortFunc, begin=0, end=None):
    """
     * Quick sort in place array
     * @param array: the array to sort in place
     * @param sortFunc: function(@a, @b) -> bool to compare two items
    """
    if end is None:
        end = len(array) - 1

    def _partition(array, begin, end):
        pivot = begin
        for i in range(begin + 1, end + 1):
            if sortFunc(array[i], array[begin]):
                pivot += 1
                array[i], array[pivot] = array[pivot], array[i]
        array[pivot], array[begin] = array[begin], array[pivot]
        return pivot

    def _quicksort(array, begin, end):
        if begin >= end:
            return
        pivot = _partition(array, begin, end)
        _quicksort(array, begin, pivot - 1)
        _quicksort(array, pivot + 1, end)

    return _quicksort(array, begin, end)


class CAP_STYLE():
    """
     * Buffer operation Cap options
    """
    # Specifies a round line buffer end cap style.
    round = 1
    # Specifies a flat line buffer end cap style.
    flat = 2
    # Specifies a square line buffer end cap style.
    square = 3


class JOIN_STYLE():
    """
     * Buffer operation Join options
    """
    # Specifies a round join style.
    round = 1
    # Specifies a mitre join style.
    mitre = 2
    # Specifies a bevel join style.
    bevel = 3


class GeomTypeId():
    # Geometry types (GeomTypeId)
    # a point
    GEOS_POINT = 0
    # a linestring
    GEOS_LINESTRING = 1
    # a linear ring (linestring with 1st point == last point)
    GEOS_LINEARRING = 2
    # a polygon
    GEOS_POLYGON = 3
    # a collection of points
    GEOS_MULTIPOINT = 4
    # a collection of linestrings
    GEOS_MULTILINESTRING = 5
    # a collection of polygons
    GEOS_MULTIPOLYGON = 6
    # a collection of heterogeneus geometries
    GEOS_GEOMETRYCOLLECTION = 7


class TopologyException(Exception):
    """
     * Indicates an invalid or inconsistent topological situation encountered
     * during processing
    """
    def __init__(self, message="", coord=None):
        if coord is None:
            msg = "TopologyException: {}".format(message)
        else:
            msg = "TopologyException: {} at {}".format(message, coord)
        Exception.__init__(self, msg)
        self.coord = coord


class PrecisionModel():
    """
     * PrecisionModel
     *
     * Specifies the precision model of the Coordinate in a Geometry.
     *
     * In other words, specifies the grid of allowable
     * points for all Geometrys.
     *
     * The makePrecise method allows rounding a coordinate to
     * a "precise" value that is, one whose
     * precision is known exactly.
     *
     * Coordinates are assumed to be precise in geometries.
     * That is, the coordinates are assumed to be rounded to the
     * precision model given for the geometry.
     * JTS input routines automatically round coordinates to the precision model
     * before creating Geometries.
     * All internal operations
     * assume that coordinates are rounded to the precision model.
     * Constructive methods (such as boolean operations) always round computed
     * coordinates to the appropriate precision model.
     *
     * Currently three types of precision model are supported:
     * - FLOATING - represents full double precision floating point.
     *   This is the default precision model used in JTS
     * - FLOATING_SINGLE - represents single precision floating point.
     * - FIXED - represents a model with a fixed number of decimal places.
     *   A Fixed Precision Model is specified by a scale factor.
     *   The scale factor specifies the grid which numbers are rounded to.
     *   Input coordinates are mapped to fixed coordinates according to the
     *   following equations:
     *   - jtsPt.x = round( inputPt.x * scale ) / scale
     *   - jtsPt.y = round( inputPt.y * scale ) / scale
     *
     * For example, to specify 3 decimal places of precision, use a scale factor
     * of 1000. To specify -3 decimal places of precision (i.e. rounding to
     * the nearest 1000), use a scale factor of 0.001.
     *
     * Coordinates are represented internally as Java double-precision values.
     * Since Java uses the IEEE-394 floating point standard, this
     * provides 53 bits of precision. (Thus the maximum precisely representable
     * integer is 9,007,199,254,740,992).
     *
    """

    FIXED = 0
    FLOATING = 1
    FLOATING_SINGLE = 2
    maximumPreciseValue = 9007199254740992.0

    def __init__(self, scale: float=0, modelType: int=None):

        if modelType is None:
            if scale != 0:
                modelType = PrecisionModel.FIXED
            else:
                modelType = PrecisionModel.FLOATING

        self.modelType = modelType
        self.scale = scale

    def _makePrecise(self, val: float) -> None:
        res = val
        if self.modelType == PrecisionModel.FLOATING_SINGLE:
            res = round(val, 6)
        elif self.modelType == PrecisionModel.FIXED:
            # Use whatever happens to be the default rounding method
            res = round(val * self.scale) / self.scale

        # assert(abs(res - val) > self.scale), "makeprecise issue"

        return res

    def makePrecise(self, coord) -> None:
        if self.modelType == PrecisionModel.FLOATING:
            return
        coord.x = self._makePrecise(coord.x)
        coord.y = self._makePrecise(coord.y)

    def getMaximumSignificantDigits(self) -> int:
        maxSigDigits = 16

        if self.modelType == PrecisionModel.FLOATING_SINGLE:
            maxSigDigits = 6

        elif self.modelType == PrecisionModel.FIXED:
            dgtsd = log(self.scale) / log(10.0)
            if dgtsd > 0:
                maxSigDigits = int(ceil(dgtsd))
            else:
                maxSigDigits = int(floor(dgtsd))

        return maxSigDigits

    def compareTo(self, other) -> int:
        sigDigits = self.getMaximumSignificantDigits()
        othDigits = other.getMaximumSignificantDigits()
        if sigDigits < othDigits:
            return -1
        elif sigDigits > othDigits:
            return 1
        return 0

    @property
    def isFloating(self) -> bool:
        return self.modelType == PrecisionModel.FLOATING_SINGLE or self.modelType == PrecisionModel.FLOATING

    def __eq__(self, other) -> bool:
        return self.scale == other.scale and self.isFloating and other.isFloating

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)


class GeometryCombiner():

    def __init__(self, geoms):

        if len(geoms) > 0:
            self._factory = geoms[0]._factory
        else:
            self._factory = None

        self.skipEmpty = False
        self.geoms = geoms

    @staticmethod
    def combine(g0, g1=None, g2=None):
        if g2 is not None:
            geoms = [g0, g1, g2]
        elif g1 is not None:
            geoms = [g0, g1]
        else:
            geoms = g0

        combiner = GeometryCombiner(geoms)
        return combiner._combine()

    def _combine(self):
        elems = []
        for geom in self.geoms:
            self.extractElements(geom, elems)

        if len(elems) == 0:
            if self._factory is not None:
                return self._factory.createGeometryCollection(None)
            return None

        return self._factory.buildGeometry(elems)

    def extractElements(self, geom, elems) -> None:

        if geom is None:
            return

        for i in range(geom.numgeoms):
            g = geom.getGeometryN(i)
            if self.skipEmpty and g.is_empty:
                continue
            elems.append(g)


class GeometryTransformer():

    def __init__(self):
        self._factory = None
        self.geom = None
        self.pruneEmptyGeometry = True
        self.preserveGeometryCollectionType = True
        self.preserveCollections = False
        self.preserveType = False
        self.skipTransformedInvalidInteriorRings = False

    def transform(self, geom):
        self.geom = geom
        self._factory = geom._factory
        type_id = geom.type_id
        if type_id == GeomTypeId.GEOS_POINT:
            return self.transformPoint(geom, None)

        elif type_id == GeomTypeId.GEOS_LINESTRING:
            return self.transformLineString(geom, None)

        elif type_id == GeomTypeId.GEOS_LINEARRING:
            return self.transformLinearRing(geom, None)

        elif type_id == GeomTypeId.GEOS_POLYGON:
            return self.transformPolygon(geom, None)

        elif type_id == GeomTypeId.GEOS_MULTIPOINT:
            return self.transformMultiPoint(geom, None)

        elif type_id == GeomTypeId.GEOS_MULTILINESTRING:
            return self.transformMultiLineString(geom, None)

        elif type_id == GeomTypeId.GEOS_MULTIPOLYGON:
            return self.transformMultiPolygon(geom, None)

        elif type_id == GeomTypeId.GEOS_GEOMETRYCOLLECTION:
            return self.transformGeometryCollection(geom, None)

        raise ValueError("Unknown Geometry subtype.")

    def createCoordinateSequence(self, coords):
        return self._factory.coordinateSequenceFactory.create(coords)

    def transformCoordinates(self, coords, parent):
        return coords.clone()

    def transformPoint(self, geom, parent):
        cs = self.transformCoordinates(geom.coords, geom)
        return self._factory.createPoint(cs)

    def transformMultiPoint(self, geom, parent):
        transGeomList = []
        for g in geom.geoms:
            transGeom = self.transformPoint(g, geom)
            if transGeom is None or transGeom.is_empty:
                continue
            transGeomList.append(transGeom)

        return self._factory.buildGeometry(transGeomList)

    def transformLinearRing(self, geom, parent):
        seq = self.transformCoordinates(geom.coords, geom)
        size = len(seq)
        if size > 0 and size < 4 and not self.preserveType:
            return self._factory.createLineString(seq)
        else:
            return self._factory.createLinearRing(seq)

    def transformLineString(self, geom, parent):
        seq = self.transformCoordinates(geom.coords, geom)
        return self._factory.createLineString(seq)

    def transformMultiLineString(self, geom, parent):
        transGeomList = []
        for g in geom.geoms:
            transGeom = self.transformLineString(g, geom)
            if transGeom is None or transGeom.is_empty:
                continue
            transGeomList.append(transGeom)

        return self._factory.buildGeometry(transGeomList)

    def transformPolygon(self, geom, parent):

        isAllValidLinearRings = True
        lr = geom.exterior
        exterior = self.transformLinearRing(lr, geom)
        if exterior is None or exterior.type_id != GeomTypeId.GEOS_LINEARRING or exterior.is_empty:
            isAllValidLinearRings = False

        interiors = []
        for lr in geom.interiors:
            hole = self.transformLinearRing(lr, geom)
            if hole is None or hole.is_empty:
                continue
            if hole.type_id != GeomTypeId.GEOS_LINEARRING:
                if self.skipTransformedInvalidInteriorRings:
                    continue
                isAllValidLinearRings = False

            interiors.append(hole)

        if isAllValidLinearRings:
            return self._factory.createPolygon(exterior, interiors)
        else:
            components = []
            if exterior is not None:
                components.append(exterior)
            components.extend(interiors)

            return self._factory.buildGeometry(components)

    def transformMultiPolygon(self, geom, parent):
        transGeomList = []
        for g in geom.geoms:
            transGeom = self.transformPolygon(g, geom)
            if transGeom is None or transGeom.is_empty:
                continue
            transGeomList.append(transGeom)

        return self._factory.buildGeometry(transGeomList)

    def transformGeometryCollection(self, geom, parent):
        transGeomList = []
        for g in geom.geoms:
            transGeom = self.transform(g)
            if transGeom is None:
                continue
            if self.pruneEmptyGeometry and transGeom.is_empty:
                continue
            transGeomList.append(transGeom)

        if self.preserveGeometryCollectionType:
            return self._factory.createGeometryCollection(transGeomList)
        else:
            return self._factory.buildGeometry(transGeomList)


class Location():
    """
    *  Used for uninitialized location values.
    """
    UNDEF = -1

    """
    * DE-9IM row index of the interiors of the first geometry and
    * column index of the interiors of the second geometry.
    * Location value for the interiors of a geometry.
    """
    INTERIOR = 0

    """
    * DE-9IM row index of the boundary of the first geometry and
    * column index of the boundary of the second geometry.
    * Location value for the boundary of a geometry.
    """
    BOUNDARY = 1

    """
    * DE-9IM row index of the exterior of the first geometry and
    * column index of the exterior of the second geometry.
    * Location value for the exterior of a geometry.
    """
    EXTERIOR = 2

    @staticmethod
    def toLocationSymbol(loc: int) -> str:
        """
         *  Converts the location value to a location symbol, for example, EXTERIOR => 'e'.
         *
         *@param  locationValue  either EXTERIOR, BOUNDARY, INTERIOR or NULL
         *@return                either 'e', 'b', 'i' or '-'
        """
        if loc == Location.UNDEF:
            return "-"
        elif loc == Location.INTERIOR:
            return 'i'
        elif loc == Location.EXTERIOR:
            return 'e'
        elif loc == Location.BOUNDARY:
            return 'b'
        else:
            return "Unknown location value: {}".format(loc)


class Dimension():
    """
     Constants representing the dimensions of a point, a curve and a surface.
     Also, shared representing the dimensions of the empty geometry and
     non-empty geometries, and a wildcard dimension meaning "any dimension".
    """
    # Dimension value for any dimension (= {FALSE, TRUE}).
    DONTCARE = -3
    # Dimension value of non-empty geometries (= {P, L, A}).
    TRUE = -2
    # Dimension value of the empty geometry (-1).
    FALSE = -1
    # Dimension value of a point (0).
    P = 0
    # Dimension value of a curve (1).
    L = 1
    # Dimension value of a surface (2).
    A = 2

    @staticmethod
    def toDimensionSymbol(val: int) -> str:
        """
         *  Converts the dimension value to a dimension symbol, for example, TRUE => 'T'.
         *
         *@param  dimensionValue  a number that can be stored in the IntersectionMatrix.
         *          Possible values are {TRUE, FALSE, DONTCARE, 0, 1, 2}.
         *@return   a character for use in the string representation of
         *      an IntersectionMatrix. Possible values are {T, F, * , 0, 1, 2}.
        """
        if val == Dimension.FALSE:
            return 'F'
        elif val == Dimension.TRUE:
            return 'T'
        elif val == Dimension.DONTCARE:
            return '*'
        elif val == Dimension.P:
            return '0'
        elif val == Dimension.L:
            return '1'
        elif val == Dimension.A:
            return '2'
        else:
            raise ValueError("Unknown dimension value: {}".format(val))

    @staticmethod
    def toDimensionValue(symbol: str) -> int:
        """
         *  Converts the dimension symbol to a dimension value, for example, '*' => DONTCARE.
         *
         *@param  dimensionSymbol  a character for use in the string representation of
         *      an IntersectionMatrix. Possible values are {T, F, * , 0, 1, 2}.
         *@return       a number that can be stored in the IntersectionMatrix.
         *              Possible values are {TRUE, FALSE, DONTCARE, 0, 1, 2}.
        """
        if symbol in {'f', 'F'}:
            return Dimension.FALSE
        elif symbol in {'t', 'T'}:
            return Dimension.TRUE
        elif symbol == '*':
            return Dimension.DONTCARE
        elif symbol == '0':
            return Dimension.P
        elif symbol == '1':
            return Dimension.L
        elif symbol == '2':
            return Dimension.A
        else:
            raise ValueError("Unknown dimension symbol: {}".format(symbol))


class Quadrant():
    """
     * Utility functions for working with quadrants, which are numbered as follows:
     * 1 | 0
     * --+--
     * 2 | 3
     *
    """
    NE = 0
    NW = 1
    SW = 2
    SE = 3
    @staticmethod
    def from_coords(p0, p1) -> int:
        """
         * Returns the quadrant of a directed line segment (specified as x and y
         * displacements, which cannot both be 0).
         * 
         * @raise ValueError if the displacements are both 0
        """
        if p1.x == p0.x and p1.y == p0.y:
            raise ValueError("Cannot compute the quadrant")

        if p1.x >= p0.x:
            if p1.y >= p0.y:
                return Quadrant.NE
            else:
                return Quadrant.SE
        else:
            if p1.y >= p0.y:
                return Quadrant.NW
            else:
                return Quadrant.SW
                
    @staticmethod
    def quadrant(dx, dy) -> int:
        """
         * Returns the quadrant of a directed line segment (specified as x and y
         * displacements, which cannot both be 0).
         * 
         * @raise ValueError if the displacements are both 0
        """
        if dx == 0.0 and dy == 0.0:
            raise ValueError("Cannot compute the quadrant")

        if dx >= 0:
            if dy >= 0:
                return Quadrant.NE
            else:
                return Quadrant.SE
        else:
            if dy >= 0:
                return Quadrant.NW
            else:
                return Quadrant.SW

    @staticmethod
    def isNorthern(quad: int) -> bool:
        return quad == Quadrant.NE or quad == Quadrant.NW


class Position():
    # An indicator that a Location is <i>on</i>
    # a GraphComponent
    ON = 0
    # An indicator that a Location is to the
    # <i>left</i> of a GraphComponent
    LEFT = 1
    # An indicator that a Location is to the
    # <i>right</i> of a GraphComponent
    RIGHT = 2

    @staticmethod
    def opposite(position: int) -> int:
        """
         * Returns LEFT if the position is RIGHT, RIGHT if
         * the position is LEFT, or the position otherwise.
        """
        if position == Position.LEFT:
            return Position.RIGHT
        if position == Position.RIGHT:
            return Position.LEFT

        logger.debug("Position.opposite: position is neither LEFT (1) nor RIGHT (2) but {}", position)

        return position


class Envelope():
    """
     * An Envelope defines a rectangulare region of the 2D coordinate plane.
     *
     * It is often used to represent the bounding box of a Geometry,
     * e.g. the minimum and maximum x and y values of the Coordinates.
     *
     * Note that Envelopes support infinite or half-infinite regions, by using
     * the values of Double_POSITIVE_INFINITY and
     * Double_NEGATIVE_INFINITY.
     *
     * When Envelope objects are created or initialized,
     * the supplies extent values are automatically sorted into the correct order.
     *
    """
    def __init__(self, x1=None, y1=None, x2=None, y2=None):
        """
         * Creates an Envelope for a region defined by
         * maximum and minimum values.
         *
         * @param  x1  the first x-value
         * @param  x2  the second x-value
         * @param  y1  the first y-value
         * @param  y2  the second y-value
        """
        if x1 is None:
            # Creates a null Envelope.
            self.minx = 0
            self.maxx = -1
            self.miny = 0
            self.maxy = -1

        elif y1 is None:

            cls = type(x1).__name__

            if cls == 'Envelope':
                # Copy constructor
                self.initByEnvelope(x1)
            else:
                # Creates an Envelope for a region defined by a single Coordinate.
                self.initByPoints(x1, x1)

        elif x2 is None:
            # Creates an Envelope for a region defined by
            # two Coordinates.
            self.initByPoints(x1, y1)

        else:
            # Creates an Envelope for a region defined by
            # maximum and minimum values.
            self.init(x1, x2, y1, y2)
    
    def initByEnvelope(self, env):
        self.minx = env.minx
        self.maxx = env.maxx
        self.miny = env.miny
        self.maxy = env.maxy
    
    def initByPoints(self, p0, p1):
        self.init(p0.x, p1.x, p0.y, p1.y)
    
    def init(self, x1, x2, y1, y2) -> None:
        if x1 < x2:
            self.minx = x1
            self.maxx = x2
        else:
            self.minx = x2
            self.maxx = x1

        if y1 < y2:
            self.miny = y1
            self.maxy = y2
        else:
            self.miny = y2
            self.maxy = y1

    def equals(self, other) -> bool:
        """
         * Returns true if the Envelope other
         * spatially equals this Envelope.
         *
         * @param  other the Envelope which this
         *               Envelope is being checked for equality
         *
         * @return true if this and other
         *         Envelope objs are spatially equal
        """
        return (other.minx == self.minx and
            other.maxx == self.maxx and
            other.miny == self.miny and
            other.maxy == self.maxy)

    def contains(self, other, y=None) -> bool:
        """
         * Tests if the Envelope other lies wholely
         * inside this Envelope (inclusive of the boundary).
         *
         * Note that this is <b>not</b> the same definition as the SFS
         * <tt>contains</tt>, which would exclude the envelope boundary.
         *
         * @param  other the Envelope to check
         * @return true if other is contained in this
         *              Envelope
         *
         * @see covers(Envelope)
        """
        return self.covers(other, y)

    def _expandToInclude(self, x, y) -> None:
        if self.isNull:
            self.minx = x
            self.maxx = x
            self.miny = y
            self.maxy = y
        else:
            if x < self.minx:
                self.minx = x
            elif x > self.maxx:
                self.maxx = x

            if y < self.miny:
                self.miny = y
            elif y > self.maxy:
                self.maxy = y

    def expandToInclude(self, other, y=None) -> None:
        if y is None:
            cls = type(other).__name__
            if cls == 'Envelope':
                if other.isNull:
                    return
                self._expandToInclude(other.minx, other.miny)
                self._expandToInclude(other.maxx, other.maxy)
            else:
                self._expandToInclude(other.x, other.y)
        else:
            self._expandToInclude(other, y)

    def covers(self, other, y=None) -> bool:
        """
         * Tests if the Envelope other lies wholely inside
         * this Envelope (inclusive of the boundary).
         *
         * @param  other the Envelope to check
         * @return true if this Envelope covers the
         * other
        """
        if y is None:
            cls = type(other).__name__
            if cls == 'Envelope':
                return (other.minx >= self.minx and
                    other.maxx <= self.maxx and
                    other.miny >= self.miny and
                    other.maxy <= self.maxy)
            else:
                # covers self by coord
                y, other = other.y, other.x

        return (self.minx <= other <= self.maxx and
                self.miny <= y <= self.maxy)

    @property
    def width(self):
        return self.maxx - self.minx

    @property
    def height(self):
        return self.maxy - self.miny

    @property
    def area(self):
        return self.width * self.height

    @property
    def isNull(self) -> bool:
        """
         * Returns true if this Envelope
         * is a "null" envelope.
         *
         * @return true if this Envelope
         *         is uninitialized or is the envelope of the
         *         empty geometry.
        """
        return self.maxx < self.minx

    def intersects(self, other, y=None) -> bool:

        if self.isNull:
            return False

        if y is None:
            if type(other).__name__ == 'Envelope':

                if other.isNull:
                    return False

                return not (other.minx > self.maxx or
                    other.maxx < self.minx or
                    other.miny > self.maxy or
                    other.maxy < self.miny)
            else:
                # intersects self by coord
                y, other = other.y, other.x

        # coords inside
        return (self.miny <= y <= self.maxy and
            self.minx <= other <= self.maxx)

    @staticmethod
    def static_intersects(p1, p2, q1, q2=None) -> bool:
        if q2 is None:
            # static point inside
            if p1.x > p2.x:
                minx, maxx = p2.x, p1.x
            else:
                minx, maxx = p1.x, p2.x

            if p1.y > p2.y:
                miny, maxy = p2.y, p1.y
            else:
                miny, maxy = p1.y, p2.y

            return maxx >= q1.x >= minx and maxy >= q1.y >= miny

        else:
            # static any of points inside

            if p1.x > p2.x:
                minp, maxp = p2.x, p1.x
            else:
                minp, maxp = p1.x, p2.x

            if q1.x > q2.x:
                minq, maxq = q2.x, q1.x
            else:
                minq, maxq = q1.x, q2.x

            if minp > maxq:
                return False

            if maxp < minq:
                return False

            if p1.y > p2.y:
                minp, maxp = p2.y, p1.y
            else:
                minp, maxp = p1.y, p2.y

            if q1.y > q2.y:
                minq, maxq = q2.y, q1.y
            else:
                minq, maxq = q1.y, q2.y

            if minp > maxq:
                return False

            if maxp < minq:
                return False

        return True

    def centre(self, coord) -> bool:
        """
         * Computes the coordinate of the centre of this envelope
         * (as long as it is non-null)
         *
         * @param centre The coordinate to write results into
         * @return NULL is the center could not be found
         * (null envelope).
        """
        if self.isNull:
            return False

        coord.x = (self.minx + self.maxx) / 2.0
        coord.y = (self.miny + self.maxy) / 2.0

        return True

    def getCentre(self):
        """
         * Computes the coordinate of the centre of this envelope
         * (as long as it is non-null)
         *
         * @param centre The coordinate to write results into
         * @return NULL is the center could not be found
         * (null envelope).
        """
        if self.isNull:
            return Coordinate()

        return Coordinate(
            (self.minx + self.maxx) / 2.0,
            (self.miny + self.maxy) / 2.0)

    def intersection(self, env, result) -> bool:
        """
         * Computes the intersection of two {Envelopes}
         *
         * @param env the envelope to intersect with
         * @param result the envelope representing the intersection of
         *               the envelopes (this will be the null envelope
         *               if either argument is null, or they do not intersect)
         * @return false if not intersection is found
        """
        if self.isNull or env.isNull or not self.intersects(env):
            return False

        if self.minx > env.minx:
            minx = self.minx
        else:
            minx = env.minx

        if self.miny > env.miny:
            miny = self.miny
        else:
            miny = env.miny

        if self.maxx < env.maxx:
            maxx = self.maxx
        else:
            maxx = env.maxx

        if self.maxy < env.maxy:
            maxy = self.maxy
        else:
            maxy = env.maxy

        result.init(minx, maxx, miny, maxy)

        return True

    def expandBy(self, x, y=None) -> None:
        if y is None:
            self.expandBy(x, x)
        else:
            self.minx -= x
            self.maxx += x
            self.miny -= y
            self.maxy += y
   
    def __str__(self) -> str:
        return "Env[x{}:{}, y{}:{}]".format(self.minx, self.maxx, self.miny, self.maxy)

    def __hash__(self):
        return hash((self.minx, self.miny, self.maxx, self.maxy))
    
    
class IntersectionMatrix():
    """
    * Implementation of Dimensionally Extended Nine-Intersection Model
    * (DE-9IM) matrix.
    *
    * Dimensionally Extended Nine-Intersection Model (DE-9IM) matrix.
    * This class can used to represent both computed DE-9IM's (like 212FF1FF2)
    * as well as patterns for matching them (like T*T******).
    *
    * Methods are provided to:
    *
    *  - set and query the elements of the matrix in a convenient fashion
    *  - convert to and from the standard string representation
    *    (specified in SFS Section 2.1.13.2).
    *  - test to see if a matrix matches a given pattern string.
    *
    * For a description of the DE-9IM, see the
    * <a href="http://www.opengis.org/techno/specs.htm">OpenGIS Simple
    * Features Specification for SQL.</a>
    *
    * \todo Suggestion: add equal and not-equal operator to this class.
    """
    def __init__(self, elements=None):
        """
         * Default constructor.
         *
         * Creates an IntersectionMatrix with Dimension.False
         * dimension values ('F').
        """
        self.firstDim = 3
        self.secondDim = 3
        self.matrix = [
            [Dimension.FALSE, Dimension.FALSE, Dimension.FALSE],
            [Dimension.FALSE, Dimension.FALSE, Dimension.FALSE],
            [Dimension.FALSE, Dimension.FALSE, Dimension.FALSE]]

        cls = type(elements).__name__

        if cls == 'str':
            self.set(elements)
        elif cls == 'IntersectionMatrix':
            _other = elements.matrix
            self.matrix[Location.INTERIOR][Location.INTERIOR] = _other[Location.INTERIOR][Location.INTERIOR]
            self.matrix[Location.INTERIOR][Location.BOUNDARY] = _other[Location.INTERIOR][Location.BOUNDARY]
            self.matrix[Location.INTERIOR][Location.EXTERIOR] = _other[Location.INTERIOR][Location.EXTERIOR]
            self.matrix[Location.BOUNDARY][Location.INTERIOR] = _other[Location.BOUNDARY][Location.INTERIOR]
            self.matrix[Location.BOUNDARY][Location.BOUNDARY] = _other[Location.BOUNDARY][Location.BOUNDARY]
            self.matrix[Location.BOUNDARY][Location.EXTERIOR] = _other[Location.BOUNDARY][Location.EXTERIOR]
            self.matrix[Location.EXTERIOR][Location.INTERIOR] = _other[Location.EXTERIOR][Location.INTERIOR]
            self.matrix[Location.EXTERIOR][Location.BOUNDARY] = _other[Location.EXTERIOR][Location.BOUNDARY]
            self.matrix[Location.EXTERIOR][Location.EXTERIOR] = _other[Location.EXTERIOR][Location.EXTERIOR]

    def matches(self, required: str) -> bool:
        """
         * Returns whether the elements of this IntersectionMatrix
         * satisfies the required dimension symbols.
         *
         * @param requiredDimensionSymbols - nine dimension symbols with
         *        which to compare the elements of this IntersectionMatrix.
         *        Possible values are {T, F, * , 0, 1, 2}.
         * @return true if this IntersectionMatrix matches the required
         *         dimension symbols.
        """
        if len(required) != 9:
            raise ValueError(
                "ValueError: Should be length 9, is [{}] instead".format(len(required))
                )

        _matrix = self.matrix
        for i in range(self.firstDim):
            for j in range(self.secondDim):
                if not self._matches(_matrix[i][j], required[3 * i + j]):
                    return False
        return True

    def _matches(self, actual, required: str) -> bool:
        """
         * Tests if given dimension value satisfies the dimension symbol.
         *
         * @param actualDimensionValue - valid dimension value stored in
         *        the IntersectionMatrix.
         *        Possible values are {TRUE, FALSE, DONTCARE, 0, 1, 2}.
         * @param requiredDimensionSymbol - a character used in the string
         *        representation of an IntersectionMatrix.
         *        Possible values are {T, F, * , 0, 1, 2}.
         * @return true if the dimension symbol encompasses the
         *         dimension value.
        """
        if type(actual).__name__ == 'str':
            m = IntersectionMatrix(actual)
            return m.matches(required)

        else:
            if required == '*':
                return True
            elif required == 'T' and (actual >= 0 or actual == Dimension.TRUE):
                return True
            elif required == 'F' and (actual == Dimension.FALSE):
                return True
            elif required == '0' and (actual == Dimension.P):
                return True
            elif required == '1' and (actual == Dimension.L):
                return True
            elif required == '2' and (actual == Dimension.A):
                return True
            else:
                return False

    @staticmethod
    def static_matches(actual, required: str) -> bool:
        """
         * Tests if given dimension value satisfies the dimension symbol.
         *
         * @param actualDimensionValue - valid dimension value stored in
         *        the IntersectionMatrix.
         *        Possible values are {TRUE, FALSE, DONTCARE, 0, 1, 2}.
         * @param requiredDimensionSymbol - a character used in the string
         *        representation of an IntersectionMatrix.
         *        Possible values are {T, F, * , 0, 1, 2}.
         * @return true if the dimension symbol encompasses the
         *         dimension value.
        """
        if type(actual).__name__ == 'str':
            m = IntersectionMatrix(actual)
            return m.matches(required)

        else:
            if required == '*':
                return True
            elif required == 'T' and (actual >= 0 or actual == Dimension.TRUE):
                return True
            elif required == 'F' and (actual == Dimension.FALSE):
                return True
            elif required == '0' and (actual == Dimension.P):
                return True
            elif required == '1' and (actual == Dimension.L):
                return True
            elif required == '2' and (actual == Dimension.A):
                return True
            else:
                return False

    def add(self, other) -> None:
        """
         * Adds one matrix to another.
         *
         * Addition is defined by taking the maximum dimension value
         * of each position in the summand matrices.
         *
         * @param other - the matrix to add.
        """
        _other = other.matrix
        for i in range(self.firstDim):
            for j in range(self.secondDim):
                self.setAtLeast(i, j, _other[i][j])

    def set(self, row, col=None, dimensionValue=None) -> None:
        """
         * Changes the value of one of this IntersectionMatrixs elements.
         * Alternative 1:
         * @param row - the row of this IntersectionMatrix, indicating
         *        the interiors, boundary or exterior of the first Geometry.
         * @param column - the column of this IntersectionMatrix,
         *        indicating the interiors, boundary or exterior of the
         *        second Geometry.
         * @param dimensionValue - the new value of the element.
         * Alternative 2:
         * @param dimensionSymbols - nine dimension symbols to which to
         *        set this IntersectionMatrix elements.
         *        Possible values are {T, F, * , 0, 1, 2}.
        """
        if col is None:
            # row is dimensionSymbols
            dimensionSymbols = row
            for i, ds in enumerate(dimensionSymbols):
                col = i % self.secondDim
                row = int((i - col) / self.firstDim)
                self.matrix[row][col] = Dimension.toDimensionValue(ds)
        else:
            self.matrix[row][col] = dimensionValue

    def setAtLeast(self, row, col=None, dimensionValue=None) -> None:
        """
         * Changes the specified element to minimumDimensionValue if the
         * element is less.
         * Alternative 1:
         * @param row - the row of this IntersectionMatrix, indicating
         *        the interiors, boundary or exterior of the first Geometry.
         * @param column -  the column of this IntersectionMatrix, indicating
         *        the interiors, boundary or exterior of the second Geometry.
         * @param minimumDimensionValue - the dimension value with which
         *        to compare the element.  The order of dimension values
         *        from least to greatest is {DONTCARE, TRUE, FALSE, 0, 1, 2}.
         * Alternative 2:
         * @param minimumDimensionSymbols -
         *        nine dimension symbols with which
         *        to compare the elements of this IntersectionMatrix.
         *        The order of dimension values from least to greatest is
         *        {DONTCARE, TRUE, FALSE, 0, 1, 2}  .
        """
        if col is None:
            dimensionSymbols = row
            for i, ds in enumerate(dimensionSymbols):
                col = i % self.secondDim
                row = int((i - col) / self.firstDim)
                self.setAtLeast(row, col, Dimension.toDimensionValue(ds))
        else:
            if self.matrix[row][col] < dimensionValue:
                self.matrix[row][col] = dimensionValue

    def setAtLeastIfValid(self, row: int, col: int, minimumDimensionValue: int) -> None:
        """
         * If row >= 0 and column >= 0, changes the specified element
         * to minimumDimensionValue if the element is less.
         * Does nothing if row <0 or column < 0.
         *
         * @param row -
         *        the row of this IntersectionMatrix,
         *        indicating the interiors, boundary or exterior of the
         *        first Geometry.
         *
         * @param column -
         *        the column of this IntersectionMatrix,
         *        indicating the interiors, boundary or exterior of the
         *        second Geometry.
         *
         * @param minimumDimensionValue -
         *        the dimension value with which
         *        to compare the element. The order of dimension values
         *        from least to greatest is {DONTCARE, TRUE, FALSE, 0, 1, 2}.
        """
        if row >= 0 and col >= 0:
            self.setAtLeast(row, col, minimumDimensionValue)

    def setAll(self, dimensionValue: int) -> None:
        """
         * Changes the elements of this IntersectionMatrix to dimensionValue.
         *
         * @param dimensionValue -
         *        the dimension value to which to set this
         *        IntersectionMatrix elements. Possible values {TRUE,
         *        FALSE, DONTCARE, 0, 1, 2}.
        """
        for i in range(self.firstDim):
            for j in range(self.secondDim):
                self.matrix[i][j] = dimensionValue

    def get(self, row: int, col: int) -> int:
        """
         * Returns the value of one of this IntersectionMatrixs elements.
         *
         * @param row -
         *        the row of this IntersectionMatrix, indicating the
         *        interiors, boundary or exterior of the first Geometry.
         *
         * @param column -
         *        the column of this IntersectionMatrix, indicating the
         *        interiors, boundary or exterior of the second Geometry.
         *
         * @return the dimension value at the given matrix position.
        """
        return self.matrix[row][col]

    @property
    def isDisjoint(self) -> bool:
        """
         * Returns true if this IntersectionMatrix is FF*FF****.
         *
         * @return true if the two Geometrys related by this
         *         IntersectionMatrix are disjoint.
        """
        return (
            self.matrix[Location.INTERIOR][Location.INTERIOR] == Dimension.FALSE and
            self.matrix[Location.INTERIOR][Location.BOUNDARY] == Dimension.FALSE and
            self.matrix[Location.BOUNDARY][Location.INTERIOR] == Dimension.FALSE and
            self.matrix[Location.BOUNDARY][Location.BOUNDARY] == Dimension.FALSE
            )

    @property
    def isIntersects(self) -> bool:
        """
         * Returns true if isDisjoint returns false.
         *
         * @return true if the two Geometrys related by this
         *         IntersectionMatrix intersect.
        """
        return not self.isDisjoint

    def isTouches(self, dimensionOfGeometryA: int, dimensionOfGeometryB: int) -> bool:
        """
         * Returns true if this IntersectionMatrix is FT*******, F**T*****
         * or F***T****.
         *
         * @param dimensionOfGeometryA - the dimension of the first Geometry.
         *
         * @param dimensionOfGeometryB - the dimension of the second Geometry.
         *
         * @return true if the two Geometry's related by this
         *         IntersectionMatrix touch, false if both Geometrys
         *         are points.
        """
        if dimensionOfGeometryA > dimensionOfGeometryB:
            return self.isTouches(dimensionOfGeometryB, dimensionOfGeometryA)

        if ((dimensionOfGeometryA == Dimension.A and dimensionOfGeometryB == Dimension.A) or
                (dimensionOfGeometryA == Dimension.L and dimensionOfGeometryB == Dimension.L) or
                (dimensionOfGeometryA == Dimension.L and dimensionOfGeometryB == Dimension.A) or
                (dimensionOfGeometryA == Dimension.P and dimensionOfGeometryB == Dimension.A) or
                (dimensionOfGeometryA == Dimension.P and dimensionOfGeometryB == Dimension.L)):
            return self.matrix[Location.INTERIOR][Location.INTERIOR] == Dimension.FALSE and (
                self._matches(self.matrix[Location.INTERIOR][Location.BOUNDARY], 'T') or
                self._matches(self.matrix[Location.BOUNDARY][Location.INTERIOR], 'T') or
                self._matches(self.matrix[Location.BOUNDARY][Location.BOUNDARY], 'T')
                )

    def isCrosses(self, dimensionOfGeometryA: int, dimensionOfGeometryB: int) -> bool:
        """
         * Returns true if this IntersectionMatrix is:
         * - T*T****** (for a point and a curve, a point and an area or
         *   a line and an area)
         * - 0******** (for two curves)
         *
         * @param dimensionOfGeometryA - he dimension of the first Geometry.
         *
         * @param dimensionOfGeometryB - the dimension of the second Geometry.
         *
         * @return true if the two Geometry's related by this
         *         IntersectionMatrix cross.
         *
         * For this function to return true, the Geometrys must be a point
         * and a curve; a point and a surface; two curves; or a curve and
         * a surface.
        """
        if ((dimensionOfGeometryA == Dimension.P and dimensionOfGeometryB == Dimension.L) or
                (dimensionOfGeometryA == Dimension.P and dimensionOfGeometryB == Dimension.A) or
                (dimensionOfGeometryA == Dimension.L and dimensionOfGeometryB == Dimension.A)):
            return (self._matches(self.matrix[Location.INTERIOR][Location.INTERIOR], 'T') and
                    self._matches(self.matrix[Location.INTERIOR][Location.EXTERIOR], 'T'))

        if ((dimensionOfGeometryA == Dimension.L and dimensionOfGeometryB == Dimension.P) or
                (dimensionOfGeometryA == Dimension.A and dimensionOfGeometryB == Dimension.P) or
                (dimensionOfGeometryA == Dimension.A and dimensionOfGeometryB == Dimension.L)):
            return (self._matches(self.matrix[Location.INTERIOR][Location.INTERIOR], 'T') and
                    self._matches(self.matrix[Location.EXTERIOR][Location.INTERIOR], 'T'))

        if (dimensionOfGeometryA == Dimension.L and dimensionOfGeometryB == Dimension.L):
            return self.matrix[Location.INTERIOR][Location.INTERIOR] == Dimension.P

        return False

    @property
    def isWithin(self) -> bool:
        """
         * Returns true if this IntersectionMatrix is T*F**F***.
         *
         * @return true if the first Geometry is within the second.
        """
        return (self._matches(self.matrix[Location.INTERIOR][Location.INTERIOR], 'T') and
            self.matrix[Location.INTERIOR][Location.EXTERIOR] == Dimension.FALSE and
            self.matrix[Location.BOUNDARY][Location.EXTERIOR] == Dimension.FALSE)

    @property
    def isContains(self) -> bool:
        """
         * Returns true if this IntersectionMatrix is T*****FF*.
         *
         * @return true if the first Geometry contains the second.
        """
        return (self._matches(self.matrix[Location.INTERIOR][Location.INTERIOR], 'T') and
            self.matrix[Location.EXTERIOR][Location.INTERIOR] == Dimension.FALSE and
            self.matrix[Location.EXTERIOR][Location.BOUNDARY] == Dimension.FALSE)

    def isEquals(self, dimensionOfGeometryA: int, dimensionOfGeometryB: int) -> bool:
        """
         * Returns true if this IntersectionMatrix is T*F**FFF*.
         *
         * @param dimensionOfGeometryA - he dimension of the first Geometry.
         * @param dimensionOfGeometryB - the dimension of the second Geometry.
         * @return true if the two Geometry's related by this
         *         IntersectionMatrix are equal; the Geometrys must have
         *         the same dimension for this function to return true
        """
        if dimensionOfGeometryA != dimensionOfGeometryB:
            return False
        return (self._matches(self.matrix[Location.INTERIOR][Location.INTERIOR], 'T') and
            self.matrix[Location.EXTERIOR][Location.INTERIOR] == Dimension.FALSE and
            self.matrix[Location.INTERIOR][Location.EXTERIOR] == Dimension.FALSE and
            self.matrix[Location.EXTERIOR][Location.BOUNDARY] == Dimension.FALSE and
            self.matrix[Location.BOUNDARY][Location.EXTERIOR] == Dimension.FALSE)

    def isOverlaps(self, dimensionOfGeometryA: int, dimensionOfGeometryB: int) -> bool:
        """
         * Returns true if this IntersectionMatrix is:
         * - T*T***T** (for two points or two surfaces)
         * - 1*T***T** (for two curves)
         *
         * @param dimensionOfGeometryA - he dimension of the first Geometry.
         * @param dimensionOfGeometryB - the dimension of the second Geometry.
         * @return true if the two Geometry's related by this
         *         IntersectionMatrix overlap.
         *
         * For this function to return true, the Geometrys must be two points,
         * two curves or two surfaces.
        """
        if ((dimensionOfGeometryA == Dimension.P and dimensionOfGeometryB == Dimension.A) or
                (dimensionOfGeometryA == Dimension.A and dimensionOfGeometryB == Dimension.P)):
            return (self._matches(self.matrix[Location.INTERIOR][Location.INTERIOR], 'T') and
                self._matches(self.matrix[Location.INTERIOR][Location.EXTERIOR], 'T') and
                self._matches(self.matrix[Location.EXTERIOR][Location.INTERIOR], 'T'))

        if (dimensionOfGeometryA == Dimension.L and dimensionOfGeometryB == Dimension.L):
            return (self.matrix[Location.INTERIOR][Location.INTERIOR] == Dimension.L and
                self._matches(self.matrix[Location.INTERIOR][Location.EXTERIOR], 'T') and
                self._matches(self.matrix[Location.EXTERIOR][Location.INTERIOR], 'T'))

        return False

    @property
    def isCovers(self) -> bool:
        """
         * Returns true if this IntersectionMatrix is T*****FF*
         * or *T****FF* or ***T**FF*
         * or ****T*FF*
         *
         * @return true if the first Geometry covers the
         * second
        """
        return ((self._matches(self.matrix[Location.INTERIOR][Location.INTERIOR], 'T') or
                self._matches(self.matrix[Location.INTERIOR][Location.BOUNDARY], 'T') or
                self._matches(self.matrix[Location.BOUNDARY][Location.INTERIOR], 'T') or
                self._matches(self.matrix[Location.BOUNDARY][Location.BOUNDARY], 'T')) and
            self.matrix[Location.EXTERIOR][Location.INTERIOR] == Dimension.FALSE and
            self.matrix[Location.EXTERIOR][Location.BOUNDARY] == Dimension.FALSE)

    @property
    def isCoveredBy(self) -> bool:
        """
         * Returns true if this IntersectionMatrix is T*F**F***
         * *TF**F*** or **FT*F***
         * or **F*TF***
         *
         * @return true if the first Geometry is covered by
         * the second
        """
        return ((self._matches(self.matrix[Location.INTERIOR][Location.INTERIOR], 'T') or
                self._matches(self.matrix[Location.INTERIOR][Location.BOUNDARY], 'T') or
                self._matches(self.matrix[Location.BOUNDARY][Location.INTERIOR], 'T') or
                self._matches(self.matrix[Location.BOUNDARY][Location.BOUNDARY], 'T')) and
            self.matrix[Location.INTERIOR][Location.EXTERIOR] == Dimension.FALSE and
            self.matrix[Location.BOUNDARY][Location.EXTERIOR] == Dimension.FALSE)

    def transpose(self):
        """
         * Transposes this IntersectionMatrix.
         *
         * @return this IntersectionMatrix as a convenience.
         *
         * \todo It returns 'this' pointer so why not to return const-pointer?
         * \todo May be it would be better to return copy of transposed matrix?
        """
        temp = self.matrix[1][0]
        self.matrix[1][0] = self.matrix[0][1]
        self.matrix[0][1] = temp
        temp = self.matrix[2][0]
        self.matrix[2][0] = self.matrix[0][2]
        self.matrix[0][2] = temp
        temp = self.matrix[2][1]
        self.matrix[2][1] = self.matrix[1][2]
        self.matrix[1][2] = temp
        return self

    def __str__(self) -> str:
        """
         * Returns a nine-character String representation of this
         * IntersectionMatrix.
         *
         * @return the nine dimension symbols of this IntersectionMatrix
         * in row-major order.
        """
        _matrix = self.matrix
        result = ''
        for i in range(self.firstDim):
            for j in range(self.secondDim):
                result = result + Dimension.toDimensionSymbol(_matrix[i][j])
        return result


class Coordinate():
    """
    * Coordinate is the lightweight class used to store coordinates.
    *
    * It is distinct from Point, which is a subclass of Geometry.
    * Unlike objects of type Point (which contain additional
    * information such as an envelope, a precision model, and spatial
    * reference system information), a Coordinate only contains
    * ordinate values and accessor methods.
    *
    * Coordinate objects are two-dimensional points, with an additional
    * z-ordinate. JTS does not support any operations on the z-ordinate except
    * the basic accessor functions.
    *
    * Constructed coordinates will have a z-ordinate of DoubleNotANumber.
    * The standard comparison functions will ignore the z-ordinate.
    *
    """
    def __init__(self, x: float=0, y: float=0, z: float=0):
        self.x = x
        self.y = y
        self.z = z

    def compareTo(self, other) -> int:
        if self.x < other.x:
            return -1
        if self.x > other.x:
            return 1
        if self.y < other.y:
            return -1
        if self.y > other.y:
            return 1
        return 0
    
    @property
    def length(self):
        return sqrt(self.x ** 2 + self.y ** 2)
        
    def distance(self, other):
        """ 2d distance """
        dx = self.x - other.x
        dy = self.y - other.y
        return sqrt(dx ** 2 + dy ** 2)

    def equals2D(self, other):
        if self.x != other.x or self.y != other.y:
            return False
        return True

    def clone(self):
        return Coordinate(self.x, self.y, self.z)

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other) -> bool:
        return self.equals2D(other)

    def __ne__(self, other) -> bool:
        return not self.equals2D(other)

    def __gt__(self, other):
        return self.compareTo(other) > 0

    def __lt__(self, other):
        return self.compareTo(other) < 0

    def __le__(self, other):
        return self.compareTo(other) <= 0

    def __ge__(self, other):
        return self.compareTo(other) >= 0

    def dot(self, other) -> float:
        """ dot product """
        return self.x * other.x + self.y * other.y + self.z * other.z

    def __mul__(self, scalar):
        if type(scalar).__name__ == 'Coordinate':
            return self.dot(scalar)
        return Coordinate(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar):
        if scalar == 0:
            raise ValueError("Division by Zero")
        return Coordinate(self.x / scalar, self.y / scalar, self.z / scalar)

    def __add__(self, other):
        return Coordinate(self.x + other.x, self.y + other.y, self.z + other.z)

    """    
    def __iadd__(self, other):
        self.x += other.x
        self.y += other.y
        return self
    """    
    
    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return Coordinate(self.x - other.x, self.y - other.y, self.z - other.z)
    
    """
    def __isub__(self, other):
        self.x -= other.x
        self.y -= other.y
        return self
    """
    
    def __rsub__(self, other):
        return self.__sub__(other)
    
    """
    # this enable [] access to make coords "shapely compatible"
    def __getitem__(self, index: int) -> float:
        if index > 2:
            raise ValueError("Coordinate has only 3 values")
        return [self.x, self.y, self.z][index]

    def __setitem__(self, index: int, value: float) -> None:
        if index > 2:
            raise ValueError("Coordinate has only 3 values")
        [self.x, self.y, self.z][index] = value
    """
    
    def __len__(self) -> int:
        return 3

    def __str__(self) -> str:
        return "({0:.12f}, {1:.12f})".format(self.x, self.y)


class Triangle():
    """
     * Represents a planar triangle, and provides methods for calculating various
     * properties of triangles.
    """
    def __init__(self, p0, p1, p2):
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2

    def inCentre(self, coord) -> None:
        """
         * The inCentre of a triangle is the point which is equidistant
         * from the sides of the triangle.  This is also the point at which the bisectors
         * of the angles meet.
         *
         * @param resultPoint the point into which to write the inCentre of the triangle
        """
        len0 = self.p1.distance(self.p2)
        len1 = self.p0.distance(self.p2)
        len2 = self.p0.distance(self.p1)
        circum = len0 + len1 + len2
        coord.x = (len0 * self.p0.x + len1 * self.p1.x + len2 * self.p2.x) / circum
        coord.y = (len0 * self.p0.y + len1 * self.p1.y + len2 * self.p2.y) / circum

    def circumcentre(self, coord)-> None:
        """
         * Computes the circumcentre of a triangle. The circumcentre is the centre of
         * the circumcircle, the smallest circle which encloses the triangle. It is
         * also the common intersection point of the perpendicular bisectors of the
         * sides of the triangle, and is the only point which has equal distance to
         * all three vertices of the triangle.
         * <p>
         * The circumcentre does not necessarily lie within the triangle. For example,
         * the circumcentre of an obtuse isoceles triangle lies outside the triangle.
         * <p>
         * This method uses an algorithm due to J.R.Shewchuk which uses normalization
         * to the origin to improve the accuracy of computation. (See <i>Lecture Notes
         * on Geometric Robustness</i>, Jonathan Richard Shewchuk, 1999).
         *
         * @param resultPoint the point into which to write the inCentre of the triangle
        """
        cx = self.p2.x
        cy = self.p2.y
        ax = self.p0.x - cx
        ay = self.p0.y - cy
        bx = self.p1.x - cx
        by = self.p1.y - cy

        denom = 2 * self.det(ax, ay, bx, by)
        numx = self.det(ay, ax * ax + ay * ay, by, bx * bx + by * by)
        numy = self.det(ax, ax * ax + ay * ay, bx, bx * bx + by * by)
        coord.x = cx - numx / denom
        coord.y = cy + numy / denom

    def det(self, m00: float, m01: float, m10: float, m11: float) -> float:
        """
         * Computes the determinant of a 2x2 matrix. Uses standard double-precision
         * arithmetic, so is susceptible to round-off error.
         *
         * @param m00
         *          the [0,0] entry of the matrix
         * @param m01
         *          the [0,1] entry of the matrix
         * @param m10
         *          the [1,0] entry of the matrix
         * @param m11
         *          the [1,1] entry of the matrix
         * @return the determinant
        """
        return m00 * m11 - m01 * m10


class CoordinateSequence(list):
    """
     * CoordinateSequence
     *
     * The internal representation of a list of coordinates inside a Geometry.
     *
     * There are some cases in which you might want Geometries to store their
     * points using something other than the GEOS Coordinate class. For example, you
     * may want to experiment with another implementation, such as an array of Xs
     * and an array of Ys. or you might want to use your own coordinate class, one
     * that supports extra attributes like M-values.
     *
     * You can do this by implementing the CoordinateSequence and
     * CoordinateSequenceFactory interfaces. You would then create a
     * GeometryFactory parameterized by your CoordinateSequenceFactory, and use
     * this GeometryFactory to create new Geometries. All of these new Geometries
     * will use your CoordinateSequence implementation.
     *
    """
    def __init__(self, coords=None, allowRepeated: bool=True, direction: bool=True):
        list.__init__(self)
        if coords is not None:
            self.add(coords, allowRepeated, direction)

    def _hasRepeatedPoints(self) -> bool:
        return CoordinateSequence.hasRepeatedPoints(self)

    @property
    def is_empty(self) -> bool:
        return len(self) == 0

    @staticmethod
    def _removeRepeatedPoints(coords):
        return [c for i, c in enumerate(coords) if i == 0 or coords[i] != coords[i - 1]]

    def add(self, coords, allowRepeated: bool=True, direction: bool=True) -> bool:
        """
         *  Add an array of coordinates
         *
         *  @param cl The coordinates
         *
         *  @param allowRepeated
         *  if set to false, repeated coordinates are collapsed
         *
         *  @param direction if false, the array is added in reverse order
         *
         *  @return true (as by general collection contract)
        """

        if type(coords).__name__ == 'Coordinate':

            if not allowRepeated and coords == self[-1]:
                return False

            self.append(coords)
            return True

        if direction:
            _c = coords
        else:
            _c = list(reversed(coords))

        if not allowRepeated:
            _c = CoordinateSequence._removeRepeatedPoints(_c)

        self.extend(_c)
        return True

    def setPoints(self, coords) -> None:
        self.clear()
        self.add(coords)

    def applyCoordinateFilter(self, f) -> None:
        for c in self:
            f.filter(c)

    @staticmethod
    def increasingDirection(coords) -> int:
        ncoords = len(coords)
        for i in range(int(ncoords / 2)):
            j = ncoords - 1 - i
            # skip equal points on both ends
            comp = coords[i].compareTo(coords[j])
            if comp != 0:
                return comp
        # array must be a palindrome - defined to be in positive direction
        return 1

    @staticmethod
    def removeRepeatedPoints(coords):
        return CoordinateSequence(CoordinateSequence._removeRepeatedPoints(coords))

    @staticmethod
    def hasRepeatedPoints(coords) -> bool:
        for i in range(1, len(coords)):
            if coords[i - 1] == coords[i]:
                return True
        return False

    def expandEnvelope(self, env) -> None:
        for coord in self:
            env.expandToInclude(coord)

    @staticmethod
    def minCoordinate(coords):
        minCoord = None
        for coord in coords:
            if minCoord is None or minCoord.compareTo(coord) > 0:
                minCoord = coord
        return minCoord

    @staticmethod
    def scroll(coords, firstCoord) -> None:

        start = coords.index(firstCoord)
        if start < 1:
            return
        end = len(coords)
        coords.setPoints(coords[start:end] + coords[0:start])

    def apply_ro(self, filter):
        """
         * Apply a fiter to each Coordinate of this sequence.
         * The filter is expected to provide a .filter(Coordinate&)
         * method.
        """
        for coord in self:
            filter.filter_ro(coord)

    def apply_rw(self, filter):
        """
         * Apply a fiter to each Coordinate of this sequence.
         * The filter is expected to provide a .filter(Coordinate&)
         * method.
        """
        for coord in self:
            filter.filter_rw(coord)

    def clone(self):
        return CoordinateSequence([co.clone() for co in self])

    @staticmethod
    def equals(cs1, cs2) -> bool:
        if cs1 is cs2:
            return True
        if cs1 is None or cs2 is None:
            return False
        if len(cs1) != len(cs2):
            return False
        for i, coord in enumerate(cs1):
            if cs2[i] != coord:
                return False
        return True

    @staticmethod
    def equals_unoriented(cs1, cs2) -> bool:
        """
         * compare two sequences for equalness
         * the coordinates of e1 are the same or the reverse of the coordinates in e2
        """
        if cs1 is cs2:
            return True

        if cs1 is None or cs2 is None:
            return False

        nCoords = len(cs1)

        if nCoords != len(cs2):
            return False

        isEqualForward = True
        isEqualReverse = True
        nCoords -= 1
        for i, coord in enumerate(cs1):
            if coord != cs2[i]:
                isEqualForward = False
            if coord != cs2[nCoords - i]:
                isEqualReverse = False
            if (not isEqualForward) and (not isEqualReverse):
                return False
        return True

    def almost_equals(self, other, tolerance):
        if self is other:
            return True
        if other is None:
            return False
        if len(self) != len(other):
            return False
        for i, coord in enumerate(self):
            if other[i].distance(coord) > tolerance:
                return False
        return True

    def __eq__(self, other) -> bool:
        return CoordinateSequence.equals(self, other)

    def __ne__(self, other) -> bool:
        return not CoordinateSequence.equals(self, other)

    def __str__(self) -> str:
        return "({})".format(", ".join([str(c) for c in self]))

    def printReverse(self) -> str:
        return "({})".format(", ".join([str(c) for c in self[::-1]]))


class CoordinateFilter():
    """
     * Geometry classes support the concept of applying a
     * coordinate filter to every coordinate in the Geometry.
     *
     * A  coordinate filter can either record information about each coordinate or
     * change the coordinate in some way. Coordinate filters implement the
     * interface CoordinateFilter. (CoordinateFilter is
     * an example of the Gang-of-Four Visitor pattern). Coordinate filters can be
     * used to implement such things as coordinate transformations, centroid and
     * envelope computation, and many other functions.
     *
     * TODO: provide geom.CoordinateInspector and geom.CoordinateMutator instead
     * of having the two versions of filter_rw and filter_ro
     *
    """
    def filter_ro(self, coord) -> None:
        """
         * Performs an operation on <code>coord</code>.
         *
         * @param  coord  a <code>Coordinate</code> to which the filter is applied.
        """
        raise NotImplementedError()

    def filter_rw(self, coord) -> None:
        """
         * Performs an operation on <code>coord</code>.
         *
         * @param  coord  a <code>Coordinate</code> to which the filter is applied.
        """
        raise NotImplementedError()


class CoordinateSequenceFilter():
    """
     *  Interface for classes which provide operations that
     *  can be applied to the coordinates in a {CoordinateSequence}.
     *  A CoordinateSequence filter can either record information about each
     *  coordinate or change the coordinate in some way.
     *  CoordinateSequence filters can be
     *  used to implement such things as coordinate transformations, centroid and
     *  envelope computation, and many other functions.
     *  For maximum efficiency, the execution of filters can be short-circuited.
     *  {Geometry} classes support the concept of applying a
     *  CoordinateSequenceFilter to each
     *  {CoordinateSequence}s they contain.
     *  <p>
     *  CoordinateSequenceFilter is
     *  an example of the Gang-of-Four Visitor pattern.
     *
     * @see Geometry.apply_ro(CoordinateSequenceFilter)
     * @see Geometry.apply_rw(CoordinateSequenceFilter)
     * @author Martin Davis
     *
    """
    def filter_ro(self, coord) -> None:
        """
         * Performs an operation on <code>coord</code>.
         *
         * @param  coord  a <code>Coordinate</code> to which the filter is applied.
        """
        raise NotImplementedError()

    def filter_rw(self, coord) -> None:
        """
         * Performs an operation on <code>coord</code>.
         *
         * @param  coord  a <code>Coordinate</code> to which the filter is applied.
        """
        raise NotImplementedError()


class GeometryFilter():
    """
     *  Geometry classes support the concept of applying
     *  a GeometryComponentFilter
     *  filter to the Geometry.
     *  The filter is applied to every component of the Geometry
     *  which is itself a Geometry.
     *  A GeometryComponentFilter filter can either
     *  record information about the Geometry
     *  or change the Geometry in some way.
     *  GeometryComponentFilter
     *  is an example of the Gang-of-Four Visitor pattern.
    """
    def filter_ro(self, coord) -> None:
        """
         * Performs an operation on <code>coord</code>.
         *
         * @param  coord  a <code>Coordinate</code> to which the filter is applied.
        """
        raise NotImplementedError()

    def filter_rw(self, coord) -> None:
        """
         * Performs an operation on <code>coord</code>.
         *
         * @param  coord  a <code>Coordinate</code> to which the filter is applied.
        """
        raise NotImplementedError()


class GeometryComponentFilter():
    """
     *  Geometry classes support the concept of applying
     *  a GeometryComponentFilter
     *  filter to the Geometry.
     *  The filter is applied to every component of the Geometry
     *  which is itself a Geometry.
     *  A GeometryComponentFilter filter can either
     *  record information about the Geometry
     *  or change the Geometry in some way.
     *  GeometryComponentFilter
     *  is an example of the Gang-of-Four Visitor pattern.
    """
    def filter_ro(self, coord) -> None:
        """
         * Performs an operation on <code>coord</code>.
         *
         * @param  coord  a <code>Coordinate</code> to which the filter is applied.
        """
        raise NotImplementedError()

    def filter_rw(self, coord) -> None:
        """
         * Performs an operation on <code>coord</code>.
         *
         * @param  coord  a <code>Coordinate</code> to which the filter is applied.
        """
        raise NotImplementedError()


class LinearComponentExtracter(GeometryComponentFilter):
    def __init__(self, comps):
        self.comps = comps

    @staticmethod
    def getLines(geom, ret):
        lce = LinearComponentExtracter(ret)
        geom.apply_ro(lce)

    def filter_rw(self, geom) -> None:
        if (geom.type_id == GeomTypeId.GEOS_LINESTRING or
                geom.type_id == GeomTypeId.GEOS_LINEARRING):
            self.comps.append(geom)

    def filter_ro(self, geom) -> None:
        if (geom.type_id == GeomTypeId.GEOS_LINESTRING or
                geom.type_id == GeomTypeId.GEOS_LINEARRING):
            self.comps.append(geom)


class Extracter(GeometryComponentFilter):
    """
     * Constructs a filter with a list in which to store the elements found.
     *
     * @param comps the container to extract into (will push_back to it)
    """
    def __init__(self, type_id: int, comps: list):
        self.comps = comps
        self.type_id = type_id

    def filter_ro(self, geom) -> None:
        if geom.type_id == self.type_id:
            self.comps.append(geom)

    def filter_rw(self, geom) -> None:
        if geom.type_id == self.type_id:
            self.comps.append(geom)


class PolygonExtracter(Extracter):
    def __init__(self, comps: list):
        Extracter.__init__(self, GeomTypeId.GEOS_POLYGON, comps)

    @staticmethod
    def getPolygons(geom, ret: list) -> None:
        pe = PolygonExtracter(ret)
        geom.apply_ro(pe)


class GeometryExtracter():
    """
     * Extracts the components of a given type from a {@link Geometry}.
    """

    @staticmethod
    def extract(type_id: int, geom, lst: list) -> None:
        """
         * Extracts the components of type <tt>clz</tt> from a {@link Geometry}
         * and adds them to the provided container.
         *
         * @param geom the geometry from which to extract
         * @param list the list to add the extracted elements to
        """
        if geom.type_id == type_id:
            lst.append(geom)
        elif geom.type_id in (
                GeomTypeId.GEOS_MULTIPOLYGON,
                GeomTypeId.GEOS_MULTILINESTRING,
                GeomTypeId.GEOS_MULTIPOINT,
                GeomTypeId.GEOS_GEOMETRYCOLLECTION
                ):
            extracter = Extracter(type_id, lst)
            geom.apply_ro(extracter)


class ComponentCoordinateExtracter(GeometryComponentFilter):
    """
     * Extracts a single representative {@link Coordinate}
     * from each connected component of a {@link Geometry}.
    """
    def __init__(self, coords: list):
        """
         * Constructs a ComponentCoordinateFilter with a list in which
         * to store Coordinates found.
        """
        GeometryComponentFilter.__init__(self)
        self.comps = coords

    @staticmethod
    def getCoordinates(geom, coords: list) -> None:
        """
         * Push the linear components from a single geometry into
         * the provided vector.
         * If more than one geometry is to be processed, it is more
         * efficient to create a single ComponentCoordinateFilter instance
         * and pass it to multiple geometries.
        """
        cce = ComponentCoordinateExtracter(coords)
        geom.apply_ro(cce)

    def filter_rw(self, geom) -> None:
        if (geom.type_id == GeomTypeId.GEOS_LINEARRING or
               geom.type_id == GeomTypeId.GEOS_LINESTRING or
               geom.type_id == GeomTypeId.GEOS_POINT):
            self.comps.append(geom.coord)

    def filter_ro(self, geom) -> None:
        if (geom.type_id == GeomTypeId.GEOS_LINEARRING or
               geom.type_id == GeomTypeId.GEOS_LINESTRING or
               geom.type_id == GeomTypeId.GEOS_POINT):
            self.comps.append(geom.coord)


class GeometryEditorOperation():
    """
     * A interface which specifies an edit operation for Geometries.
    """
    def edit(self, factory):
        """
         * Edits a Geometry by returning a new Geometry with a modification.
         * The returned Geometry might be the same as the Geometry passed in.
         *
         * @param geometry the Geometry to modify
         *
         * @param factory the factory with which to construct the modified
         *                Geometry (may be different to the factory of the
         *                input geometry)
         *
         * @return a new Geometry which is a modification of the input Geometry
        """
        raise NotImplementedError()


class CoordinateOperation(GeometryEditorOperation):
    """
     * A GeometryEditorOperation which modifies the coordinate list of a
     * Geometry.
     * Operates on Geometry subclasses which contains a single coordinate list.
    """
    def edit(self, geom, factory):
        """
         * Return a newly created geometry, ownership to caller
        """
        if geom.type_id == GeomTypeId.GEOS_LINEARRING:
            newCoords = self._edit(geom.coords, geom)
            return factory.createLinearRing(newCoords)

        elif geom.type_id == GeomTypeId.GEOS_LINESTRING:
            newCoords = self._edit(geom.coords, geom)
            return factory.createLineString(newCoords)

        elif geom.type_id == GeomTypeId.GEOS_POINT:
            newCoords = self._edit(geom.coords, geom)
            return factory.createPoint(newCoords)

        return geom.clone()

    def _edit(self, coords, geom):
        """
         * Edits the array of Coordinate from a Geometry.
         *
         * @param coordinates the coordinate array to operate on
         * @param geometry the geometry containing the coordinate list
         * @return an edited coordinate array (which may be the same as
         *         the input)
        """
        raise NotImplementedError()


class ShortCircuitedGeometryVisitor():
    def __init__(self):
        self.done = False
        
    def applyTo(self, geom):
        for i in range(geom.numgeoms):
            element = geom.getGeometryN(i)
            if element.type_id in [
                    GeomTypeId.GEOS_GEOMETRYCOLLECTION,
                    GeomTypeId.GEOS_MULTIPOLYGON,
                    GeomTypeId.GEOS_MULTILINESTRING,
                    GeomTypeId.GEOS_MULTIPOINT
                    ]:
                self.applyTo(element)
            else:
                self.visit(element)
                if self.isDone():
                    self.done = True
            if self.done:
                return
            

class GeometryEditor():
    """
     * Supports creating a new Geometry which is a modification of an existing one.
     * Geometry objects are intended to be treated as immutable.
     * This class allows you to "modify" a Geometry
     * by traversing it and creating a new Geometry with the same overall
     * structure but possibly modified components.
     *
     * The following kinds of modifications can be made:
     *
     * - the values of the coordinates may be changed.
     *   Changing coordinate values may make the result Geometry invalid;
     *   this is not checked by the GeometryEditor
     * - the coordinate lists may be changed
     *   (e.g. by adding or deleting coordinates).
     *   The modifed coordinate lists must be consistent with their original
     *   parent component
     *   (e.g. a LinearRing must always have at least 4 coordinates, and the
     *   first and last coordinate must be equal)
     * - components of the original geometry may be deleted
     *   (e.g. holes may be removed from a Polygon, or LineStrings removed
     *   from a MultiLineString). Deletions will be propagated up the component
     *   tree appropriately.
     *
     * Note that all changes must be consistent with the original Geometry's
     * structure
     * (e.g. a Polygon cannot be collapsed into a LineString).
     *
     * The resulting Geometry is not checked for validity.
     * If validity needs to be enforced, the new Geometry's isValid should
     * be checked.
     *
     * @see Geometry::isValid
     *
    """
    def __init__(self, factory=None):
        """
         * Creates a new GeometryEditor object which will create
         * an edited Geometry with the same GeometryFactory as the
         * input Geometry.
         * @param factory the GeometryFactory to create the edited Geometry with
        """

        """
         * The factory used to create the modified Geometry
        """
        # GeometryFactory
        self._factory = factory

    def editPolygon(self, polygon, operation):
        newPolygon = operation.edit(polygon, self._factory)
        if newPolygon.is_empty:
            if newPolygon._factory != self._factory:
                return self._factory.createPolygon(None, None)
            else:
                return newPolygon
        # LinearRing
        shell = self.edit(newPolygon.exterior, operation)
        if shell.is_empty:
            return self._factory.createPolygon(None, None)

        holes = []
        for hole in newPolygon.interiors:
            newHole = self.edit(hole, operation)
            if newHole.is_empty:
                continue
            else:
                holes.append(newHole)

        return self._factory.createPolygon(shell, holes)

    def editGeometryCollection(self, collection, operation):
        newColl = operation.edit(collection, self._factory)

        geoms = []
        for geom in newColl.geoms:
            newGeom = self.edit(geom, operation)
            if newGeom is None or newGeom.is_empty:
                continue
            else:
                geoms.append(newGeom)

        if newColl.type_id == GeomTypeId.GEOS_MULTIPOINT:
            return self._factory.createMultiPoint(geoms)

        elif newColl.type_id == GeomTypeId.GEOS_MULTILINESTRING:
            return self._factory.createMultiLineString(geoms)

        elif newColl.type_id == GeomTypeId.GEOS_MULTIPOLYGON:
            return self._factory.createMultiPolygon(geoms)

        else:
            return self._factory.createGeometryCollection(geoms)

    def edit(self, geom, operation):
        """
         * Edit the input Geometry with the given edit operation.
         * Clients will create subclasses of GeometryEditorOperation or
         * CoordinateOperation to perform required modifications.
         *
         * @param geometry the Geometry to edit
         * @param operation the edit operation to carry out
         * @return a new Geometry which is the result of the editing
         *
        """
        # if client did not supply a GeometryFactory, use the one from the input Geometry
        if self._factory is None:
            self._factory = geom._factory

        if geom.type_id in [
                GeomTypeId.GEOS_GEOMETRYCOLLECTION,
                GeomTypeId.GEOS_MULTILINESTRING,
                GeomTypeId.GEOS_MULTIPOINT,
                GeomTypeId.GEOS_MULTIPOLYGON]:
            return self.editGeometryCollection(geom, operation)

        elif geom.type_id == GeomTypeId.GEOS_POLYGON:
            return self.editPolygon(geom, operation)

        elif geom.type_id in [
                GeomTypeId.GEOS_LINEARRING,
                GeomTypeId.GEOS_LINESTRING,
                GeomTypeId.GEOS_POINT]:
            return operation.edit(geom, self._factory)

        else:
            raise ValueError("Unsupported geometry type {}".format(type(geom).__name__))
        return None
