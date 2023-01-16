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
    Coordinate,
    GeomTypeId,
    CoordinateFilter,
    CoordinateOperation,
    GeometryEditor
    )
from .index_quadtree import DoubleBits


class CommonBits():
    """
     * Determines the maximum number of common most-significant
     * bits in the mantissa of one or numbers.
     *
     * Can be used to compute the double-precision number which
     * is represented by the common bits.
     * If there are no common bits, the number computed is 0.0.
     *
    """
    
    def __init__(self):
        self.isFirst = True
        self.common = 0
        
    def add(self, num: float) -> None:
        
        if self.isFirst:
            self.common = num
            self.isFirst = False
            return
        
        self.common = DoubleBits.maximumCommonMantissa(self.common, num)
        

class Translater(CoordinateFilter):

    def __init__(self, trans):
        CoordinateFilter.__init__(self)
        self.trans = trans

    def filter_ro(self, coord) -> None:
        pass

    def filter_rw(self, coord) -> None:
        coord.x += self.trans.x
        coord.y += self.trans.y


class CommonCoordinateFilter(CoordinateFilter):
    def __init__(self):
        CoordinateFilter.__init__(self)
        self.commonBitsX = CommonBits()
        self.commonBitsY = CommonBits()

    def filter_ro(self, coord) -> None:
        self.commonBitsX.add(coord.x)
        self.commonBitsY.add(coord.y)

    def filter_rw(self, coord) -> None:
        pass

    @property
    def common(self):
        return Coordinate(self.commonBitsX.common, self.commonBitsY.common)


class CommonBitsRemover():
    """
     * Allow computing and removing common mantissa bits from one or
     * more Geometries.
    """
    def __init__(self):
        # Coordinate
        self.common = None
        self._disabled = False
        self.ccFilter = CommonCoordinateFilter()

    def add(self, geom) -> None:
        """
         * Add a geometry to the set of geometries whose common bits are
         * being computed.  After this method has executed the
         * common coordinate reflects the common bits of all added
         * geometries.
         *
         * @param geom a Geometry to test for common bits
        """
        if self._disabled:
            return
        geom.apply_ro(self.ccFilter)
        self.common = self.ccFilter.common
        self._disabled = self._disabled or (self.common.x == 0.0 and self.common.y == 0.0)
        
    def removeCommonBits(self, geom):
        """
         * Removes the common coordinate bits from a Geometry.
         * The coordinates of the Geometry are changed.
         *
         * @param geom the Geometry from which to remove the common
         *             coordinate bits
         * @return the shifted Geometry
        """
        # 
        if self._disabled:
            return geom
        invCoord = Coordinate(-self.common.x, -self.common.y)
        trans = Translater(invCoord)
        geom.apply_rw(trans)
        geom.geometryChanged()
        return geom

    def addCommonBits(self, geom):
        """
         * Adds the common coordinate bits back into a Geometry.
         * The coordinates of the Geometry are changed.
         *
         * @param geom the Geometry to which to add the common coordinate bits
         * @return the shifted Geometry
        """
        if self._disabled:
            return geom
        trans = Translater(self.common)
        geom.apply_rw(trans)
        geom.geometryChanged()
        return geom


class PrecisionReducerCoordinateOperation(CoordinateOperation):

    def __init__(self, precisionModel, removeCollapsed):
        self.precisionModel = precisionModel
        self.removeCollapsed = removeCollapsed

    def _edit(self, coords, geom):
        if len(coords) == 0:
            return None
        vc = []
        for coord in coords:
            newCoord = Coordinate(coord.x, coord.y, coord.z)
            self.precisionModel.makePrecise(newCoord)
            vc.append(newCoord)

        reducedCoords = geom._factory.coordinateSequenceFactory.create(vc)
        noRepeatedCoords = geom._factory.coordinateSequenceFactory.create(vc, False)

        """
         * Check to see if the removal of repeated points
         * collapsed the coordinate List to an invalid length
         * for the type of the parent geometry.
         * It is not necessary to check for Point collapses,
         * since the coordinate list can
         * never collapse to less than one point.
         * If the length is invalid, return the full-length coordinate array
         * first computed, or null if collapses are being removed.
         * (This may create an invalid geometry - the client must handle this.)
        """
        minLength = 0
        if geom.type_id == GeomTypeId.GEOS_LINESTRING:
            minLength = 2
        elif geom.type_id == GeomTypeId.GEOS_LINEARRING:
            minLength = 4

        if self.removeCollapsed:
            reducedCoords = None

        if len(noRepeatedCoords) < minLength:
            return reducedCoords

        return noRepeatedCoords


class GeometryPrecisionReducer():
    """
     * Reduces the precision of a {@link Geometry}
     * according to the supplied {@link PrecisionModel},
     * ensuring that the result is topologically valid.
    """
    def __init__(self, precisionModel=None, geometryFactory=None, preventTopologyCheck=False):
        """
         * Create a reducer that will change the precision model of the
         * new reduced Geometry
        """
        self._factory = geometryFactory
        if geometryFactory is not None and precisionModel is None:
            self.precisionModel = geometryFactory.precisionModel
        else:    
            self.precisionModel = precisionModel
            
        self.removeCollapsed = True
        self.isPointwise = False
        # prevent infinite loop
        self.preventTopologyCheck = preventTopologyCheck
        
    def _reducePointwise(self, geom):
        
        if self._factory is not None:
            geomEdit = GeometryEditor(self._factory)
        else:
            geomEdit = GeometryEditor()
        """
         * For polygonal geometries, collapses are always removed, in order
         * to produce correct topology
        """
        finalRemoveCollapsed = self.removeCollapsed
        
        if geom.dimension > 1:
            finalRemoveCollapsed = True

        prco = PrecisionReducerCoordinateOperation(self.precisionModel, finalRemoveCollapsed)
        return geomEdit.edit(geom, prco)
        
    def _reduce(self, geom):
        # dynamic import to prevent dep loop
        from .op_valid import IsValidOp
        
        res = self._reducePointwise(geom)

        if self.isPointwise:
            return res

        if res.type_id not in [
                GeomTypeId.GEOS_POLYGON,
                GeomTypeId.GEOS_MULTIPOLYGON]:
            return res
        
        # Geometry is polygonal - test if topology needs to be fixed
        ivo = IsValidOp(res)
        
        # poygon is valid, nothing to do
        if ivo.is_valid:
            return res
        """
        # Not all invalidities can be fixed by this code
        # attempt to fix self intersections for multipolygons
        if res.type_id == GeomTypeId.GEOS_MULTIPOLYGON and ivo.validErr.errorType in [
                TopologyErrors.eRingSelfIntersection,
                TopologyErrors.eTooFewPoints]:
            logger.debug("ATTEMPT_TO_FIX: %s", ivo.validErr)
            res = self.fixMultipolygonTopology(res)
            logger.debug("ATTEMPT_TO_FIX: %s succeeded", ivo.validErr)
            return res
        """    
        # hack to fix topology -> may lead to infinite loop
        return self.fixPolygonalTopology(res)
    
    def fixMultipolygonTopology(self, geom):
        logger.debug("GeometryPrecisionReducer.fixMultipolygonTopology() union() hack")
        geomToUnion = geom
        
        """
         * If precision model was *not* changed, need to flip
         * geometry to targetPM, buffer in that model, then flip back
        """
        if self._factory is None:
            tmpFactory = self.createFactory(geom._factory, self.precisionModel)
            geomToUnion = tmpFactory.createGeometry(geom)
        
        res = geomToUnion.union()
        
        if self._factory is None:
            res = geom._factory.createGeometry(geomToUnion)
            
        return res
   
    def fixPolygonalTopology(self, geom):
        logger.debug("GeometryPrecisionReducer.fixPolygonalTopology() buffer(0) hack")
        geomToBuffer = geom
        
        """
         * If precision model was *not* changed, need to flip
         * geometry to targetPM, buffer in that model, then flip back
        """
        if self._factory is None:
            tmpFactory = self.createFactory(geom._factory, self.precisionModel)
            geomToBuffer = tmpFactory.createGeometry(geom)
        
        # buffer may call this one back in infinite loop
        bufGeom = geomToBuffer.buffer(0)
        
        if self._factory is None:
            bufGeom = geom._factory.createGeometry(bufGeom)
            
        return bufGeom

    def createFactory(self, oldGF, newPM):
        return oldGF.clone(newPM)

    @staticmethod
    def reduce(geom, precisionModel=None, preventTopologyCheck=False):
        """
         * Convenience method for doing precision reduction
         * on a single geometry,
         * with collapses removed
         * and keeping the geometry precision model the same,
         * and preserving polygonal topology.
         *
         * @param g the geometry to reduce
         * @param precModel the precision model to use
         * @return the reduced geometry
        """
        reducer = GeometryPrecisionReducer(precisionModel=precisionModel, preventTopologyCheck=preventTopologyCheck)
        return reducer._reduce(geom)

    @staticmethod
    def reducePointwise(geom, precisionModel, preventTopologyCheck=False):
        """
         * Convenience method for doing precision reduction
         * on a single geometry,
         * with collapses removed
         * and keeping the geometry precision model the same,
         * but NOT preserving valid polygonal topology.
         *
         * @param g the geometry to reduce
         * @param precModel the precision model to use
         * @return the reduced geometry
        """
        reducer = GeometryPrecisionReducer(precisionModel=precisionModel, preventTopologyCheck=preventTopologyCheck)
        reducer.isPointwise = True
        return reducer._reduce(geom)
