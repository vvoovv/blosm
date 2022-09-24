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


from .simplify import TopologyPreservingSimplifier
from .algorithms import BoundaryNodeRule
from .shared import (
    logger,
    GeomTypeId,
    PrecisionModel,
    TopologyException
    )
from .precision import (
    CommonBitsRemover,
    GeometryPrecisionReducer
    )
from .op_overlay import (
    GeometrySnapper
    )
from .op_simple import IsSimpleOp
from .op_valid import IsValidOp, TopologyErrors


CBR_BEFORE_SNAPPING = True
GEOS_CHECK_VALIDITY = True


def check_valid(geom, label: str, doThrow: bool=False, validOnly: bool=False) -> bool:
    
    if not GEOS_CHECK_VALIDITY:
        return True
        
    if geom.type_id in [
            GeomTypeId.GEOS_LINESTRING,
            GeomTypeId.GEOS_LINEARRING,
            GeomTypeId.GEOS_MULTILINESTRING]:
            
        if not validOnly:
            # Lineal geoms
            sop = IsSimpleOp(geom, BoundaryNodeRule.getBoundaryEndPoint())
            if not sop.is_simple:
                logger.debug("%s is invalid geometry is not simple", label)
                if doThrow:
                    raise TopologyException("{} is not simple".format(label))
                return False
    
    else:
        ivo = IsValidOp(geom)
        if not ivo.is_valid:
            logger.debug("%s is invalid %s", label, ivo.validErr)
            if doThrow:
                raise TopologyException("{} is invalid".format(label))
            return False

    return True


def fix_self_intersections(geom, label: str):
    # Only multi-components can be fixed by UnaryUnion
    if geom.type_id < 4:
        return geom

    ivo = IsValidOp(geom)
    # poygon is valid, nothing to do
    if ivo.is_valid:
        return geom
    
    # Not all invalidities can be fixed by this code
    if ivo.validErr.errorType in [
            TopologyErrors.eRingSelfIntersection,
            TopologyErrors.eTooFewPoints]:
        logger.debug("ATTEMPT_TO_FIX: %s", ivo.validErr)
        geom = geom.union()
        logger.debug("ATTEMPT_TO_FIX: %s succeeded", ivo.validErr)
        return geom
    
    logger.debug("invalidity detected: %s", ivo.validErr)
    
    return geom


def SnapOp(geom0, geom1, _Op):
    optype = type(_Op).__name__
    snapTolerance = GeometrySnapper.computeOverlaySnapTolerance(geom0, geom1)

    if CBR_BEFORE_SNAPPING:
        cbr = CommonBitsRemover()
        cbr.add(geom0)
        cbr.add(geom1)
        rG0 = cbr.removeCommonBits(geom0.clone())
        rG1 = cbr.removeCommonBits(geom1.clone())
    else:
        rG0, rG1 = geom0, geom1

    snapper0 = GeometrySnapper(rG0)
    snapG0 = snapper0.snapTo(rG1, snapTolerance)

    snapper1 = GeometrySnapper(rG1)
    snapG1 = snapper1.snapTo(snapG0, snapTolerance)

    result = _Op.execute(snapG0, snapG1)
    check_valid(result, "{}: result (before common-bits addition)".format(optype))

    if CBR_BEFORE_SNAPPING:
        cbr.addCommonBits(result)

    return result


def BinaryOp(geom0, geom1, _Op):

    origException = None
    optype = type(_Op).__name__
    
    try:
        res = _Op.execute(geom0, geom1)
        check_valid(res, "{} Overlay result between original inputs".format(optype), True, True)
        logger.debug("%s Attempt with original input succeeded", optype)
        return res
    except TopologyException as ex:
        logger.warning("%s Attempt with original input failed : %s", optype, ex)
        origException = ex
        # geom0._factory.output([geom0, geom1], name="failing", multiple=True)
        # if ex.coord is not None:
        #    geom0._factory.outputCoord(ex.coord, name=str(ex))
        pass

    check_valid(geom0, "{} Input geom 0".format(optype), True, True)
    check_valid(geom1, "{} Input geom 1".format(optype), True, True)

    # USE_COMMONBITS_POLICY
    logger.debug("%s Trying with CBR", optype)

    try:
        cbr = CommonBitsRemover()
        cbr.add(geom0)
        cbr.add(geom1)
        rg0 = cbr.removeCommonBits(geom0.clone())
        rg1 = cbr.removeCommonBits(geom1.clone())
        check_valid(rg0, "{} CBR: geom 0 (after common-bits removal)".format(optype))
        check_valid(rg1, "{} CBR: geom 1 (after common-bits removal)".format(optype))

        ret = _Op.execute(rg0, rg1)
        check_valid(ret, "{} CBR: result (before common-bits addition)".format(optype))

        cbr.addCommonBits(ret)
        check_valid(ret, "{} CBR: result (after common-bits addition)".format(optype), True)
        logger.info("%s CBR succeeded", optype)
        return ret
        
    except TopologyException as ex:
        logger.warning("%s Attempt with CBR failed %s", optype, ex)
        # if ex.coord is not None:
        #    geom0._factory.outputCoord(ex.coord, name=str(ex))
        pass

    # USE_SNAPPING_POLICY
    logger.debug("%s Trying with snapping", optype)

    try:
        ret = SnapOp(geom0, geom1, _Op)
        check_valid(ret, "{}: result".format(optype), True, True)
        logger.info("%s SnapOp succeeded", optype)
        return ret
    except TopologyException as ex:
        logger.warning("%s Attempt with SnapOp failed %s", optype, ex)
        # if ex.coord is not None:
        #    geom0._factory.outputCoord(ex.coord, name=str(ex))
        pass

    # USE_PRECISION_REDUCTION_POLICY
    logger.debug("%s Trying with precision reduction", optype)

    try:
        g0scale = geom0._factory.precisionModel.scale
        g1scale = geom1._factory.precisionModel.scale

        logger.debug("%s Original input scales are %s and %s", optype, g0scale, g1scale)
        maxScale = 1e16
        # Don't use a scale biffer than the input one
        if g0scale > 0 and g0scale < maxScale:
            maxScale = g0scale
        if g1scale > 0 and g1scale < maxScale:
            maxScale = g1scale
        scale = maxScale
        while scale >= 1:
            pm = PrecisionModel(scale=scale)
            gf = geom0._factory.clone(pm)

            logger.debug("%s Trying with scale %s", optype, scale)
            reducer = GeometryPrecisionReducer(geometryFactory=gf)
            rg0 = reducer._reduce(geom0)
            rg1 = reducer._reduce(geom1)

            check_valid(rg0, "{} PR: geom 0 (after precision reduction)".format(optype))
            check_valid(rg1, "{} PR: geom 1 (after precision reduction)".format(optype))
            try:
                ret = _Op.execute(rg0, rg1)
                if geom0._factory.precisionModel.compareTo(geom1._factory.precisionModel) < 0:
                    ret = geom0._factory.createGeometry(ret)
                else:
                    ret = geom1._factory.createGeometry(ret)

                check_valid(ret, "{} PR: result (after restore of original precision)".format(optype), True)
                logger.info("%s Attempt with scale %s succeded", optype, scale)
                
                return ret
                
            except TopologyException as ex:
                logger.debug("%s Attempt with reduced scale %s failed %s", optype, scale, ex)
                if scale == 1:
                    raise ex
                pass

            scale /= 10.0
    except TopologyException as ex:
        logger.warning("%s Attempt with precision reduction failed %s", optype, ex)
        # if ex.coord is not None:
        #    geom0._factory.outputCoord(ex.coord, name=str(ex))
        pass
    
    # USE_TP_SIMPLIFY_POLICY
    logger.debug("%s Trying with simplify", optype)

    try:
        maxTolerance = 0.04
        minTolerance = 0.01
        tolStep = 0.01
        tol = minTolerance
        while tol <= maxTolerance:
            logger.debug("%s Trying simplifying with tolerance %s", optype, tol)
            rg0 = TopologyPreservingSimplifier.simplify(geom0, tol)
            rg1 = TopologyPreservingSimplifier.simplify(geom1, tol)
            try:
                ret = _Op.execute(rg0, rg1)
                logger.info("%s Attempt simplified with tolerance (%s) %s succeeded", optype, tol, ex)
                return ret
            except TopologyException as ex:
                logger.debug("%s Attempt simplified with tolerance (%s) %s", optype, tol, ex)
                if tol >= maxTolerance:
                    raise ex
                pass
            tol += tolStep

    except TopologyException as ex:
        logger.warning("%s Attempt with simplified failed %s", optype, ex)
        # if ex.coord is not None:
        #    geom0._factory.outputCoord(ex.coord, name=str(ex))
        pass
    
    logger.error("%s No attempt worked to union", optype)

    raise origException
