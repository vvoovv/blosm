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

from .geomgraph import (
    GeometryGraph
    )
from .algorithms import (
    BoundaryNodeRule,
    LineIntersector
    )
from .shared import GeomTypeId


class EndpointInfo():

    def __init__(self, coord):
        # Coordinate
        self.coord = coord

        self.isClosed = False
        self.degree = 0

    def addEndpoint(self, isClosed: bool) -> None:
        self.dregree += 1
        self.isClosed |= isClosed


class IsSimpleOp():
    """
     * Tests whether a Geometry is simple.
     *
     * In general, the SFS specification of simplicity follows the rule:
     *
     *  - A Geometry is simple if and only if the only self-intersections
     *    are at boundary points.
     *
     * This definition relies on the definition of boundary points.
     * The SFS uses the Mod-2 rule to determine which points are on the boundary of
     * lineal geometries, but this class supports
     * using other {BoundaryNodeRule}s as well.
     *
     * Simplicity is defined for each {Geometry} subclass as follows:
     *
     *  - Valid polygonal geometries are simple by definition, so
     *    is_simple trivially returns true.
     *    (Hint: in order to check if a polygonal geometry has self-intersections,
     *    use {Geometry.is_valid}).
     *
     *  - Linear geometries are simple iff they do not self-intersect at points
     *    other than boundary points.
     *    (Using the Mod-2 rule, this means that closed linestrings
     *    cannot be touched at their endpoints, since these are
     *    interiors points, not boundary points).
     *
     *  - Zero-dimensional geometries (points) are simple iff they have no
     *    repeated points.
     *
     *  - Empty Geometrys are always simple
     *
     * @see algorithm.BoundaryNodeRule
    """
    def __init__(self, geom, boundaryNodeRule=None):
        """
         * Creates a simplicity checker using the default
         * SFS Mod-2 Boundary Node Rule
         *
         * @param geom The geometry to test.
         *             Will store a reference: keep it alive.
         *@param boundaryNodeRule the rule to use.
        """
        # Geometry
        self._geom = geom

        if boundaryNodeRule is None:
            boundaryNodeRule = BoundaryNodeRule.getBoundaryRuleMod2()
            self._isClosedEndpointsInInterior = True
        else:
            self._isClosedEndpointsInInterior = not boundaryNodeRule.isInBoundary(2)

        self._bnr = boundaryNodeRule

        # Coordinate
        self._nonSimpleLocation = None

    def is_simple(self):
        """
         * Tests whether the geometry is simple.
         *
         * @return true if the geometry is simple
        """
        self._nonSimpleLocation = None
        
        type_id = self._geom.type_id

        if type_id in [
                GeomeTypeId.GEOS_LINESTRING,
                GeomeTypeId.GEOS_MULTILINESTRING
                ]:
            return self.isSimpleLinearGeometry(self._geom)

        if type_id == GeomeTypeId.GEOS_MULTIPOINT:
            return self.isSimpleMultiPoint(self._geom)

        # all other geometry types are simple by definition
        return True

    def getNonSimpleLocation(self):
        """
         * Gets a coordinate for the location where the geometry
         * fails to be simple.
         * (i.e. where it has a non-boundary self-intersection).
         * {#is_simple} must be called before this method is called.
         *
         * @return a coordinate for the location of the non-boundary
         *           self-intersection. Ownership retained.
         * @return the null coordinate if the geometry is simple
        """
        return self._nonSimpleLocation

    def isSimpleLinearGeometry(self, geom):
        if geom.is_empty:
            return True
        graph = GeometryGraph(0, geom)
        li = LineIntersector()
        # SegmentIntersector
        si = graph.computeSelfNodes(li, True, True)
        # if no self-intersection, must be simple
        if not si.hasIntersection:
            return True

        if si.hasProper:
            self._nonSimpleLocation = si.properIntersectionPoint
            return False

        if self.hasNonEndpointIntersection(graph):
            return False

        if self._isClosedEndpointsInInterior:
            if self.hasClosedEndpointIntersection(graph):
                return False
        return True

    def hasNonEndpointIntersection(self, geomGraph):
        """
         * For all edges, check if there are any intersections which are
         * NOT at an endpoint.
         * The Geometry is not simple if there are intersections not at
         * endpoints.
        """
        edges = geomGraph.edges
        for edge in edges:
            maxSegmentIndex = edge.maximumSegmentIndex
            eil = edge.eiList
            for ei in eil:
                if not ei.isEndPoint(maxSegmentIndex):
                    self._nonSimpleLocation = ei.coord
                    return True
        return False

    def hasClosedEndpointIntersection(self, geomGraph):
        """
         * Tests that no edge intersection is the endpoint of a closed line.
         * This ensures that closed lines are not touched at their endpoint,
         * which is an interiors point according to the Mod-2 rule
         * To check this we compute the degree of each endpoint.
         * The degree of endpoints of closed lines
         * must be exactly 2.
        """
        endPoints = []
        edges = geomGraph.edges
        for edge in edges:
            isClosed = edge.isClosed
            p0 = edge.coords[0]
            self.addEndPoint(endPoints, p0, isClosed)
            p1 = edge.coords[-1]
            self.addEndPoint(endPoints, p1, isClosed)

        for eiInfo in endPoints:
            if eiInfo.isClosed and eiInfo.degree != 2:
                self._nonSimpleLocation = eiInfo.coord
                return True

        return False

    def addEndPoint(self, endPoints, coord, isClosed):
        """
         * Add an endpoint to the map, creating an entry for it if none exists
        """
        eiInfo = None
        for ep in endPoints:
            if ep.coord == coord:
                eiInfo = ep
        if eiInfo is None:
            eiInfo = EndpointInfo(coord)
            endPoints.append(eiInfo)
        eiInfo.addEndpoint(isClosed)

    def isSimpleMultiPoint(self, multiPoint):
        raise NotImplementedError()
