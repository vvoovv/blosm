# -*- coding:utf-8 -*-

# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

# <pep8 compliant>

# ----------------------------------------------------------
# Author: Stephen Leger (s-leger)
#
# ----------------------------------------------------------


from .op_polygonize import PolygonizeOp
from .op_linemerge import LineMerger
from .shared import (
    logger,
    quicksort,
    CoordinateSequence
    )


class PolygonsUnionOp():
    """
     * Fast but dumb union of a set of adjascent polygons
     * Assume polygons allready contains proper intersections as vertices
     * and are valids
    """
    def __init__(self, geoms):
        """
         * @param geoms: collection of polygons
        """
        self._factory = geoms[0]._factory
        self.segmap = {}
        self.geoms = geoms

    def addPolygonRing(self, geom) -> None:

        # skip empty component
        if geom.is_empty:
            return

        coords = CoordinateSequence.removeRepeatedPoints(geom.coords)

        if len(coords) < 4:
            return

        prec = coords.pop(0)
        x0, y0 = prec.x, prec.y
        for next in coords:

            x1, y1 = next.x, next.y

            if x0 < x1:
                key = (x0, y0, x1, y1)
            elif x0 > x1:
                key = (x1, y1, x0, y0)
            elif y0 < y1:
                key = (x0, y0, x1, y1)
            else:
                key = (x1, y1, x0, y0)

            if self.segmap.get(key) is None:
                if prec != next:
                    self.segmap[key] = [prec, next]
            else:
                del self.segmap[key]

            prec = next
            x0, y0 = x1, y1

    def addPolygon(self, geom) -> None:
        # LinearRing
        self.addPolygonRing(geom.exterior)
        if geom.interiors is not None:
            for hole in geom.interiors:
                self.addPolygonRing(hole)

    @staticmethod
    def union(geoms):
        """
         * @param geoms: collection of polygons
        """
        if len(geoms) == 0:
            return None

        op = PolygonsUnionOp(geoms)
        return op._union()
    
    @staticmethod
    def poly_area_gt(a, b):
        return a.exterior_area > b.exterior_area
        
    @staticmethod
    def filter_nested(polys):
        """
          Filter out nested touching holes
        """
        quicksort(polys, PolygonsUnionOp.poly_area_gt)
        to_remove = []
        n_polys = len(polys)
        
        for i, poly in enumerate(polys):
            if i in to_remove:
                continue
            for hole in poly.interiors:
                for j in range(i + 1, n_polys):
                    other = polys[j]
                    if (hole.envelope.equals(other.envelope) and
                            CoordinateSequence.equals_unoriented(hole.coords, other.exterior.coords)):
                        to_remove.append(j)
        to_remove.sort()
        for i in reversed(to_remove):
            polys.pop(i)
        
    def _union(self):
        # copy points from input Geometries.
        # This ensures that any Point geometries
        # in the input are considered for inclusion in the result set
        logger.debug("PolygonsUnionOp._union() segments map")
        
        for geom in self.geoms:
            self.addPolygon(geom)

        logger.debug("PolygonsUnionOp._union() build lines(%s)", len(self.segmap))

        segs = self.segmap.values()

        lines = []
        for seg in segs:
            lines.append(self._factory.createLineString(seg))

        logger.debug("PolygonsUnionOp._union() LineMerger.merge(%s)", len(lines))

        lines = LineMerger.merge(lines)

        logger.debug("PolygonsUnionOp._union() PolygonizeOp.polygonize(%s)", len(lines))
        polys = PolygonizeOp.polygonize(lines)

        # filter out nested touching holes
        # Sort polygons by area before this check
        # to ensure right check order
        PolygonsUnionOp.filter_nested(polys)
            
        logger.debug("PolygonsUnionOp._union() done %s", len(polys))
        return polys
