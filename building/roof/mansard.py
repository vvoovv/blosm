"""
This file is part of blender-osm (OpenStreetMap importer for Blender).
Copyright (C) 2014-2017 Vladimir Elistratov
prokitektura+support@gmail.com

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from .hipped import RoofHipped
from .flat import RoofFlat
from mathutils import Vector


class RoofMansard(RoofHipped):
    """
    The mansard roof shape is implemented only for a quadrangle building outline.
    For the other building outlines a flat roof is created.
    """
    
    def make(self, bldgMaxHeight, roofMinHeight, bldgMinHeight, osm):
        if self.makeFlat:
            return RoofFlat.make(self, bldgMaxHeight, roofMinHeight, bldgMinHeight, osm)
        else:
            verts = self.verts
            wallIndices = self.wallIndices
            roofIndices = self.roofIndices
            polygon = self.polygon
            _indices = polygon.indices
            
            if not bldgMinHeight is None:
                indexOffset = _indexOffset = len(verts)
                # verts
                verts.extend(Vector((v.x, v.y, roofMinHeight)) for v in polygon.verts)
                # the starting side
                wallIndices.append((_indices[-1], _indices[0], indexOffset, indexOffset + polygon.n - 1))
                wallIndices.extend(
                    (_indices[i-1], _indices[i], indexOffset + i, indexOffset + i - 1) for i in range(1, polygon.n)
                )
                # new values for the polygon indices
                polygon.indices = tuple(_indexOffset + i for i in range(polygon.n))
            
            indexOffset = len(verts)
            self.h /= 2.
            polygon.inset(2., roofIndices, self.h)
            # new values for the polygon indices
            polygon.indices = tuple(indexOffset + i for i in range(polygon.n))
            self.wallIndices = roofIndices
            super().make(bldgMaxHeight, roofMinHeight+self.h, None, osm)
            self.wallIndices = wallIndices
            
            # restore the original indices
            polygon.indices = _indices
        
        return True