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

from util import zAxis
from . import Roof


class RoofPyramidal(Roof):
    """
    A Blender object to deal with buildings or building part with a pyramidal roof
    """
    
    defaultHeight = 3.
    
    def make(self, bldgMaxHeight, roofMinHeight, bldgMinHeight, osm):
        polygon = self.polygon
        verts = self.verts
        roofIndices = self.roofIndices
        indices = polygon.indices
        
        if not bldgMinHeight is None:
            indexOffset = len(verts)
            # Create sides for the prism with the height <roofMinHeight - bldgMinHeight>,
            # that is based on the <polygon>
            polygon.sidesPrism(roofMinHeight, self.wallIndices)
        
        # index for the top vertex
        topIndex = len(verts)
        verts.append(
            polygon.center + (bldgMaxHeight - (roofMinHeight if bldgMinHeight is None else bldgMinHeight)) * zAxis
        )
        
        # indices for triangles that form the pyramidal roof
        if bldgMinHeight is None:
            roofIndices.extend(
                (indices[i-1], indices[i], topIndex) for i in range(polygon.n)
            )
        else:
            # the starting triangle
            roofIndices.append((indexOffset + polygon.n - 1, indexOffset, topIndex))
            roofIndices.extend(
                (i - 1, i, topIndex) for i in range(indexOffset + 1, indexOffset + polygon.n)
            )
            
        return True