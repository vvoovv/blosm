"""
This file is part of blender-osm (OpenStreetMap importer for Blender).
Copyright (C) 2014-2018 Vladimir Elistratov
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

import bpy
from building.layer import BuildingLayer

_isBlender280 = bpy.app.version[1] >= 80


class RealisticBuildingLayer(BuildingLayer):
    
    # the name for the base UV map
    uvName = "UVMap"
    
    # the name for the auxiliary UV map used to keep the size of a BMFace
    uvNameSize = "size"
    
    def prepare(self, instance):
        uv_layers = instance.obj.data.uv_layers if _isBlender280 else instance.obj.data.uv_textures
        uv_layers.new(name=self.uvName)
        uv_layers.new(name=self.uvNameSize)
        super().prepare(instance)