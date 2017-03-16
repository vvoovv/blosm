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

from mathutils import Vector
from renderer import Renderer
from util.blender import createEmptyObject


class Layer:
    
    def __init__(self, layerId, app):
        self.app = app
        self.id = layerId
        terrain = app.terrain
        hasTerrain = bool(terrain)
        # The following two lines of the code mean that
        # if a terrain is set, then we force:
        # <self.singleObject = True> and <self.layered = True>
        self.singleObject = app.singleObject or hasTerrain
        self.layered = app.layered or hasTerrain
        # instance of BMesh
        self.bm = None
        # Blender object
        self.obj = None
        self.materialIndices = []
        # Blender parent object
        self.parent = None
        # apply SHRINKWRAP modifier if a terrain is set
        self.swModifier = hasTerrain
        # slice flat mesh to project it on the terrain correctly
        self.sliceMesh = hasTerrain and app.sliceFlatLayers
        # set layer offsets <self.location>, <self.meshZ> and <self.parentLocation>
        # <self.location> is used for a Blender object
        # <self.meshZ> is used for vertices of a BMesh
        # <self.parentLocation> is used for an EMPTY Blender object serving
        # as a parent for Blender objects of the layer
        self.parentLocation = None
        meshZ = 0.
        _z = app.layerOffsets[layerId]
        if hasTerrain:
            # here we have <self.singleObject is True> and <self.layered is True>
            location = Vector((0., 0., terrain.maxZ + terrain.layerOffset))
            self.swOffset = _z if _z else app.swOffset
        elif self.singleObject and self.layered:
            location = Vector((0., 0., _z))
        elif not self.singleObject and self.layered:
            location = None
            # it's the only case when <self.parentLocation> is needed
            self.parentLocation = Vector((0., 0., app.layerOffsets[layerId]))
        elif self.singleObject and not self.layered:
            location = None
            meshZ = _z
        elif not self.singleObject and not self.layered:
            location = Vector((0., 0., _z))
        self.location = location
        self.meshZ = meshZ
        
    def getParent(self):
        # The method is called currently in the single place of the code:
        # in <Renderer.prerender(..)> if (not layer.singleObject and app.layered)
        parent = self.parent
        if not self.parent:
            parent = createEmptyObject(
                self.name,
                self.parentLocation.copy(),
                empty_draw_size=0.01
            )
            parent.parent = Renderer.parent
            self.parent = parent
        return parent
    
    @property
    def name(self):
        return "%s_%s" % (Renderer.name, self.id)