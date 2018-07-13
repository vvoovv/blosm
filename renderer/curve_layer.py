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

import os
import bpy
from .layer import MeshLayer
from util.blender import appendObjectsFromFile, createDiffuseMaterial


class CurveLayer(MeshLayer):
    
    # Blender layer index to place a way profile
    profileLayerIndex = 1
    
    # Blender file with way profiles
    assetFile = "way_profiles.blend"
    
    def __init__(self, layerId, app):
        super().__init__(layerId, app)
        self.assetPath = os.path.join(app.assetPath, self.assetFile)

    def getDefaultZ(self, app):
        return app.wayZ

    def getDefaultSwOffset(self, app):
        return app.swWayOffset

    def finalizeBlenderObject(self, obj):
        """
        Slice Blender MESH object, add modifiers
        """
        # set a bevel object for the curve
        curve = obj.data
        # the name of the bevel object
        bevelName = "profile_%s" % self.id
        bevelObj = bpy.data.objects.get(bevelName)
        if not (bevelObj and bevelObj.type == 'CURVE'):
            bevelObj = appendObjectsFromFile(self.assetPath, bevelName)[0]
            if bevelObj:
                # move <obj> to the Blender layer with the index <self.profileLayerIndex>
                bevelObj.layers[self.profileLayerIndex] = True
                bevelObj.layers[0] = False
        if bevelObj and bevelObj.type == 'CURVE':
            curve.bevel_object = bevelObj
        # set a material
        # the material name is simply <id> of the layer
        name = self.id
        material = bpy.data.materials.get(name)
        curve.materials.append(
            material or createDiffuseMaterial(name, self.app.colors.get(name, self.app.defaultColor))
        )
        
        if self.modifiers:
            self.addShrinkwrapModifier(obj, self.app.terrain.terrain, self.swOffset)