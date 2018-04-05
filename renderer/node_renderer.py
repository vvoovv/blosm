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
from . import Renderer, assignTags
from util.osm import parseNumber


class BaseNodeRenderer(Renderer):
    
    def __init__(self, app):
        self.app = app
    
    def renderNode(self, node, osm):
        tags = node.tags
        layer = node.l
        
        coords = node.getData(osm)
        
        # calculate z-coordinate of the object
        z = parseNumber(tags["min_height"], 0.) if "min_height" in tags else 0.
        if self.app.terrain:
            terrainOffset = self.app.terrain.project(coords)
            if terrainOffset is None:
                # the point is outside of the terrain
                return
            z += terrainOffset[2]
        
        obj = self.createBlenderObject(
            self.getName(node),
            (coords[0], coords[1], z),
            layer.getParent(),
            layer.id
        )
        
        if obj:
            # assign OSM tags to the blender object
            assignTags(obj, node.tags)

    @classmethod
    def createBlenderObject(self, name, location, parent, sourceName):
        if sourceName in bpy.data.objects:
            obj = bpy.data.objects.new(name, bpy.data.objects[sourceName].data)
            if location:
                obj.location = location
            bpy.context.scene.objects.link(obj)
            if parent:
                # perform parenting
                obj.parent = parent
            return obj