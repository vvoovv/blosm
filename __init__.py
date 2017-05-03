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

bl_info = {
    "name": "Import OpenStreetMap (.osm)",
    "author": "Vladimir Elistratov <prokitektura+support@gmail.com>",
    "version": (2, 2, 1),
    "blender": (2, 7, 8),
    "location": "File > Import > OpenStreetMap (.osm)",
    "description": "Import a file in the OpenStreetMap format (.osm)",
    "warning": "",
    "wiki_url": "https://github.com/vvoovv/blender-osm/wiki/Documentation",
    "tracker_url": "https://github.com/vvoovv/blender-osm/issues",
    "support": "COMMUNITY",
    "category": "Import-Export",
}

import os, sys

# force cleanup of sys.modules to avoid conflicts with the other addons for Blender
for m in [
        "app", "building", "gui", "manager", "material", "parse",
        "renderer", "terrain", "util", "defs", "setup"
    ]:
    sys.modules.pop(m, 0)

def _checkPath():
    path = os.path.dirname(__file__)
    if path in sys.path:
        sys.path.remove(path)
    # make <path> the first one to search for a module
    sys.path.insert(0, path)
_checkPath()

import bpy, bmesh

from util.transverse_mercator import TransverseMercator
from renderer import Renderer
from parse import Osm
import app, gui
from defs import Keys

from setup import setup

# set addon version
app.app.version = bl_info["version"]


class BlenderOsmPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__
    
    dataDir= bpy.props.StringProperty(
        name = "",
        subtype = 'DIR_PATH',
        description = "Directory to store downloaded OpenStreetMap and terrain files"
    )
    
    def draw(self, context):
        layout = self.layout
        layout.label("Directory to store downloaded OpenStreetMap and terrain files:")
        layout.prop(self, "dataDir")


class ImportData(bpy.types.Operator):
    """Import data: OpenStreetMap or terrain"""
    bl_idname = "blender_osm.import_data"  # important since its how bpy.ops.blender_osm.import_data is constructed
    bl_label = "blender-osm"
    bl_description = "Import data of the selected type (OpenStreetMap or terrain)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        # path to the directory for assets
        basePath = os.path.dirname(os.path.realpath(__file__))
        dataType = context.scene.blender_osm.dataType
        
        if dataType == "osm":
            return self.importOsm(context, basePath)
        elif dataType == "terrain":
            return self.importTerrain(context, basePath)
        
        return {'FINISHED'}
    
    def importOsm(self, context, basePath):
        a = app.app
        try:
            a.initOsm(self, context, basePath, BlenderOsmPreferences.bl_idname)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'FINISHED'}
        
        scene = context.scene
        kwargs = {}
        
        self.setObjectMode(context)
        
        osm = Osm(a)
        setup(a, osm)
        
        if "lat" in scene and "lon" in scene and not a.ignoreGeoreferencing:
            kwargs["projection"] = TransverseMercator(lat=scene["lat"], lon=scene["lon"])
        else:
            kwargs["projectionClass"] = TransverseMercator
        
        osm.parse(a.osmFilepath, **kwargs)
        if a.loadMissingMembers and a.incompleteRelations:
            try:
                a.loadMissingWays(osm)
            except Exception as e:
                self.report({'ERROR'}, str(e))
                a.loadMissingMembers = False
            a.processIncompleteRelations(osm)
        a.process()
        a.render()
        
        # setting 'lon' and 'lat' attributes for <scene> if necessary
        if not "projection" in kwargs:
            # <kwargs["lat"]> and <kwargs["lon"]> have been set in osm.parse(..)
            scene["lat"] = osm.lat
            scene["lon"] = osm.lon
        
        if not a.has(Keys.mode3d):
            a.show()
        
        a.clean()
        
        return {'FINISHED'}
    
    def importTerrain(self, context, basePath):
        a = app.app
        try:
            a.initTerrain(self, context, basePath, BlenderOsmPreferences.bl_idname)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'FINISHED'}
        
        scene = context.scene
        if "lat" in scene and "lon" in scene and not a.ignoreGeoreferencing:
            lat = scene["lat"]
            lon = scene["lon"]
            setLatLon = False
        else:
            lat = (a.minLat+a.maxLat)/2.
            lon = (a.minLon+a.maxLon)/2.
            setLatLon = True
        
        a.projection = TransverseMercator(lat=lat, lon=lon)
        a.importTerrain(context)
        
        # set custom parameter "lat" and "lon" to the active scene
        if setLatLon:
            scene["lat"] = lat
            scene["lon"] = lon
        return {'FINISHED'}
    
    def setObjectMode(self, context):
        # setting active object if there is no active object
        if context.mode != "OBJECT":
            scene = context.scene
            # if there is no object in the scene, only "OBJECT" mode is available
            if not scene.objects.active:
                scene.objects.active = scene.objects[0]
            bpy.ops.object.mode_set(mode="OBJECT")


def register():
    bpy.utils.register_module(__name__)
    app.register()
    gui.register()

def unregister():
    bpy.utils.unregister_module(__name__)
    app.unregister()
    gui.unregister()

# This allows you to run the script directly from blenders text editor
# to test the addon without having to install it.
if __name__ == "__main__":
    register()