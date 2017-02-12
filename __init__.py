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
    "version": (2, 2, 0),
    "blender": (2, 7, 8),
    "location": "File > Import > OpenStreetMap (.osm)",
    "description": "Import a file in the OpenStreetMap format (.osm)",
    "warning": "",
    "wiki_url": "https://github.com/vvoovv/blender-osm/wiki/Documentation",
    "tracker_url": "https://github.com/vvoovv/blender-osm/issues",
    "support": "COMMUNITY",
    "category": "Import-Export",
}

import os, sys, math

def _checkPath():
    path = os.path.dirname(__file__)
    if path in sys.path:
        sys.path.remove(path)
    # make <path> the first one to search for a module
    sys.path.insert(0, path)
_checkPath()

import bpy, bmesh

from util.transverse_mercator import TransverseMercator
from util.polygon import Polygon
from renderer import Renderer
from parse import Osm
import app, gui
from defs import Keys

from setup import setup


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
    """Import data: OpenStreetMap, terrain or base overlay for the terrain"""
    bl_idname = "blender_osm.import_data"  # important since its how bpy.ops.blender_osm.import_data is constructed
    bl_label = "Import data of the selected type (OSM, terrain or terrain overlay)"
    bl_description = "Import data of the selected type (OpenStreetMap, terrain or base overlay for the terrain)"
    bl_options = {'REGISTER', 'UNDO'}

    # ImportHelper mixin class uses this
    filename_ext = ".osm"

    filter_glob = bpy.props.StringProperty(
        default="*.osm;*.xml",
        options={"HIDDEN"},
    )

    def execute(self, context):
        a = app.app
        try:
            # path to the directory for assets is given as an argument to <a.init(..)>
            a.init(
                self,
                context,
                os.path.dirname(os.path.realpath(__file__)),
                BlenderOsmPreferences.bl_idname
            )
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {"FINISHED"}
        #Renderer.init(context)
        
        # tangent to check if an angle of the polygon is straight
        Polygon.straightAngleTan = math.tan(math.radians( abs(180.-a.straightAngleThreshold) ))
        
        scene = context.scene
        kwargs = {}
        
        # setting active object if there is no active object
        if context.mode != "OBJECT":
            # if there is no object in the scene, only "OBJECT" mode is provided
            if not scene.objects.active:
                scene.objects.active = context.scene.objects[0]
            bpy.ops.object.mode_set(mode="OBJECT")
        
        osm = Osm(a)
        setup(a, osm)
        
        if "lat" in scene and "lon" in scene and not self.ignoreGeoreferencing:
            kwargs["projection"] = TransverseMercator(lat=scene["lat"], lon=scene["lon"])
        else:
            kwargs["projectionClass"] = TransverseMercator
        
        osm.parse(**kwargs)
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
        
        return {"FINISHED"}


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