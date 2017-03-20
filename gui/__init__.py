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

import bpy
import webbrowser
from mathutils import Vector
from app import app
from defs import Keys
from util.transverse_mercator import TransverseMercator


class OperatorSelectExtent(bpy.types.Operator):
    bl_idname = "blender_osm.select_extent"
    bl_label = "select"
    bl_description = "Select extent for your area of interest on a geographical map"
    bl_options = {'INTERNAL'}
    
    url = "http://prokitektura.com/blender-osm/extent/"
    
    def invoke(self, context, event):
        bv = bpy.app.version
        av = app.version
        webbrowser.open_new_tab(
            "%s?blender_version=%s.%s&addon=blender-osm&addon_version=%s.%s.%s" %
            (self.url, bv[0], bv[1], av[0], av[1], av[2])
        )
        return {'FINISHED'}


class OperatorPasteExtent(bpy.types.Operator):
    bl_idname = "blender_osm.paste_extent"
    bl_label = "paste"
    bl_description = "Paste extent (chosen on the geographical map) for your area of interest from the clipboard"
    bl_options = {'INTERNAL', 'UNDO'}
    
    def invoke(self, context, event):
        addon = context.scene.blender_osm
        coords = context.window_manager.clipboard
        
        if not coords:
            self.report({'ERROR'}, "Nothing to paste!")
            return {'FINISHED'}
        try:
            # parse the string from the clipboard to get coordinates of the extent
            coords = tuple( map(lambda s: float(s), coords[(coords.find('=')+1):].split(',')) )
            if len(coords) != 4:
                raise ValueError
        except ValueError:
            self.report({'ERROR'}, "Invalid string to paste!")
            return {'FINISHED'}
        
        addon.minLon = coords[0]
        addon.minLat = coords[1]
        addon.maxLon = coords[2]
        addon.maxLat = coords[3]
        return {'FINISHED'}
    

class OperatorExtentFromActive(bpy.types.Operator):
    bl_idname = "blender_osm.extent_from_active"
    bl_label = "from active"
    bl_description = "Use extent from the active Blender object"
    bl_options = {'INTERNAL', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        scene = context.scene
        return context.object and context.object.type == "MESH" and "lat" in scene and "lon" in scene
    
    def invoke(self, context, event):
        scene = context.scene
        addon = scene.blender_osm
        obj = context.object
        projection = TransverseMercator(lat=scene["lat"], lon=scene["lon"])
        # transform <obj.bound_box> to the world system of coordinates
        bound_box = tuple(obj.matrix_world*Vector(v) for v in obj.bound_box)
        bbox = []
        for i in (0,1):
            for f in (min, max):
                bbox.append(
                    f( bound_box, key=lambda v: v[i] )[i]
                )
        bbox = (
            projection.toGeographic(bbox[0], bbox[2]),
            projection.toGeographic(bbox[1], bbox[3])
        )
        addon.minLat = bbox[0][0]
        addon.maxLat = bbox[1][0]
        addon.minLon = bbox[0][1]
        addon.maxLon = bbox[1][1]
        return {'FINISHED'}


class PanelExtent(bpy.types.Panel):
    bl_label = "blender-osm"
    bl_space_type = "VIEW_3D"
    bl_region_type = "TOOLS"
    bl_context = "objectmode"
    bl_category = "osm"

    def draw(self, context):
        layout = self.layout
        addon = context.scene.blender_osm
        
        if addon.osmSource == "server" or addon.dataType != "osm":
            box = layout.box()
            row = box.row()
            row.alignment = "CENTER"
            row.label("Extent:")
            row = box.row(align=True)
            row.operator("blender_osm.select_extent")
            row.operator("blender_osm.paste_extent")
            row.operator("blender_osm.extent_from_active")
            
            split = box.split(percentage=0.25)
            split.label()
            split.split(percentage=0.67).prop(addon, "maxLat")
            row = box.row()
            row.prop(addon, "minLon")
            row.prop(addon, "maxLon")
            split = box.split(percentage=0.25)
            split.label()
            split.split(percentage=0.67).prop(addon, "minLat")
        
        box = layout.box()
        row = box.row(align=True)
        row.prop(addon, "dataType", text="")
        row.operator("blender_osm.import_data", text="import")


class PanelSettings(bpy.types.Panel):
    bl_label = "Settings"
    bl_space_type = "VIEW_3D"
    bl_region_type = "TOOLS"
    bl_context = "objectmode"
    bl_category = "osm"
    
    def draw(self, context):
        addon = context.scene.blender_osm
        
        dataType = addon.dataType
        if dataType == "osm":
            self.drawOsm(context)
        elif dataType == "terrain":
            self.drawTerrain(context)
        elif dataType == "overlay":
            self.drawOverlay(context)
    
    def drawOsm(self, context):
        layout = self.layout
        addon = context.scene.blender_osm
        
        box = layout.box()
        box.prop(addon, "osmSource", text="Import from")
        if addon.osmSource == "file":
            box.prop(addon, "osmFilepath", text="File")
        
        layout.box().prop_search(addon, "terrainObject", context.scene, "objects")
            
        if app.has(Keys.mode3d):
            layout.prop(addon, "mode", expand=True)
        
        box = layout.box()
        box.prop(addon, "buildings")
        box.prop(addon, "water")
        box.prop(addon, "forests")
        box.prop(addon, "vegetation")
        box.prop(addon, "highways")
        box.prop(addon, "railways")
        
        box = layout.box()
        split = box.split(percentage=0.67)
        split.label("Default roof shape:")
        split.prop(addon, "defaultRoofShape", text="")
        box.prop(addon, "levelHeight")
        box.prop(addon, "defaultNumLevels")
        box.prop(addon, "straightAngleThreshold")
        
        box = layout.box()
        box.prop(addon, "singleObject")
        box.prop(addon, "layered")
        
        layout.box().prop(addon, "ignoreGeoreferencing")
        
        if addon.terrainObject and addon.terrainObject in context.scene.objects:
            box = layout.box()
            box.prop(addon, "sliceFlatLayers")
            if addon.sliceFlatLayers:
                box.prop(addon, "sliceSize")
    
    def drawTerrain(self, context):
        self.layout.prop(context.scene.blender_osm, "ignoreGeoreferencing")
    
    def drawOverlay(self, context):
        self.layout.label("overlay is under development")


class BlenderOsmProperties(bpy.types.PropertyGroup):
    
    terrainObject = bpy.props.StringProperty(
        name = "Terrain",
        description = "Blender object for the terrain"
    )
    
    osmSource = bpy.props.EnumProperty(
        name = "Import OpenStreetMap from",
        items = (
            ("server", "server", "remote server"),
            ("file", "file", "file on the local disk")
        ),
        description = "From where to import OpenStreetMap data: remote server or a file on the local disk",
        default = "server"
    )
    
    osmFilepath = bpy.props.StringProperty(
        name = "OpenStreetMap file",
        subtype = 'FILE_PATH',
        description = "Path to an OpenStreetMap file for import"
    )
    
    dataType = bpy.props.EnumProperty(
        name = "Data",
        items = (
            ("osm", "OpenStreetMap", "OpenStreetMap"),
            ("terrain", "terrain", "Terrain")
            #("overlay", "base overlay", "Base overlay for the terrain, e.g. satellite imagery or maps")
        ),
        description = "Data type for import",
        default = "osm"
    )
    
    mode = bpy.props.EnumProperty(
        name = "Mode: 3D or 2D",
        items = (("3D","3D","3D"), ("2D","2D","2D")),
        description = "Import data in 3D or 2D mode",
        default = "3D"
    )
    
    # extent bounds: minLat, maxLat, minLon, maxLon
    
    minLat = bpy.props.FloatProperty(
        name="min lat",
        description="Minimum latitude of the imported extent",
        precision = 4,
        min = -89.,
        max = 89.,
        default=55.748
    )

    maxLat = bpy.props.FloatProperty(
        name="max lat",
        description="Maximum latitude of the imported extent",
        precision = 4,
        min = -89.,
        max = 89.,
        default=55.756
    )

    minLon = bpy.props.FloatProperty(
        name="min lon",
        description="Minimum longitude of the imported extent",
        precision = 4,
        min = -180.,
        max = 180.,
        default=37.6117
    )

    maxLon = bpy.props.FloatProperty(
        name="max lon",
        description="Maximum longitude of the imported extent",
        precision = 4,
        min = -180.,
        max = 180.,
        default=37.624
    )
    
    buildings = bpy.props.BoolProperty(
        name = "Import buildings",
        description = "Import building outlines",
        default = True
    )
    
    water = bpy.props.BoolProperty(
        name = "Import water objects",
        description = "Import water objects (rivers and lakes)",
        default = True
    )
    
    forests = bpy.props.BoolProperty(
        name = "Import forests",
        description = "Import forests and woods",
        default = True
    )
    
    vegetation = bpy.props.BoolProperty(
        name = "Import other vegetation",
        description = "Import other vegetation (grass, meadow, scrub)",
        default = True
    )
    
    highways = bpy.props.BoolProperty(
        name = "Import roads and paths",
        description = "Import roads and paths",
        default = False
    )
    
    railways = bpy.props.BoolProperty(
        name = "Import railways",
        description = "Import railways",
        default = False
    )
    
    defaultRoofShape = bpy.props.EnumProperty(
        items = (("flat", "flat", "flat shape"), ("gabled", "gabled", "gabled shape")),
        description = "Roof shape for a building if the roof shape is not set in OpenStreetMap",
        default = "flat"
    )
    
    singleObject = bpy.props.BoolProperty(
        name = "Import as a single object",
        description = "Import OSM objects as a single Blender mesh objects instead of separate ones",
        default = True
    )
    
    layered = bpy.props.BoolProperty(
        name = "Arrange into layers",
        description = "Arrange imported OSM objects into layers (buildings, highways, etc)",
        default = True
    )

    ignoreGeoreferencing = bpy.props.BoolProperty(
        name = "Ignore existing georeferencing",
        description = "Ignore existing georeferencing and make a new one",
        default = False
    )
    
    levelHeight = bpy.props.FloatProperty(
        name = "Level height",
        description = "Height of a level in meters to use for OSM tags building:levels and building:min_level",
        default = 3.
    )
    
    defaultNumLevels = bpy.props.IntProperty(
        name = "Default number of levels",
        description = "Default number of levels for a building if the number of levels or " +
            "the building height aren't set in OSM tags",
        min = 1,
        subtype = 'UNSIGNED',
        default = 2
    )
    
    straightAngleThreshold = bpy.props.FloatProperty(
        name = "Straight angle threshold",
        description = "Threshold for an angle of the building outline: when consider it as straight one. "+
            "It may be important for calculation of the longest side of the building outline for a gabled roof.",
        default = 179.8,
        min = 170.,
        max = 179.95,
        step = 10 # i.e. step/100 == 0.1
    )
    
    loadMissingMembers = bpy.props.BoolProperty(
        name = "Load missing members of relations",
        description = "Relation members aren't contained in the OSM file " +
            "if they are located outside of the OSM file extent. " +
            "Enable this option to load the missiong members of the relations " +
            "either from a local file (if available) or from the server.",
        default = True
    )
    
    sliceFlatLayers = bpy.props.BoolProperty(
        name = "Slice flat layers",
        description = "Slice mesh of flat layers (water, forest, vegetaion) " +
        "to project them on the terrain correctly",
        default = True
    )
    
    sliceSize = bpy.props.FloatProperty(
        name = "Slice size",
        description = "Slice size in meters",
        default = 30.,
        min = 10.,
        step = 100 # i.e. step/100 == 1.
    )
    
    # Terrain settings
    # SRTM3 data are sampled at either 3 arc-second and contain 1201 lines and 1201 samples
    # or 1 arc-second and contain 3601 lines and 3601 samples
    terrainResolution = bpy.props.EnumProperty(
        name="Resolution",
        items=(("1", "1 arc-second", "1 arc-second"), ("3", "3 arc-second", "3 arc-second")),
        description="Spation resolution",
        default="1"
    )
    
    terrainPrimitiveType = bpy.props.EnumProperty(
        name="Mesh primitive type: quad or triangle",
        items=(("quad","quad","quad"),("triangle","triangle","triangle")),
        description="Primitive type used for the terrain mesh: quad or triangle",
        default="quad"
    )


def register():
    bpy.utils.register_module(__name__)
    # a group for all GUI attributes related to blender-osm
    bpy.types.Scene.blender_osm = bpy.props.PointerProperty(type=BlenderOsmProperties)

def unregister():
    bpy.utils.unregister_module(__name__)
    del bpy.types.Scene.blender_osm