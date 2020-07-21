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

bl_info = {
    "name": "blender-osm",
    "author": "Vladimir Elistratov <prokitektura+support@gmail.com>",
    "version": (2, 4, 21),
    "blender": (2, 80, 0),
    "location": "Right side panel for Blender 2.8x (left side panel for Blender 2.79))> \"osm\" tab",
    "description": "One click download and import of OpenStreetMap, terrain, satellite imagery, web maps",
    "warning": "",
    "wiki_url": "https://github.com/vvoovv/blender-osm/wiki/Premium-Version",
    "tracker_url": "https://github.com/vvoovv/blender-osm/issues",
    "support": "COMMUNITY",
    "category": "Import-Export"
}

import os, sys, textwrap

# force cleanup of sys.modules to avoid conflicts with the other addons for Blender
for m in [
        "app", "building", "gui", "manager", "material", "parse", "realistic", "overlay",
        "renderer", "terrain", "util", "defs", "setup"
    ]:
    sys.modules.pop(m, 0)

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
    sys.path.insert(1, os.path.join(path, "pml"))
_checkPath()


import bpy, bmesh, bgl, blf

from util.transverse_mercator import TransverseMercator
from renderer import Renderer
from parse.osm import Osm
import app, gui
from defs import Keys

# set addon version
app.app.version = bl_info["version"]
app.app.isPremium = os.path.isdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "realistic"))
_isBlender280 = bpy.app.version[1] >= 80


class BlenderOsmPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__
    
    dataDir = bpy.props.StringProperty(
        name = '',
        subtype = 'DIR_PATH',
        description = "Directory to store downloaded OpenStreetMap and terrain files"
    )
    
    assetsDir = bpy.props.StringProperty(
        name = '',
        subtype = 'DIR_PATH',
        description = "Directory with assets (building_materials.blend, vegetation.blend). "+
            "It can be also set in the addon GUI"
    )
    
    mapboxAccessToken = bpy.props.StringProperty(
        name = "Mapbox access token",
        description = "A string token to access overlays from Mapbox company"
    )
    
    osmServer = bpy.props.EnumProperty(
        name = "OSM data server",
        items = (
            ("overpass-api.de", "overpass-api.de: 8 cores, 128 GB RAM", "overpass-api.de: 8 cores, 96 GB RAM"),
            ("openstreetmap.fr", "openstreetmap.fr: 8 cores, 16 GB RAM", "openstreetmap.fr: 8 cores, 16 GB RAM"),
            ("kumi.systems", "kumi.systems: 20 cores, 256GB RAM", "kumi.systems: 20 cores, 256GB RAM")
        ),
        description = "OSM data server if the default one is inaccessible",
        default = "overpass-api.de"
    )
    
    enableExperimentalFeatures = bpy.props.BoolProperty(
        name = "Enable export (experimental)",
        description = "Enable export to the popular 3D formats. Experimental feature! Use it with caution!",
        default = False
    )
    
    def draw(self, context):
        layout = self.layout
        
        if app.app.isPremium:
            box = layout.box()
            box.label(text="Thank you for purchasing the premium version!")
        
        layout.label(text="Directory to store downloaded OpenStreetMap and terrain files:")
        layout.prop(self, "dataDir")
        
        if app.app.isPremium:
            layout.label(text="Directory with assets (building_materials.blend, vegetation.blend):")
            layout.prop(self, "assetsDir")
        
        layout.separator()
        split = layout.split(factor=0.9) if _isBlender280 else layout.split(percentage=0.9)
        split.prop(self, "mapboxAccessToken")
        split.operator("blosm.get_mapbox_token", text="Get it!")
        
        layout.separator()
        layout.box().label(text="Advanced settings:")
        # Extensions might come later
        #layout.operator("blosm.load_extensions", text="Load extensions")
        layout.prop(self, "osmServer")
        
        layout.box().prop(self, "enableExperimentalFeatures", text="Enable experimental features")

app.app.addonName = BlenderOsmPreferences.bl_idname


class OperatorGetMapboxToken(bpy.types.Operator):
    bl_idname = "blosm.get_mapbox_token"
    bl_label = ""
    bl_description = "Get Mapbox access token"
    bl_options = {'INTERNAL'}
    
    url = "https://www.mapbox.com/account/access-tokens"
    
    def execute(self, context):
        import webbrowser
        webbrowser.open_new_tab(self.url)
        return {'FINISHED'}


class OperatorLoadExtensions(bpy.types.Operator):
    bl_idname = "blosm.load_extensions"
    bl_label = ""
    bl_description = "Scan Blender addons, find extensions for blender-osm and load them"
    bl_options = {'INTERNAL'}
    
    def execute(self, context):
        numExtensions = app.app.loadExtensions(context)
        self.report({'INFO'},
            "No extension found" if not numExtensions else\
            ("Loaded 1 extension" if numExtensions==1 else "Loaded %s extensions" % numExtensions)
        )
        return {'FINISHED'}


class OperatorImportData(bpy.types.Operator):
    """Import data: OpenStreetMap or terrain"""
    bl_idname = "blender_osm.import_data"  # important since its how bpy.ops.blender_osm.import_data is constructed
    bl_label = "blender-osm"
    bl_description = "Import data of the selected type (OpenStreetMap or terrain)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        a = app.app
        dataType = context.scene.blender_osm.dataType
        
        a.projection = None
        a.setAttributes(context)
        
        if dataType == "osm":
            return self.importOsm(context)
        elif dataType == "terrain":
            return self.importTerrain(context)
        elif dataType == "overlay":
            return self.importOverlay(context)
        elif dataType == "gpx":
            return self.importGpx(context)
        elif dataType == "geojson":
            return self.importGeoJson(context)
        
        return {'FINISHED'}
    
    def importOsm(self, context):
        a = app.app
        addon = context.scene.blender_osm
        
        try:
            a.initOsm(self, context)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        
        setupScript = addon.setupScript
        if setupScript:
            setup_function = self.loadSetupScript(setupScript)
            if not setup_function:
                return {'CANCELLED'}
        else:
            if a.mode is a.realistic:
                if a.enableExperimentalFeatures:
                    from setup.realistic_dev import setup as setup_function
                else:
                    from setup.premium_default import setup as setup_function
            else:
                from setup.base import setup as setup_function
        
        scene = context.scene
        
        self.setObjectMode(context)
        bpy.ops.object.select_all(action='DESELECT')
        
        osm = Osm(a)
        setup_function(a, osm)
        a.createLayers(osm)
        
        setLatLon = False
        if "lat" in scene and "lon" in scene and not a.ignoreGeoreferencing:
            osm.setProjection(scene["lat"], scene["lon"])
        elif a.osmSource == "server":
            osm.setProjection( (a.minLat+a.maxLat)/2., (a.minLon+a.maxLon)/2. )
            setLatLon = True
        else:
            setLatLon = True
        
        createFlatTerrain = a.mode is a.realistic and a.forests
        forceExtentCalculation = createFlatTerrain and a.osmSource == "file"
        
        osm.parse(a.osmFilepath, forceExtentCalculation=forceExtentCalculation)
        if a.loadMissingMembers and a.incompleteRelations:
            try:
                a.loadMissingWays(osm)
            except Exception as e:
                self.report({'ERROR'}, str(e))
                a.loadMissingMembers = False
            a.processIncompleteRelations(osm)
        
        if forceExtentCalculation:
            a.minLat = osm.minLat
            a.maxLat = osm.maxLat
            a.minLon = osm.minLon
            a.maxLon = osm.maxLon

        # check if have a terrain Blender object set
        a.setTerrain(
            context,
            createFlatTerrain = createFlatTerrain,
            createBvhTree = True
        )
        
        a.initLayers()
        
        a.process()
        a.render()
        
        # setting <lon> and <lat> attributes for <scene> if necessary
        if setLatLon:
            # <osm.lat> and <osm.lon> have been set in osm.parse(..)
            self.setCenterLatLon(context, osm.lat, osm.lon)
        
        a.clean()
        
        return {'FINISHED'}
    
    def getCenterLatLon(self, context):
        a = app.app
        scene = context.scene
        if "lat" in scene and "lon" in scene and not a.ignoreGeoreferencing:
            lat = scene["lat"]
            lon = scene["lon"]
            setLatLon = False
        else:
            lat = (a.minLat+a.maxLat)/2.
            lon = (a.minLon+a.maxLon)/2.
            setLatLon = True
        return lat, lon, setLatLon
    
    def setCenterLatLon(self, context, lat, lon):
        context.scene["lat"] = lat
        context.scene["lon"] = lon
    
    def loadSetupScript(self, setupScript):
        setupScript = os.path.realpath(bpy.path.abspath(setupScript))
        if not os.path.isfile(setupScript):
            self.report({'ERROR'},
                "The script file doesn't exist"
            )
            return None
        import imp
        # remove extension from the path
        setupScript = os.path.splitext(setupScript)[0]
        moduleName = os.path.basename(setupScript)
        try:
            _file, _pathname, _description = imp.find_module(moduleName, [os.path.dirname(setupScript)])
            module = imp.load_module(moduleName, _file, _pathname, _description)
            _file.close()
            return module.setup
        except Exception as e:
            print("File \"%s\", line %s" % (e.filename, e.lineno))
            print(e.text)
            print("Error: %s", e.msg)
            self.report({'ERROR'},
                "Unable to execute the setup script! See the error message in the Blender console!"
            )
            return None
    
    def importTerrain(self, context):
        a = app.app
        try:
            a.initTerrain(context)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'FINISHED'}
        
        lat, lon, setLatLon = self.getCenterLatLon(context)
        a.setProjection(lat, lon)
        
        a.importTerrain(context)
        
        # set the custom parameters <lat> and <lon> to the active scene
        if setLatLon:
            self.setCenterLatLon(context, lat, lon)
        return {'FINISHED'}
    
    def importOverlay(self, context):
        a = app.app
        
        # find the Blender area holding 3D View
        for area in bpy.context.screen.areas:
            if area.type == "VIEW_3D":
                a.area = area
                break
        else:
            a.area = None
            
        lat, lon, setLatLon = self.getCenterLatLon(context)
        a.setProjection(lat, lon)
        
        try:
            a.initOverlay(context)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        
        terrainObject = context.scene.objects.get(context.scene.blender_osm.terrainObject)
        
        minLon, minLat, maxLon, maxLat = a.getExtentFromObject(terrainObject, context)\
            if terrainObject else\
            (a.minLon, a.minLat, a.maxLon, a.maxLat)
        
        a.overlay.prepareImport(minLon, minLat, maxLon, maxLat)
        
        bpy.ops.blender_osm.control_overlay()
        
        # set the custom parameters <lat> and <lon> to the active scene
        if setLatLon:
            self.setCenterLatLon(context, lat, lon)
        
        return {'FINISHED'}
    
    def importGpx(self, context):
        from parse.gpx import Gpx
        from gpx import GpxRenderer
        
        a = app.app
        try:
            a.initGpx(context, BlenderOsmPreferences.bl_idname)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        
        scene = context.scene
        
        self.setObjectMode(context)
        bpy.ops.object.select_all(action='DESELECT')
        
        gpx = Gpx(a)
        
        if "lat" in scene and "lon" in scene and not a.ignoreGeoreferencing:
            a.setProjection(scene["lat"], scene["lon"])
            setLatLon = False
        else:
            setLatLon = True
        
        gpx.parse(a.gpxFilepath)
        
        # check if have a terrain Blender object set
        a.setTerrain(
            context,
            createFlatTerrain = False,
            createBvhTree = False
        )
        
        GpxRenderer(a).render(gpx)
        
        # setting <lon> and <lat> attributes for <scene> if necessary
        if setLatLon:
            # <gpx.lat> and <gpx.lon> have been set in gpx.parse(..)
            self.setCenterLatLon(context, gpx.lat, gpx.lon)
        return {'FINISHED'}

    def importGeoJson(self, context):
        from parse.geojson import GeoJson
        
        a = app.app
        addon = context.scene.blender_osm
        
        try:
            a.initGeoJson(self, context)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        
        setupScript = addon.setupScript
        if setupScript:
            setup_function = self.loadSetupScript(setupScript)
            if not setup_function:
                return {'CANCELLED'}
        else:
            if a.mode is a.realistic:
                from setup.premium_default import setup as setup_function
            else:
                from setup.geojson_base import setup as setup_function
        
        scene = context.scene
        
        self.setObjectMode(context)
        bpy.ops.object.select_all(action='DESELECT')
        
        data = GeoJson(a)
        setup_function(a, data)
        a.createLayers(data)
        
        setLatLon = False
        if "lat" in scene and "lon" in scene and not a.ignoreGeoreferencing:
            a.setProjection(scene["lat"], scene["lon"])
        elif a.coordinatesAsFilter:
            a.setProjection( (a.minLat+a.maxLat)/2., (a.minLon+a.maxLon)/2. )
        else:
            setLatLon = True
        
        createFlatTerrain = a.mode is a.realistic and a.forests
        forceExtentCalculation = createFlatTerrain
        
        data.parse(a.osmFilepath, forceExtentCalculation=forceExtentCalculation)
        
        if forceExtentCalculation:
            a.minLat = data.minLat
            a.maxLat = data.maxLat
            a.minLon = data.minLon
            a.maxLon = data.maxLon

        # check if have a terrain Blender object set
        a.setTerrain(
            context,
            createFlatTerrain = createFlatTerrain,
            createBvhTree = True
        )
        
        a.initLayers()
        
        a.process()
        a.render()
        
        # setting <lon> and <lat> attributes for <scene> if necessary
        if setLatLon:
            # <osm.lat> and <osm.lon> have been set in osm.parse(..)
            self.setCenterLatLon(context, data.lat, data.lon)
        
        a.clean()
        
        return {'FINISHED'}
    
    def setObjectMode(self, context):
        scene = context.scene
        # setting active object if there is no active object
        if context.mode != "OBJECT":
            # if there is no object in the scene, only "OBJECT" mode is available
            if _isBlender280:
                if not context.view_layer.objects.active:
                    context.view_layer.objects.active = scene.objects[0]
            else:
                if not scene.objects.active:
                    scene.objects.active = scene.objects[0]
            bpy.ops.object.mode_set(mode="OBJECT")
        # Also deselect the active object since the operator
        # <bpy.ops.object.select_all(action='DESELECT')> does not affect hidden objects and
        # the hidden active object
        if _isBlender280:
            if context.view_layer.objects.active:
                context.view_layer.objects.active.select_set(False)
        else:
            if scene.objects.active:
                scene.objects.active.select = False


class OperatorControlOverlay(bpy.types.Operator):
    bl_idname = "blender_osm.control_overlay"
    bl_label = ""
    bl_description = "Control overlay import and display progress in the 3D View"
    bl_options = {'INTERNAL'}
    
    lineWidth = 70 # in characters

    def modal(self, context, event):
        if event.type == 'TIMER':
            hasTiles = app.app.overlay.importNextTile()
            if not hasTiles:
                self.stop(context)
                if app.app.overlay.finalizeImport():
                    self.report({'INFO'}, "Overlay import is finished!")
                else:
                    self.report({'ERROR'}, "Probably something is wrong with the tile server!")
                # cleanup
                app.app.area = None
                return {'FINISHED'}

        return {'RUNNING_MODAL'}

    def execute(self, context):
        wm = context.window_manager
        
        self._drawHandle = bpy.types.SpaceView3D.draw_handler_add(
            self.drawMessage,
            tuple(),
            'WINDOW',
            'POST_PIXEL'
        )
        
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)
        
        return {'RUNNING_MODAL'}

    def stop(self, context):
        context.window_manager.event_timer_remove(self._timer)
        bpy.types.SpaceView3D.draw_handler_remove(self._drawHandle, 'WINDOW')
        app.app.stateMessage = None
        self._timer = None
        self._drawHandle = None

    def drawMessage(self):
        message = app.app.stateMessage
        if message:
            # draw message
            if _isBlender280:
                fontId = 0
                blf.color(fontId, 0., 1., 0., 1.)
            else:
                bgl.glColor4f(0., 1., 0., 1.)
            if len(message)<=self.lineWidth:
                self.drawLine(message, 60)
            else:
                for i,line in enumerate(reversed(textwrap.wrap(message, self.lineWidth))):
                    self.drawLine(line, 60+35*i)
    
    def drawLine(self, content, yPosition):
        fontId = 0
        blf.position(fontId, 15, yPosition, 0)
        blf.size(fontId, 20, 72)
        blf.draw(fontId, content)
    
    def setHeaderText(self, context):
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.header_text_set(app.app.stateMessage)
                app.app.stateMessage = ''
                return


_classes = (
    BlenderOsmPreferences,
    OperatorGetMapboxToken,
    OperatorLoadExtensions,
    OperatorImportData,
    OperatorControlOverlay
)

def register():
    for c in _classes:
        bpy.utils.register_class(c)
    gui.register()
    if app.app.has(Keys.mode3dRealistic):
        import realistic
        realistic.register()

def unregister():
    for c in _classes:
        bpy.utils.unregister_class(c)
    gui.unregister()
    if app.app.has(Keys.mode3dRealistic):
        import realistic
        realistic.unregister()