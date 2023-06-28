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
    "version": (2, 6, 6),
    "blender": (2, 80, 0),
    "location": "Right side panel > \"osm\" tab",
    "description": "One click download and import of OpenStreetMap, terrain, satellite imagery, web maps",
    "warning": "",
    "wiki_url": "https://github.com/vvoovv/blender-osm/wiki/Premium-Version",
    "tracker_url": "https://github.com/vvoovv/blender-osm/issues",
    "support": "COMMUNITY",
    "category": "Import-Export",
    "blosmAssets": "2021.05.07"
}

import os, sys, json, textwrap

# force cleanup of sys.modules to avoid conflicts with the other addons for Blender
for m in [
        "app", "building", "gui", "manager", "material", "parse", "realistic", "overlay",
        "renderer", "terrain", "util", "defs", "setup", "ape"
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
import gui, ape
import app.blender as blenderApp
from defs import Keys

# set the minimum version for BLOSM assets
blenderApp.app.setMinAssetsVersion(bl_info["blosmAssets"])
# set addon version
blenderApp.app.version = bl_info["version"]
blenderApp.app.isPremium = os.path.isdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "realistic"))


class BlosmPreferences(bpy.types.AddonPreferences, ape.AssetPackageEditor):
    bl_idname = __name__
    
    screenType: bpy.props.EnumProperty(
        name = "Screen type for addon preferences",
        items = (
            ("preferences", "preferences", "preferences"),
            ("ape", "asset package editor", "asset package editor")
        ),
        description = "Preferences of Asset Package Editor",
        default = "preferences"
    )
    
    dataDir: bpy.props.StringProperty(
        name = '',
        subtype = 'DIR_PATH',
        description = "Directory to store downloaded OpenStreetMap and terrain files"
    )
    
    assetsDir: bpy.props.StringProperty(
        name = '',
        subtype = 'DIR_PATH',
        description = "Directory with assets (building_materials.blend, vegetation.blend). "+
            "It can be also set in the addon GUI"
    )
    
    mapboxAccessToken: bpy.props.StringProperty(
        name = "Mapbox access token",
        description = "A string token to access overlays from Mapbox company"
    )
    
    arcgisAccessToken: bpy.props.StringProperty(
        name = "ArcGIS access token",
        description = "A string token (API Key) to access satellite imagery from ArcGIS location service"
    )
    
    googleMapsApiKey: bpy.props.StringProperty(
        name = "Google 3D Tiles Key",
        description = "A string token (API Key) to access 3D Tiles by Google"
    )
    
    osmServer: bpy.props.EnumProperty(
        name = "OSM data server",
        items = (
            ("overpass-api.de", "overpass-api.de: 8 cores, 128 GB RAM", "overpass-api.de: 8 cores, 128 GB RAM"),
            ("vk maps", "VK Maps: 56 cores, 384 GB RAM", "VK Maps: 56 cores, 384 GB RAM"),
            ("kumi.systems", "kumi.systems: 20 cores, 256 GB RAM", "kumi.systems: 20 cores, 256 GB RAM")
        ),
        description = "OSM data server if the default one is inaccessible",
        default = "overpass-api.de"
    )
    
    enableExperimentalFeatures: bpy.props.BoolProperty(
        name = "Enable export (experimental)",
        description = "Enable export to the popular 3D formats. Experimental feature! Use it with caution!",
        default = False
    )
    
    def draw(self, context):
        layout = self.layout
        
        if blenderApp.app.isPremium:
            layout.box().label(text="Thank you for purchasing the premium version!")
            if self.enableExperimentalFeatures:
                layout.row().prop(self, "screenType", expand=True)
        
        if self.screenType == "ape":
            self.drawApe(context)
        else:
            layout.label(text="Directory to store downloaded OpenStreetMap and terrain files:")
            layout.prop(self, "dataDir")
            
            if blenderApp.app.isPremium:
                layout.label(text="Directory with assets (building_materials.blend, vegetation.blend):")
                layout.prop(self, "assetsDir")
            
            layout.separator()
            layout.label(text="Paste one or more access tokens to get satellite imagery:")
            
            split = layout.split(factor=0.9)
            split.prop(self, "arcgisAccessToken")
            split.operator("blosm.get_arcgis_token", text="Get it!")
            
            split = layout.split(factor=0.9)
            split.prop(self, "mapboxAccessToken")
            split.operator("blosm.get_mapbox_token", text="Get it!")
            
            split = layout.split(factor=0.9)
            split.prop(self, "googleMapsApiKey")
            split.operator("blosm.get_google_maps_api_key", text="Get it!")
            
            layout.separator()
            layout.box().label(text="Advanced settings:")
            # Extensions might come later
            #layout.operator("blosm.load_extensions", text="Load extensions")
            layout.prop(self, "osmServer")
            
            if blenderApp.app.isPremium:
                layout.prop(self, "enableExperimentalFeatures", text="Enable experimental features")

blenderApp.app.addonName = BlosmPreferences.bl_idname


class BLOSM_OT_GetArcgisToken(bpy.types.Operator):
    bl_idname = "blosm.get_arcgis_token"
    bl_label = ""
    bl_description = "Get ArcGIS access token"
    bl_options = {'INTERNAL'}
    
    url = "https://developers.arcgis.com/sign-up/"
    
    def execute(self, context):
        import webbrowser
        webbrowser.open_new_tab(self.url)
        return {'FINISHED'}


class BLOSM_OT_GetMapboxToken(bpy.types.Operator):
    bl_idname = "blosm.get_mapbox_token"
    bl_label = ""
    bl_description = "Get Mapbox access token"
    bl_options = {'INTERNAL'}
    
    url = "https://www.mapbox.com/account/access-tokens"
    
    def execute(self, context):
        import webbrowser
        webbrowser.open_new_tab(self.url)
        return {'FINISHED'}


class BLOSM_OT_GetGoogleMapsApiKey(bpy.types.Operator):
    bl_idname = "blosm.get_google_maps_api_key"
    bl_label = ""
    bl_description = "Get Google 3D Tiles Key"
    bl_options = {'INTERNAL'}
    
    url = "https://developers.google.com/maps/documentation/tile/get-api-key"
    
    def execute(self, context):
        import webbrowser
        webbrowser.open_new_tab(self.url)
        return {'FINISHED'}

"""
class BLOSM_OT_LoadExtensions(bpy.types.Operator):
    bl_idname = "blosm.load_extensions"
    bl_label = ""
    bl_description = "Scan Blender addons, find extensions for blender-osm and load them"
    bl_options = {'INTERNAL'}
    
    def execute(self, context):
        numExtensions = blednerApp.app.loadExtensions(context)
        self.report({'INFO'},
            "No extension found" if not numExtensions else\
            ("Loaded 1 extension" if numExtensions==1 else "Loaded %s extensions" % numExtensions)
        )
        return {'FINISHED'}
"""


class BLOSM_OT_ImportData(bpy.types.Operator):
    """Import data: OpenStreetMap or terrain"""
    bl_idname = "blosm.import_data"  # important since its how bpy.ops.blosm.import_data is constructed
    bl_label = "blender-osm"
    bl_description = "Import data of the selected type (OpenStreetMap or terrain)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        a = blenderApp.app
        dataType = context.scene.blosm.dataType
        
        self.setup(context, a.addonName)
        
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
        elif dataType == "google-3d-tiles":
            return self.importGoogle3dTiles(context)
        elif dataType == "geojson":
            return self.importGeoJson(context)
        
        return {'FINISHED'}

    def setup(self, context, addonName):
        # check if the file <setup_execute.py> is available
        setup_function = self.loadSetupScript(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "setup_execute.py"),
            reportError=False
        )
        if setup_function:
            setup_function(context, addonName)
    
    def importOsm(self, context):
        a = blenderApp.app
        addon = context.scene.blosm
        
        try:
            a.initOsm(self, context)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        
        createFlatTerrain = a.mode is a.realistic and a.forests
        forceExtentCalculation = createFlatTerrain and a.osmSource == "file"
        
        setupScript = addon.setupScript
        if setupScript:
            setup_function = self.loadSetupScript(setupScript, reportError=True)
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
            # This is the case if <a.osmSource == "file"> and if the first condition is not true
            setLatLon = True
        
        osm.parse(a.osmFilepath, forceExtentCalculation=forceExtentCalculation)
        if a.loadMissingMembers and a.incompleteRelations:
            try:
                a.loadMissingWays(osm)
            except Exception as e:
                self.report({'ERROR'}, str(e))
                a.loadMissingMembers = False
            a.processIncompleteRelations(osm)
            if not osm.projection:
                # <osm.projection> wasn't set so far if there were only incomplete relations that
                # satisfy <osm.conditions>.
                # See also the comments in <parse.osm.__init__.py>
                # at the end of the method <osm.parse(..)>
                osm.setProjection( (osm.minLat+osm.maxLat)/2., (osm.minLon+osm.maxLon)/2. )
        
        if forceExtentCalculation:
            a.minLat = osm.minLat
            a.maxLat = osm.maxLat
            a.minLon = osm.minLon
            a.maxLon = osm.maxLon
        
        # Check if have a terrain Blender object set
        # At this point <a.projection> is set, so we can set the terrain
        a.setTerrain(
            context,
            createFlatTerrain = createFlatTerrain,
            createBvhTree = True
        )
        
        a.initLayers()
        
        a.process()
        a.render()
        
        # Set <lon> and <lat> attributes for <scene> if necessary.
        # <osm.projection> is set in <osm.setProjection(..)> along with <osm.lat> and <osm.lon>
        # So we test if <osm.projection> is set, that also means that <osm.lat> and <osm.lon> are also set.
        if setLatLon and osm.projection:
            # <osm.lat> and <osm.lon> have been set in osm.parse(..)
            self.setCenterLatLon(context, osm.lat, osm.lon)
        
        a.clean()
        
        return {'FINISHED'}
    
    def getCenterLatLon(self, context):
        a = blenderApp.app
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
    
    def loadSetupScript(self, setupScript, reportError):
        setupScript = os.path.realpath(bpy.path.abspath(setupScript))
        if not os.path.isfile(setupScript):
            if reportError:
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
        except Exception:
            self.report({'ERROR'},
                "Unable to execute the setup script! See the error message in the Blender console!"
            )
            return None
    
    def importTerrain(self, context):
        a = blenderApp.app
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
        a = blenderApp.app
        blosm = context.scene.blosm
        
        # find the Blender area holding 3D View
        for area in context.screen.areas:
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
        
        terrainObject = context.scene.objects.get(blosm.terrainObject)
        
        minLon, minLat, maxLon, maxLat = a.getExtentFromObject(terrainObject, context)\
            if terrainObject else\
            (a.minLon, a.minLat, a.maxLon, a.maxLat)
        
        a.overlay.prepareImport(minLon, minLat, maxLon, maxLat)
        
        if blosm.commandLineMode:
            hasTiles = True
            while hasTiles:
                hasTiles = blenderApp.app.overlay.importNextTile()
            if blenderApp.app.overlay.finalizeImport():
                self.report({'INFO'}, "Overlay import is finished!")
            else:
                self.report({'ERROR'}, "Probably something is wrong with the tile server!")
        else:
            bpy.ops.blosm.control_overlay()
        
        # set the custom parameters <lat> and <lon> to the active scene
        if setLatLon:
            self.setCenterLatLon(context, lat, lon)
        
        return {'FINISHED'}
    
    def importGoogle3dTiles(self, context):
        from threed_tiles.manager import BaseManager
        from threed_tiles.blender import BlenderRenderer
        
        renderer = BlenderRenderer()
        manager = BaseManager("https://tile.googleapis.com/v1/3dtiles/root.json", renderer)
        
        a = blenderApp.app
        try:
            a.init3dTiles(context, manager, "google")
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        
        manager.centerLat, manager.centerLon, setLatLonHeight = self.getCenterLatLon(context)
        sceneHasHeight = "height" in context.scene
        if not setLatLonHeight and sceneHasHeight:
            manager.centerHeight = context.scene["height"]
        else:
            manager.calculateCenterHeight = True
        
        manager.render(a.minLon, a.minLat, a.maxLon, a.maxLat)
        
        if setLatLonHeight:
            context.scene["lat"] = manager.centerLat
            context.scene["lon"] = manager.centerLon
        
        return {'FINISHED'}
    
    def importGpx(self, context):
        from parse.gpx import Gpx
        from gpx import GpxRenderer
        
        a = blenderApp.app
        try:
            a.initGpx(context, BlosmPreferences.bl_idname)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        
        scene = context.scene
        
        self.setObjectMode(context)
        bpy.ops.object.select_all(action='DESELECT')
        
        gpx = Gpx(a)
        
        if "lat" in scene and "lon" in scene and not a.ignoreGeoreferencing:
            gpx.setProjection(scene["lat"], scene["lon"])
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
        
        a = blenderApp.app
        addon = context.scene.blosm
        
        try:
            a.initGeoJson(self, context)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        
        createFlatTerrain = a.mode is a.realistic and a.forests
        forceExtentCalculation = createFlatTerrain
        
        # check if have a terrain Blender object set
        a.setTerrain(
            context,
            createFlatTerrain = createFlatTerrain,
            createBvhTree = True
        )
        
        setupScript = addon.setupScript
        if setupScript:
            setup_function = self.loadSetupScript(setupScript, reportError=True)
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
        
        data.parse(a.osmFilepath, forceExtentCalculation=forceExtentCalculation)
        
        if forceExtentCalculation:
            a.minLat = data.minLat
            a.maxLat = data.maxLat
            a.minLon = data.minLon
            a.maxLon = data.maxLon
        
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
            if not context.view_layer.objects.active:
                context.view_layer.objects.active = scene.objects[0]
            bpy.ops.object.mode_set(mode="OBJECT")
        # Also deselect the active object since the operator
        # <bpy.ops.object.select_all(action='DESELECT')> does not affect hidden objects and
        # the hidden active object
        if context.view_layer.objects.active:
            context.view_layer.objects.active.select_set(False)


class BLOSM_OT_ControlOverlay(bpy.types.Operator):
    bl_idname = "blosm.control_overlay"
    bl_label = ""
    bl_description = "Control overlay import and display progress in the 3D View"
    bl_options = {'INTERNAL'}
    
    lineWidth = 70 # in characters

    def modal(self, context, event):
        if event.type == 'TIMER':
            hasTiles = blenderApp.app.overlay.importNextTile()
            if not hasTiles:
                self.stop(context)
                if blenderApp.app.overlay.finalizeImport():
                    self.report({'INFO'}, "Overlay import is finished!")
                else:
                    self.report({'ERROR'}, "Probably something is wrong with the tile server!")
                # cleanup
                blenderApp.app.area = None
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
        blenderApp.app.stateMessage = None
        self._timer = None
        self._drawHandle = None

    def drawMessage(self):
        message = blenderApp.app.stateMessage
        if message:
            # draw message
            fontId = 0
            blf.color(fontId, 0., 1., 0., 1.)
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
                area.header_text_set(blenderApp.app.stateMessage)
                blenderApp.app.stateMessage = ''
                return


_classes = (
    BlosmPreferences,
    BLOSM_OT_GetArcgisToken,
    BLOSM_OT_GetMapboxToken,
    BLOSM_OT_GetGoogleMapsApiKey,
    #BLOSM_OT_LoadExtensions,
    BLOSM_OT_ImportData,
    BLOSM_OT_ControlOverlay
)

def register():
    for c in _classes:
        bpy.utils.register_class(c)
    gui.register()
    ape.register()
    if blenderApp.app.has(Keys.mode3dRealistic):
        import realistic
        realistic.register()

def unregister():
    for c in _classes:
        bpy.utils.unregister_class(c)
    gui.unregister()
    ape.unregister()
    if blenderApp.app.has(Keys.mode3dRealistic):
        import realistic
        realistic.unregister()