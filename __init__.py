"""
This file is a part of Blosm addon for Blender.
Copyright (C) 2014-2023 Vladimir Elistratov
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

import os, textwrap, tomllib

with open(os.path.join(os.path.dirname(__file__), "blender_manifest.toml"), "rb") as _file:
    _manifest = tomllib.load(_file)


import bpy, bmesh, blf

from .app import blender as blenderApp
# Important: <blenderApp.app> attributes must be set before importing gui, since <app.isPro> is used there
# set the minimum version for BLOSM assets
blenderApp.app.setMinAssetsVersion("2021.05.07")
# set addon version
blenderApp.app.version = tuple(int(part) for part in _manifest["version"].split('.'))
blenderApp.app.isPro = os.path.isdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "realistic"))

from .renderer import Renderer
from .parse.osm import Osm
from . import gui, ape
from .defs import Keys
from .progressive import blender as progressiveImport
from .downloads.access import OnlineAccessError


class BlosmPreferences(bpy.types.AddonPreferences, ape.AssetPackageEditor):
    bl_idname = __package__
    
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
        description = "Existing data directory required for every OSM import; other downloads use extension storage when empty"
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
        name = "Google 3D Tiles key",
        description = "A string token (API Key) to access 3D Tiles by Google"
    )

    cesiumIonAccessToken: bpy.props.StringProperty(
        name = "Cesium ion access token",
        description = "Cesium ion token with assets:read permission for Google Photorealistic 3D Tiles"
    )
    
    osmServer: bpy.props.EnumProperty(
        name = "OSM data server",
        items = (
            ("overpass-api.de", "overpass-api.de: 2 servers each with 16 cores, 128 GB RAM", "overpass-api.de: 2 servers each with 8 cores and 128 GB RAM"),
            ("private.coffee", "Private.coffee: 4 servers each with 20 cores, 256 GB RAM", "Private.coffee: 4 servers each with 20 cores and 256 GB RAM"),
            ("vk maps", "VK Maps: 2 servers each with 56 cores, 384 GB RAM", "VK Maps: 2 servers each with 56 cores and 384 GB RAM")
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
        
        if blenderApp.app.isPro:
            layout.box().label(text="Thank you for purchasing the Pro version!")
            if self.enableExperimentalFeatures:
                layout.row().prop(self, "screenType", expand=True)
        else:
            row = layout.box().row()
            row.operator("blosm.get_pro_version", text="Unlock more features with Blosm Pro!")
        
        if self.screenType == "ape":
            self.drawApe(context)
        else:
            layout.label(text="Data directory (empty uses Blender's extension storage):")
            layout.prop(self, "dataDir")
            
            if blenderApp.app.isPro:
                layout.label(text="Directory with assets (building_materials.blend, vegetation.blend):")
                layout.prop(self, "assetsDir")
            
            layout.separator()
            credentials = layout.box()
            credentials.label(text="Access tokens and keys")
            credentials.label(text="3D Tiles:")

            split = credentials.split(factor=0.9)
            split.prop(self, "googleMapsApiKey")
            split.operator("blosm.get_google_maps_api_key", text="Get it!")

            split = credentials.split(factor=0.9)
            split.prop(self, "cesiumIonAccessToken")
            split.operator("blosm.get_cesium_ion_token", text="Get it!")

            credentials.separator()
            credentials.label(text="Overlays:")
            
            split = credentials.split(factor=0.9)
            split.prop(self, "arcgisAccessToken")
            split.operator("blosm.get_arcgis_token", text="Get it!")
            
            split = credentials.split(factor=0.9)
            split.prop(self, "mapboxAccessToken")
            split.operator("blosm.get_mapbox_token", text="Get it!")
            
            layout.separator()
            layout.box().label(text="Advanced settings:")
            # Extensions might come later
            #layout.operator("blosm.load_extensions", text="Load extensions")
            layout.prop(self, "osmServer")
            
            if blenderApp.app.isPro:
                layout.prop(self, "enableExperimentalFeatures", text="Enable experimental features")

blenderApp.app.addonName = BlosmPreferences.bl_idname


class BLOSM_OT_GetArcgisToken(bpy.types.Operator):
    bl_idname = "blosm.get_arcgis_token"
    bl_label = ""
    bl_description = "Get ArcGIS API key"
    bl_options = {'INTERNAL'}
    
    url = "https://location.arcgis.com/sign-up/"
    
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


class BLOSM_OT_GetCesiumIonToken(bpy.types.Operator):
    bl_idname = "blosm.get_cesium_ion_token"
    bl_label = ""
    bl_description = "Get Cesium ion access token"
    bl_options = {'INTERNAL'}

    url = "https://ion.cesium.com/tokens"

    def execute(self, context):
        import webbrowser
        webbrowser.open_new_tab(self.url)
        return {'FINISHED'}


class BLOSM_OT_GetProVersion(bpy.types.Operator):
    bl_idname = "blosm.get_pro_version"
    bl_label = ""
    bl_description = "Get the Pro version of Blosm"
    bl_options = {'INTERNAL'}
    
    url = "https://prochitecture.com/blosm/pro/"
    
    def execute(self, context):
        import webbrowser
        webbrowser.open_new_tab(self.url)
        return {'FINISHED'}

"""
class BLOSM_OT_LoadExtensions(bpy.types.Operator):
    bl_idname = "blosm.load_extensions"
    bl_label = ""
    bl_description = "Scan Blender addons, find extensions for Blosm and load them"
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
    bl_label = "Blosm"
    bl_description = "Import data of the selected type (OpenStreetMap, Google 3D Tiles, terrain, etc)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return not progressiveImport.is_busy()

    def execute(self, context):
        if progressiveImport.is_busy():
            self.report({'WARNING'}, 'A Blosm import is already running')
            return {'CANCELLED'}
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
        elif dataType == "3d-tiles":
            return self.import3dTiles(context)
        
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
        from .app.osm_import import import_osm
        return import_osm(self, context)

    def getCenterLatLon(self, context):
        a = blenderApp.app
        scene = context.scene
        if "lat" in scene and "lon" in scene and a.relativeToInitialImport:
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
        try:
            from .util.setup_script import load_setup_script
            return load_setup_script(setupScript)
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
            return {'CANCELLED'}
        
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
            try:
                while hasTiles:
                    hasTiles = blenderApp.app.overlay.importNextTile()
            except OnlineAccessError as exc:
                self.report({'ERROR'}, str(exc))
                a.area = None
                return {'CANCELLED'}
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
    
    def import3dTiles(self, context):
        from .threed_tiles.manager import BaseManager
        from .threed_tiles.blender import BlenderRenderer
        
        addon = context.scene.blosm
        google3dTiles = addon.threedTilesSource == "google"
        viaIon = google3dTiles and addon.google3dTilesAccessVia == 'cesium_ion'
        if viaIon:
            from .threed_tiles.ion_access import CesiumIonManager
        progressive = addon.progressiveImport and not addon.commandLineMode
        if progressive:
            if bpy.app.background or context.window is None:
                self.report({'ERROR'}, 'Progressive import requires an interactive Blender window. Disable Progressive import for background jobs.')
                return {'CANCELLED'}
            if context.mode != 'OBJECT':
                self.report({'ERROR'}, 'Start progressive import in Object Mode')
                return {'CANCELLED'}
            from .threed_tiles.progressive import ProgressiveRenderer, TilesImport

        if not blenderApp.app.isPro and google3dTiles and addon.lodOf3dTiles == "lod6":
            def _draw_pro_popup(self, context):
                self.layout.label(text="This level of detail is available in the Pro version")
                self.layout.separator()
                self.layout.operator("blosm.get_pro_version", text="Get the Pro version!")

            bpy.context.window_manager.popup_menu(
                _draw_pro_popup,
                title="Blosm",
                icon='INFO'
            )
            return {'CANCELLED'}
        
        renderer = (ProgressiveRenderer if progressive else BlenderRenderer)(
            "Google 3D Tiles" if google3dTiles else "3D Tiles",
            addon.join3dTilesObjects,
            addon.instanceName,
            report=(lambda levels, message: progressiveImport.notify(next(iter(levels)), message)) if progressive else self.report
        )
        manager = (CesiumIonManager if viaIon else BaseManager)(
            "https://tile.googleapis.com/v1/3dtiles/root.json" if google3dTiles else addon.threedTilesUrl,
            renderer
        )
        manager.concurrentDownloads = addon.concurrentDownloads
        manager.downloadWorkers = addon.downloadWorkers
        
        a = blenderApp.app
        try:
            a.init3dTiles(context, manager)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        
        manager.centerLat, manager.centerLon, setLatLonHeight = self.getCenterLatLon(context)
        
        if not setLatLonHeight and "height_offset" in context.scene:
            renderer.heightOffset = context.scene["height_offset"]
        else:
            renderer.calculateHeightOffset = True
        
        if progressive:
            try:
                task = TilesImport(manager, (a.minLon, a.minLat, a.maxLon, a.maxLat), context, setLatLonHeight)
                progressiveImport.start(task, context)
            except Exception as exc:
                self.report({'ERROR'}, str(exc))
                return {'CANCELLED'}
            return {'FINISHED'}

        result = manager.render(a.minLon, a.minLat, a.maxLon, a.maxLat)
        
        if len(result) == 1:
            # a critical error happend
            self.report({'ERROR'}, result[0])
            return {'CANCELLED'}
        
        numImportedTiles, errors = result
        if numImportedTiles:
            if setLatLonHeight:
                context.scene["lat"] = manager.centerLat
                context.scene["lon"] = manager.centerLon
            
            if renderer.calculateHeightOffset:
                context.scene["height_offset"] = renderer.heightOffset
                
            self.report(
                {'INFO'},
                "Imported %s 3D Tiles. %s errors occured during the import. " % (numImportedTiles, len(errors)) +\
                "See the System Console."\
                    if errors else\
                    "Successfully imported %s 3D Tiles." % numImportedTiles
            )
        else:
            self.report(
                {'ERROR'},
                "No 3D tiles were imported. %s errors occured during the import. " % len(errors) +\
                "See the System Console."
            )
        
        if errors:
            for error in errors:
                print(error)
        
        return {'FINISHED'} if numImportedTiles else {'CANCELLED'}
    
    def importGpx(self, context):
        from .parse.gpx import Gpx
        from .gpx import GpxRenderer
        
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
        
        if "lat" in scene and "lon" in scene and a.relativeToInitialImport:
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
            try:
                hasTiles = blenderApp.app.overlay.importNextTile()
            except OnlineAccessError as exc:
                self.stop(context)
                blenderApp.app.area = None
                self.report({'ERROR'}, str(exc))
                return {'CANCELLED'}
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
        if progressiveImport.is_busy():
            self.report({'WARNING'}, 'A Blosm import is already running')
            return {'CANCELLED'}
        wm = context.window_manager
        
        self._drawHandle = bpy.types.SpaceView3D.draw_handler_add(
            self.drawMessage,
            tuple(),
            'WINDOW',
            'POST_PIXEL'
        )
        
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)
        blenderApp.app.overlay_running = True
        
        return {'RUNNING_MODAL'}

    def stop(self, context):
        blenderApp.app.overlay_running = False
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
        blf.size(fontId, 20)
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
    BLOSM_OT_GetCesiumIonToken,
    BLOSM_OT_GetProVersion,
    #BLOSM_OT_LoadExtensions,
    BLOSM_OT_ImportData,
    BLOSM_OT_ControlOverlay
)

def register():
    progressiveImport.register()
    for c in _classes:
        bpy.utils.register_class(c)
    gui.register()
    ape.register()
    if blenderApp.app.has(Keys.mode3dRealistic):
        from .import realistic
        realistic.register()
    # Pro-only: register optional baking tools if available
    if blenderApp.app.isPro:
        try:
            from .util.blender_extra import baking as _baking
            if hasattr(_baking, 'register'):
                _baking.register()
        except Exception:
            # optional module; ignore if missing
            pass

def unregister():
    progressiveImport.unregister()
    for c in _classes:
        bpy.utils.unregister_class(c)
    gui.unregister()
    ape.unregister()
    if blenderApp.app.has(Keys.mode3dRealistic):
        from . import realistic
        realistic.unregister()
    if blenderApp.app.isPro:
        try:
            from .util.blender_extra import baking as _baking
            if hasattr(_baking, 'unregister'):
                _baking.unregister()
        except Exception:
            pass
