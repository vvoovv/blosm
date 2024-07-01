"""
This file is a part of Blosm addon for Blender.
Copyright (C) 2014-2023 Vladimir Elistratov, Alain (al1brn)
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
import os, math, webbrowser, json
from bpy.app.handlers import persistent
from app.blender import app
from defs import Keys

_has3dRealistic = app.has(Keys.mode3dRealistic)

if _has3dRealistic:
    from realistic.material.renderer import FacadeWithColor


def getDataTypes():
        items = [
            ("osm", "OpenStreetMap", "OpenStreetMap"),
            ("terrain", "terrain", "Terrain"),
            ("overlay", "image overlay", "Image overlay for the terrain, e.g. satellite imagery or a map"),
            ("3d-tiles", "3D Tiles", "Photorealistic 3D Tiles by Google and other providers. " +\
                "3D Tiles is an open standard by the Open Geospatial Consortium (OGC) " +\
                "for streaming massive 3D geospatial content such as Photogrammetry and 3D Buildings"
            ),
            ("gpx", "gpx", "GPX tracks")
        ]
        if app.has(Keys.geojson):
            items.append(("geojson", "GeoJson", "GeoJson"))
        return items


_blenderMaterials = (
    ("appartments", "fo"),
    ("appartments", "fs"),
    ("office", "fo"),
    ("office", "fs"),
    ("glass", "fs"),
    ("neoclassical", "fo"),
    ("neoclassical", "fs"),
    ("brick", "ms"),
    ("plaster", "ms"),
    ("metal", "ms"),
    ("roof_tiles", "ms"),
    ("concrete", "ms"),
    ("gravel", "ms")
)

# default number of levels and its relative weight
_defaultLevels = (
    (4, 10),
    (5, 40),
    (6, 10)
)


def getBlenderMaterials(self, context):
    materialType = context.scene.blosm.materialType
    return tuple((m[0], m[0], m[0]) for m in _blenderMaterials if m[1] == materialType)


_gnSetups2d = [('-', '', '')]
def updateGnBlendFile2d(self, context):
    _gnSetups2d.clear()
    
    filepath = os.path.realpath(
        bpy.path.abspath(self.gnBlendFile2d)
    )
    if os.path.isfile(filepath):
        with bpy.data.libraries.load(filepath) as (data_from, data_to):
            if data_from.node_groups:
                _gnSetups2d.extend((gnSetup, gnSetup, gnSetup) for gnSetup in data_from.node_groups)
    else:
        _gnSetups2d.append(('-', '', ''))
        self.gnSetup2d = '-'


def getGnSetups2d(self, context):
    return _gnSetups2d


def addDefaultLevels():
    defaultLevels = bpy.context.scene.blosm.defaultLevels
    if not defaultLevels:
        for n, w in _defaultLevels:
            e = defaultLevels.add()
            e.levels = n
            e.weight = w

# This handler is needed to set the defaults for <context.scene.blosm.defaultLevels>
# just after the addon registration
def _onRegister(scene):
    addDefaultLevels()
    # the handler isn't needed anymore, so we remove it
    bpy.app.handlers.scene_update_post.remove(_onRegister)
def _onRegister280():
    addDefaultLevels()
    return
# This handler is needed to set the defaults for <context.scene.blosm.defaultLevels>
# after each start of Blender or reloading the start-up file via Ctrl N or loading any Blender file.
# That's why the persistent decorator is used
@persistent
def _onFileLoaded(scene):
    addDefaultLevels()


class BLOSM_UL_DefaultLevels(bpy.types.UIList):
    
    def draw_item(self, context, layout, data, item, icon, active_data, active_property):
        row = layout.row()
        row.prop(item, "levels")
        row.prop(item, "weight")


class BlosmDefaultLevelsEntry(bpy.types.PropertyGroup):
    levels: bpy.props.IntProperty(
        subtype='UNSIGNED',
        min = 1,
        max = 50,
        default = 5,
        description="Default number of levels"
    )
    weight: bpy.props.IntProperty(
        subtype='UNSIGNED',
        min = 1,
        max = 100,
        default = 10,
        description="Weight between 1 and 100 for the default number of levels"
    )


class BLOSM_OT_ReloadAssetPackageList(bpy.types.Operator):
    bl_idname = "blosm.reload_asset_package_list"
    bl_label = "Reload asset package list"
    bl_description = "Reload the list of asset packages"
    bl_options = {'INTERNAL'}
    
    def invoke(self, context, event):
        # check if all needed paths do exist
        assetsDir = app.getAssetsDir(context)
        try:
            app.validateAssetsDir(assetsDir)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        
        # additional validation
        if not os.path.isfile( os.path.join(assetsDir, "asset_packages.json") ):
            self.report({'ERROR'}, "The assets directory doesn't contain the file \"asset_packages.json\"")
            _enumAssetPackages.clear()
            return {'CANCELLED'}
        
        if len( app.getAssetPackageList(assetsDir) ) == 1:
            self.report({'WARNING'}, "There is only the default asset package. Use the Asset Package Editor to add one")
        
        global _loadAssetPackages
        _loadAssetPackages = True
        return {'FINISHED'}


class BLOSM_OT_SelectExtent(bpy.types.Operator):
    bl_idname = "blosm.select_extent"
    bl_label = "select"
    bl_description = "Select extent for your area of interest on a geographical map"
    bl_options = {'INTERNAL'}
    
    url = "http://prochitecture.com/blender-osm/extent/"
    
    def invoke(self, context, event):
        bv = bpy.app.version
        av = app.version
        isPremium = "premium" if app.isPremium else ""
        webbrowser.open_new_tab(
            "%s?blender_version=%s.%s&addon=blosm&addon_version=%s%s.%s.%s" %
            (self.url, bv[0], bv[1], isPremium, av[0], av[1], av[2])
        )
        return {'FINISHED'}


class BLOSM_OT_PasteExtent(bpy.types.Operator):
    bl_idname = "blosm.paste_extent"
    bl_label = "paste"
    bl_description = "Paste extent (chosen on the geographical map) for your area of interest from the clipboard"
    bl_options = {'INTERNAL', 'UNDO'}
    
    def invoke(self, context, event):
        addon = context.scene.blosm
        coords = context.window_manager.clipboard
        
        if not coords:
            self.report({'ERROR'}, "Nothing to paste!")
            return {'CANCELLED'}
        try:
            # parse the string from the clipboard to get coordinates of the extent
            coords = tuple( map(lambda s: float(s), coords[(coords.find('=')+1):].split(',')) )
            if len(coords) != 4:
                raise ValueError
        except ValueError:
            self.report({'ERROR'}, "Invalid string to paste!")
            return {'CANCELLED'}
        
        addon.minLon = coords[0]
        addon.minLat = coords[1]
        addon.maxLon = coords[2]
        addon.maxLat = coords[3]
        return {'FINISHED'}


class BLOSM_OT_ExtentFromActive(bpy.types.Operator):
    bl_idname = "blosm.extent_from_active"
    bl_label = "from active"
    bl_description = "Use extent from the active Blender object"
    bl_options = {'INTERNAL', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        scene = context.scene
        return context.object and context.object.type == "MESH" and "lat" in scene and "lon" in scene
    
    def invoke(self, context, event):
        scene = context.scene
        addon = scene.blosm
        
        if not app.projection:
            from util.transverse_mercator import TransverseMercator
            app.projection = TransverseMercator(lat=scene["lat"], lon=scene["lon"])
        
        addon.minLon, addon.minLat, addon.maxLon, addon.maxLat = app.getExtentFromObject(
            context.object,
            context
        )
        
        return {'FINISHED'}


class BLOSM_OT_ReplaceMaterials(bpy.types.Operator):
    bl_idname = "blosm.replace_materials"
    bl_label = "Replace Materials"
    bl_description = "Replace materials for the selected objects"
    bl_options = {'INTERNAL', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        return context.selected_objects
    
    def invoke(self, context, event):
        addon = context.scene.blosm
        if addon.replaceMaterialsWith == "export-ready":
            self.replaceWithExportReadyMaterials(context)
        else: # if addon.replaceMaterialsWith == "custom":
            materialName = addon.replacementMaterial
            if not materialName:
                self.report({'ERROR'}, "Material for the replacement is not set!")
                return {'CANCELLED'}
            elif not materialName in bpy.data.materials:
                self.report({'ERROR'}, "Material name for the replacement is not correct!")
                return {'CANCELLED'}
            
            templateMaterial = bpy.data.materials[materialName]
            imageTextureNodeName = ''
            if templateMaterial.use_nodes and templateMaterial.node_tree.nodes:
                # # check if there is an Image Texture node with the label "BASE COLOR"
                for node in templateMaterial.node_tree.nodes:
                    if node.type == 'TEX_IMAGE' and node.label == "BASE COLOR":
                        imageTextureNodeName = node.name
                        break
            self.replaceMaterialsWithTemplate(templateMaterial, imageTextureNodeName, context)
            
        return {'FINISHED'}
    
    def replaceWithExportReadyMaterials(self, context):
        # create a template material for export
        templateMaterial = bpy.data.materials.new("Material for Export")
        templateMaterial.use_nodes = True
        imageTextureNode = templateMaterial.node_tree.nodes.new('ShaderNodeTexImage')
        imageTextureNode.label = "BASE COLOR"
        imageTextureNode.location = (-300, 300)
        # create links
        templateMaterial.node_tree.links.new(
            templateMaterial.node_tree.nodes["Principled BSDF"].inputs['Base Color'],
            imageTextureNode.outputs['Color']
        )
        
        self.replaceMaterialsWithTemplate(templateMaterial, imageTextureNode.name, context)
    
    def replaceMaterialsWithTemplate(self, templateMaterial, imageTextureNodeName, context):
        for obj in [obj for obj in context.selected_objects if obj.type == 'MESH']:
            for slot in obj.material_slots:
                if slot and slot.material:
                    material = templateMaterial.copy()
                    currentMaterial = slot.material
                    if imageTextureNodeName and currentMaterial.use_nodes and currentMaterial.node_tree.nodes:
                        # find the Image Texture node with the label "BASE COLOR"
                        for node in currentMaterial.node_tree.nodes:
                            if node.type == 'TEX_IMAGE' and node.label == "BASE COLOR":
                                material.node_tree.nodes[imageTextureNodeName].image = node.image
                                break
                    slot.material = material


class BLOSM_OT_Gn2d_Info(bpy.types.Operator):
    bl_idname = "blosm.gn2d_info"
    bl_label = "Geometry Nodes"
    bl_description = "Information on applying Geometry Nodes to imported building footprints"
    bl_options = {'INTERNAL'}
    
    def invoke(self, context, event):
        webbrowser.open_new_tab("https://github.com/vvoovv/blosm/wiki/Applying-Geometry-Nodes-to-Building-Footprints")
        return {'FINISHED'}


class BLOSM_OT_LevelsAdd(bpy.types.Operator):
    bl_idname = "blosm.default_levels_add"
    bl_label = "+"
    bl_description = "Add an entry for the default number of levels. " +\
        "Enter both the number of levels and its relative weight between 1 and 100"
    bl_options = {'INTERNAL'}
    
    def invoke(self, context, event):
        context.scene.blosm.defaultLevels.add()
        return {'FINISHED'}


class BLOSM_OT_LevelsDelete(bpy.types.Operator):
    bl_idname = "blosm.default_levels_delete"
    bl_label = "-"
    bl_description = "Delete the selected entry for the default number of levels"
    bl_options = {'INTERNAL'}
    
    @classmethod
    def poll(cls, context):
        return len(context.scene.blosm.defaultLevels) > 1
    
    def invoke(self, context, event):
        addon = context.scene.blosm
        defaultLevels = addon.defaultLevels
        defaultLevels.remove(addon.defaultLevelsIndex)
        if addon.defaultLevelsIndex >= len(defaultLevels):
            addon.defaultLevelsIndex = 0
        return {'FINISHED'}


class BLOSM_PT_Copyright(bpy.types.Panel):
    bl_label = "Copyright"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "Blosm"
    
    @classmethod
    def poll(cls, context):
        return context.scene.blosm.copyright
    
    def draw(self, context):
        self.layout.label(text = context.scene.blosm.copyright)


class BLOSM_PT_Extent(bpy.types.Panel):
    bl_label = "Blosm"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "Blosm"

    def draw(self, context):
        layout = self.layout
        addon = context.scene.blosm
        
        if (addon.dataType == "osm" and addon.osmSource == "server") or\
            (addon.dataType == "overlay" and not bpy.data.objects.get(addon.terrainObject)) or\
            addon.dataType == "terrain" or\
            addon.dataType == "3d-tiles" or\
            (addon.dataType == "geojson" and addon.coordinatesAsFilter):
            box = layout.box()
            row = box.row()
            row.alignment = "CENTER"
            row.label(text="Extent:")
            row = box.row(align=True)
            row.operator("blosm.select_extent")
            row.operator("blosm.paste_extent")
            row.operator("blosm.extent_from_active")
            
            split = box.split(factor=0.25)
            split.label(text="")
            split.split(factor=0.67).prop(addon, "maxLat")
            row = box.row()
            row.prop(addon, "minLon")
            row.prop(addon, "maxLon")
            split = box.split(factor=0.25)
            split.label(text="")
            split.split(factor=0.67).prop(addon, "minLat")
        
        box = layout.box()
        row = box.row(align=True)
        row.prop(addon, "dataType", text="")
        row.operator("blosm.import_data", text="import")


class PanelRealisticTools():#(bpy.types.Panel):
    bl_label = "Realistic mode"
    bl_space_type = "VIEW_3D"
    bl_region_type = "TOOLS"
    #bl_context = "objectmode"
    bl_category = "Blosm"
    
    @classmethod
    def poll(cls, context):
        addon = context.scene.blosm
        return _has3dRealistic and addon.dataType == "osm"\
            and addon.mode == "3Drealistic"
    
    def draw(self, context):
        layout = self.layout
        addon = context.scene.blosm

        layout.prop(addon, "treeDensity")
        
        box = layout.box()
        row = box.row(align=True)
        row.prop(addon, "makeRealisticLayer", text="")
        row.operator("blosm.make_realistic")
        box.operator("blosm.flatten_selected")
        
        layout.operator("blosm.make_polygon")


class BLOSM_PT_Settings(bpy.types.Panel):
    bl_label = "Settings"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "Blosm"
    
    def draw(self, context):
        addon = context.scene.blosm
        
        dataType = addon.dataType
        if dataType == "osm":
            self.drawOsm(context)
        elif dataType == "terrain":
            self.drawTerrain(context)
        elif dataType == "overlay":
            self.drawOverlay(context)
        elif dataType == "3d-tiles":
            self.draw3dTiles(context)
        elif dataType == "gpx":
            self.drawGpx(context)
        elif dataType == "geojson":
            self.drawOsm(context)
    
    def drawOsm(self, context):
        layout = self.layout
        addon = context.scene.blosm
        mode3dRealistic = _has3dRealistic and addon.mode == "3Drealistic"
        isOsm = addon.dataType == "osm"
        isGeoJson = addon.dataType == "geojson"
        
        box = layout.box()
        if isOsm:
            box.prop(addon, "osmSource", text="Import from")
        if isGeoJson or addon.osmSource == "file":
            box.prop(addon, "osmFilepath", text="File")
        if isGeoJson:
            box.prop(addon, "coordinatesAsFilter")
        
        layout.box().prop_search(addon, "terrainObject", context.scene, "objects")
            
        layout.prop(addon, "mode", expand=True)
        
        prefs = context.preferences.addons
        enableExperimentalFeatures = (app.addonName in prefs and prefs[app.addonName].preferences.enableExperimentalFeatures) or not app.addonName in prefs
        
        if enableExperimentalFeatures and mode3dRealistic:

            split = layout.box().split(factor=0.5)
            split.label(text="Asset package:")
            row = split.row()
            row.prop(addon, "assetPackage", text='')
            row.operator("blosm.reload_asset_package_list", icon='FILE_REFRESH', text='')
            
            layout.box().prop(addon, "importForExport")
            
            box = layout.box()
            box.box().prop(addon, "buildings")
            
            box = box.box()
            box.prop(addon, "forests", text="Import forests and separate trees")
            if addon.forests:
                box.prop(addon, "treeDensity")
            
            layout.box().prop(addon, "singleObject")
            
            layout.box().prop(addon, "relativeToInitialImport")
            
            box = layout.box()
            box.label(text = "[Advanced]:")
            
            split = box.split(factor=0.5)
            split.label(text="Setup script:")
            split.prop(addon, "setupScript", text="")
        else:
            if mode3dRealistic:
                box = layout.box()
                box.prop(addon, "buildings")
                if addon.buildings:
                    box.prop(addon, "litWindows")
                    box.separator()
                    self._drawBuildingSettings(box, addon)
                
                # forests and trees
                box = layout.box()
                box.prop(addon, "forests", text="Import forests and separate trees")
                if addon.forests:
                    box.prop(addon, "treeDensity")
            else:
                box = layout.box()
                box.prop(addon, "buildings")
                box.prop(addon, "water")
                box.prop(addon, "forests")
                box.prop(addon, "vegetation")
                box.prop(addon, "highways")
                box.prop(addon, "railways")
            
            if not mode3dRealistic and addon.mode != "2D":
                self._drawBuildingSettings(layout.box(), addon)
            
            #box.prop(addon, "straightAngleThreshold")
            
            layout.box().prop(addon, "singleObject")
            
            if addon.mode == "2D" and addon.singleObject:
                box = layout.box()
                row = box.row()
                row.label(text="[Geometry Nodes]:")
                row.operator("blosm.gn2d_info", text='', icon='QUESTION')
                box.prop(addon, "gnBlendFile2d", text="File")
                box.prop(addon, "gnSetup2d", text="Name")
                
                if addon.gnSetup2d != '-':
                    self._drawDefaultLevels(box, addon)
            
            layout.box().prop(addon, "relativeToInitialImport")
            
            box = layout.box()
            box.label(text = "[Advanced]:")
            split = box.split(factor=0.5)
            split.label(text="Setup script:")
            split.prop(addon, "setupScript", text="")
    
    def _drawBuildingSettings(self, box, addon):
        split = box.split(factor=0.67)
        split.label(text="Default roof shape:")
        split.prop(addon, "defaultRoofShape", text="")
        box.prop(addon, "levelHeight")
        
        self._drawDefaultLevels(box, addon)
    
    def _drawDefaultLevels(self, box, addon):
        column = box.column()
        split = column.split(factor=0.67, align=True)
        split.label(text="Default number of levels:")
        split.operator("blosm.default_levels_add")
        split.operator("blosm.default_levels_delete")
        
        column.template_list(
            "BLOSM_UL_DefaultLevels", "",
            addon, "defaultLevels", addon, "defaultLevelsIndex",
            rows=3
        )
    
    def drawTerrain(self, context):
        addon = context.scene.blosm
        
        self.layout.box().prop(addon, "relativeToInitialImport")
        
        # Vertex reduction: reduction ratio selection + total vertex computation
        count = 3600 // int(addon.terrainResolution)
        lat = int((addon.maxLat - addon.minLat) * count)
        lon = int((addon.maxLon - addon.minLon) * count)
        ratio = int(addon.terrainReductionRatio)
        verts = (lat//ratio+1)*(lon//ratio+1)
        
        box = self.layout.box()
        box.label(text="[Advanced] Vertex reduction")
        box.prop(addon, "terrainReductionRatio")
        box.label(text="Vertex count: {:,d}".format(verts))
    
    def drawOverlay(self, context):
        layout = self.layout
        addon = context.scene.blosm
        
        layout.box().prop_search(addon, "terrainObject", context.scene, "objects")
        
        box = layout.box()
        box.prop(addon, "overlayType")
        if addon.overlayType == "custom":
            #box = layout.box()
            box.label(text="Paste overlay URL here:")
            box.prop(addon, "overlayUrl")
        
        box = layout.box()
        box.prop(addon, "setOverlayMaterial")
        box.prop(addon, "saveOverlayToFile")
        
        box = self.layout.box()
        box.label(text="[Advanced]")
        split = box.split(factor=0.7)
        split.label(text="Max number of tiles:")
        split.prop(addon, "maxNumTiles", text='')
        _squreTextureSize = 256 * round( math.sqrt(addon.maxNumTiles) )
        box.label(text="Like a square texture {:,d}x{:,d} px".format(_squreTextureSize, _squreTextureSize))
    
    def draw3dTiles(self, context):
        layout = self.layout
        addon = context.scene.blosm
        
        box = layout.box()
        
        box.row().prop(addon, "threedTilesType", expand=True)
        
        if addon.threedTilesType == "custom":
            box.label(text="Paste URL of root tile here:")
            box.prop(addon, "threedTilesUrl")
        
        box.label(text="Level of details:")
        box.prop(addon, "lodOf3dTiles", text='')
        layout.prop(addon, "join3dTilesObjects")
        layout.prop(addon, "relativeToInitialImport")
    
    def drawGpx(self, context):
        layout = self.layout
        addon = context.scene.blosm
        
        layout.box().prop(addon, "gpxFilepath", text="File")
        
        layout.box().prop_search(addon, "terrainObject", context.scene, "objects")
        
        layout.box().prop(addon, "gpxProjectOnTerrain")
        
        layout.box().prop(addon, "relativeToInitialImport")


class BLOSM_PT_Tools(bpy.types.Panel):
    bl_label = "Tools"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "Blosm"
    
    @classmethod
    def poll(cls, context):
        return context.scene.blosm.dataType == "3d-tiles" and context.selected_objects
    
    def draw(self, context):
        layout = self.layout
        addon = context.scene.blosm
        
        layout.label(text="Replace materials with")
        layout.row().prop(addon, "replaceMaterialsWith", expand=True)
        if addon.replaceMaterialsWith == "custom":
            layout.prop_search(addon, "replacementMaterial", bpy.data, "materials")
        layout.operator("blosm.replace_materials")


class BLOSM_PT_BpyProj(bpy.types.Panel):
    bl_label = "Projection"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "Blosm"
    
    @classmethod
    def poll(cls, context):
        return "bpyproj" in context.preferences.addons
    
    def draw(self, context):
        import bpyproj
        bpyproj.draw(context, self.layout)


_loadAssetPackages = False
_enumAssetPackages = []
def getAssetPackages(self, context):
    global _loadAssetPackages
    if _loadAssetPackages:
        # It has been just tested in the operator <blosm.reload_asset_package_list>
        # that the file does exist
        apList = app.getAssetPackageList( app.getAssetsDir(context) )
        _enumAssetPackages.clear()
        _enumAssetPackages.extend(tuple(apInfo) for apInfo in apList)
        _loadAssetPackages = False
    elif not _enumAssetPackages:
        _enumAssetPackages.append(("default", "default", "default"))
    return _enumAssetPackages


class BlosmProperties(bpy.types.PropertyGroup):
    
    commandLineMode: bpy.props.BoolProperty(
        name = "Command line mode",
        description = "The option specifies if the addon is used in the command line mode",
        default = False
    )
    
    instanceName: bpy.props.StringProperty(
        name = "Instance Name",
        description = "Instance name for running multiple instances of Blender",
        default = ''
    )
    
    terrainObject: bpy.props.StringProperty(
        name = "Terrain",
        description = "Blender object for the terrain"
    )
    
    osmSource: bpy.props.EnumProperty(
        name = "Import OpenStreetMap from",
        items = (
            ("server", "server", "remote server"),
            ("file", "file", "file on the local disk")
        ),
        description = "From where to import OpenStreetMap data: remote server or a file on the local disk",
        default = "server"
    )
    
    osmFilepath: bpy.props.StringProperty(
        name = "OpenStreetMap file",
        subtype = 'FILE_PATH',
        description = "Path to an OpenStreetMap file for import"
    )
    
    dataType: bpy.props.EnumProperty(
        name = "Data",
        items = getDataTypes(),
        description = "Data type for import",
        default = "osm"
    )
    
    mode: bpy.props.EnumProperty(
        name = "Mode: 3D realistic, 3D simple or 2D"\
            if _has3dRealistic else\
            "Mode: 3D or 2D",
        items = (
                ("3Drealistic","3D realistic","3D realistic"),
                ("3Dsimple","3D simple","3D simple"),
                ("2D","2D","2D")
            ) if _has3dRealistic else\
            (("3Dsimple","3D","3D"), ("2D","2D","2D")),
        description = ("Import data with textures and 3D objects (3D realistic) " +
            "or without them (3D simple) " +
            "or 2D only")\
            if _has3dRealistic else\
            "Import data in 3D or 2D mode",
        default = "3Drealistic" if _has3dRealistic else "3Dsimple"
    )
    
    importForExport: bpy.props.BoolProperty(
        name = "Import for export",
        description = "Import OpenStreetMap buildings ready for export to the popular 3D formats",
        default = False
    )
    
    # extent bounds: minLat, maxLat, minLon, maxLon
    
    minLat: bpy.props.FloatProperty(
        name="min lat",
        description="Minimum latitude of the imported extent",
        precision = 4,
        min = -89.,
        max = 89.,
        default= 40.7463 if _has3dRealistic else 51.33
    )

    maxLat: bpy.props.FloatProperty(
        name="max lat",
        description="Maximum latitude of the imported extent",
        precision = 4,
        min = -89.,
        max = 89.,
        default= 40.7523 if _has3dRealistic else 51.33721
    )

    minLon: bpy.props.FloatProperty(
        name="min lon",
        description="Minimum longitude of the imported extent",
        precision = 4,
        min = -180.,
        max = 180.,
        default= -73.9892 if _has3dRealistic else 12.36902
    )

    maxLon: bpy.props.FloatProperty(
        name="max lon",
        description="Maximum longitude of the imported extent",
        precision = 4,
        min = -180.,
        max = 180.,
        default= -73.9812 if _has3dRealistic else 12.37983
    )
    
    coordinatesAsFilter: bpy.props.BoolProperty(
        name = "Use coordinates as filter",
        description = "Use coordinates as a filter for the import from the file",
        default = False
    )
    
    buildings: bpy.props.BoolProperty(
        name = "Import buildings",
        description = "Import building outlines",
        default = True
    )
    
    water: bpy.props.BoolProperty(
        name = "Import water objects",
        description = "Import water objects (rivers and lakes)",
        default = True
    )
    
    forests: bpy.props.BoolProperty(
        name = "Import forests",
        description = "Import forests and woods",
        default = not _has3dRealistic
    )
    
    vegetation: bpy.props.BoolProperty(
        name = "Import other vegetation",
        description = "Import other vegetation (grass, meadow, scrub)",
        default = True
    )
    
    highways: bpy.props.BoolProperty(
        name = "Import roads and paths",
        description = "Import roads and paths",
        default = True
    )
    
    railways: bpy.props.BoolProperty(
        name = "Import railways",
        description = "Import railways",
        default = False
    )
    
    defaultRoofShape: bpy.props.EnumProperty(
        items = (("flat", "flat", "flat shape"), ("gabled", "gabled", "gabled shape")),
        description = "Roof shape for a building if the roof shape is not set in OpenStreetMap",
        default = "flat"
    )
    
    singleObject: bpy.props.BoolProperty(
        name = "Import as a single object",
        description = "Import OSM objects as a single Blender mesh objects instead of separate ones",
        default = True
    )

    relativeToInitialImport: bpy.props.BoolProperty(
        name = "Relative to initial import",
        description = "Import relative to the initial import if it is available",
        default = True
    )
    
    levelHeight: bpy.props.FloatProperty(
        name = "Level height",
        description = "Average height of a level in meters to use for OSM tags building:levels and building:min_level",
        default = 3.
    )
    
    defaultLevels: bpy.props.CollectionProperty(type = BlosmDefaultLevelsEntry)
    
    defaultLevelsIndex: bpy.props.IntProperty(
        subtype='UNSIGNED',
        default = 0,
        description = "Index of the active entry for the default number of levels"
    )
    
    straightAngleThreshold: bpy.props.FloatProperty(
        name = "Straight angle threshold",
        description = "Threshold for an angle of the building outline: when consider it as straight one. "+
            "It may be important for calculation of the longest side of the building outline for a gabled roof.",
        default = 175.5,
        min = 170.,
        max = 179.95,
        step = 10 # i.e. step/100 == 0.1
    )
    
    loadMissingMembers: bpy.props.BoolProperty(
        name = "Load missing members of relations",
        description = "Relation members aren't contained in the OSM file " +
            "if they are located outside of the OSM file extent. " +
            "Enable this option to load the missiong members of the relations " +
            "either from a local file (if available) or from the server.",
        default = True
    )
    
    subdivide: bpy.props.BoolProperty(
        name = "Subdivide curves, flat layers",
        description = "Subdivide Blender curves representing roads and paths and " +
        "polygons representing flat layers (water, forest, vegetation) " +
        "to project them on the terrain correctly",
        default = True
    )
    
    subdivisionSize: bpy.props.FloatProperty(
        name = "Subdivision size",
        description = "Subdivision size in meters",
        default = 10.,
        min = 5.,
        step = 100 # i.e. step/100 == 1.
    )
    
    # Terrain settings
    # SRTM3 data are sampled at either 3 arc-second and contain 1201 lines and 1201 samples
    # or 1 arc-second and contain 3601 lines and 3601 samples
    terrainResolution: bpy.props.EnumProperty(
        name="Resolution",
        items=(("1", "1 arc-second", "1 arc-second"), ("3", "3 arc-second", "3 arc-second")),
        description="Spation resolution",
        default="1"
    )
    
    terrainPrimitiveType: bpy.props.EnumProperty(
        name="Mesh primitive type: quad or triangle",
        items=(("quad","quad","quad"),("triangle","triangle","triangle")),
        description="Primitive type used for the terrain mesh: quad or triangle",
        default="quad"
    )

    # Number of vertex reduction
    # The Reduction Ratio is a divider of 1200
    terrainReductionRatio: bpy.props.EnumProperty(
        name="Ratio",
        items=(
            ("1","100%","No reduction"),
            ("2","25%","Divide par 4"),
            ("5","4%","Divide by 25"),
            ("10","1%","Divide by 100"),
            ("20","0.25%","Divide by 400"),
            ("50","0.04%","Divide by 2500"),
            ("100","0.01%","Divide by 10,000"),
            ("200","0.0025%","Divide by 40,000"),
            ("400","0.0006%","Divide by 160,000"),
            ("600","0.0003%","Divide by 360,000"),
            ("1200","0.0001%","Divide by 1,440,000")
        ),
        description="Reduction ratio for the number of vertices",
        default="1"
    )
    
    #
    # Overlay settings
    #
    overlayType: bpy.props.EnumProperty(
        name = "Overlay",
        items = (
            #("bing-aerial", "Bing Aerial", "Bing Aerial"),
            ("arcgis-satellite", "ArcGIS Satellite", "ArcGIS Satellite"),
            ("mapbox-satellite", "Mapbox Satellite", "Mapbox Satellite"),
            ("osm-mapnik", "OSM Mapnik", "OpenStreetMap Mapnik"),
            ("mapbox-streets", "Mapbox Streets", "Mapbox Streets"),
            ("custom", "Custom URL", "A URL template for the custom image overlay")
        ),
        description = "Image overlay type",
        default = "arcgis-satellite"
    )
    
    overlayUrl: bpy.props.StringProperty(
        name = '',
        description = "URL for the custom image overlay. Use {z}/{x}/{y} in the URL or "+
            "Mapbox style URL mapbox://styles/... "+
            "See http://wiki.openstreetmap.org/wiki/Slippy_map_tilenames for details about "+
            "the URL format."
    )
    
    setOverlayMaterial: bpy.props.BoolProperty(
        name = "Set default material",
        description = "Set the default Cycles material and " +
            "use the image overlay in the \"Image Texture\" node",
        default = True
    )
    
    saveOverlayToFile: bpy.props.BoolProperty(
        name = "Save overlay to file",
        description = "Save the image overlay to a local file if checked or " +
            "pack it to the current .blend file otherwise",
        default = False
    )
    
    maxNumTiles: bpy.props.IntProperty(
        name = "Maximum number of overlay tiles",
        subtype = 'UNSIGNED',
        min = 128,
        default = 256,
        description = "Maximum number of overlay tiles. Each tile has dimensions 256x256 pixels"
    )
    
    ####################################
    # Settings for the 3D Tiles import
    ####################################
    
    threedTilesType: bpy.props.EnumProperty(
        name = "Type of 3D Tiles",
        items = (
            ("google", "Google", "3D Tiles by Google (similar to the 3D content in Google Maps or Google Earth)"),
            ("custom", "Custom", "3D Tiles defined by a custom URL of the root tile")
        ),
        description = "Type of 3D Tiles",
        default = "google"   
    )
    
    threedTilesUrl: bpy.props.StringProperty(
        name = '',
        description = "URL of the root tile"
    )
    
    lodOf3dTiles: bpy.props.EnumProperty(
        name = "Level of details",
        items = (
            ("lod1", "whole city", "whole city"),
            ("lod2", "districts", "districts"),
            ("lod3", "groups of buildings", "groups of buildings"),
            ("lod4", "separate buildings", "separate buildings"),
            ("lod5", "buildings with details", "buildings with details"),
            ("lod6", "buildings with more details", "buildings with more details")
        ),
        description = "Choose a level of details (LoD)",
        default = "lod3"
    )
    
    join3dTilesObjects: bpy.props.BoolProperty(
        name = "Join 3D Tiles objects",
        description = "Join 3D Tiles objects and remove double vertices",
        default = True
    )
    
    copyright: bpy.props.StringProperty(
        name = "Copyright message",
        description = "Copyright for the imported content",
        default = ''
    )
    
    cacheJsonFiles: bpy.props.BoolProperty(
        name = "Cache JSON Files",
        description = "Cache JSON Files that define tilesets",
        default = False
    )
    
    cache3dFiles: bpy.props.BoolProperty(
        name = "Cache 3D Files",
        description = "Cache 3D Files (for example in .glb format)",
        default = False
    )
    
    replaceMaterialsWith: bpy.props.EnumProperty(
        name = "Replace materials with",
        items = (
            ("export-ready", "export-ready", "replace with export-ready materials"),
            ("custom", "custom", "replace with custom materials based on the given one")
        ),
        description = "Replace materials for the selected objects",
        default = "export-ready"
    )
    
    replacementMaterial: bpy.props.StringProperty(
        name = "Material",
        description = "A custom material to replace materials with for the selected objects"
    )
    
    ####################################
    # Settings for the GPX track import
    ####################################
    
    gpxFilepath: bpy.props.StringProperty(
        name = "GPX file",
        subtype = 'FILE_PATH',
        description = "Path to a GPX file for import"
    )
    
    gpxProjectOnTerrain: bpy.props.BoolProperty(
        name="Project GPX-track on terrain",
        description="Project GPX-track on the terrain if checked or use elevations from GPX-track for z-coordinate otherwise",
        default=True
    )
    
    gpxImportType: bpy.props.EnumProperty(
        name = "Import as curve or mesh",
        items = (
            ("curve", "curve", "Blender curve"),
            ("mesh", "mesh", "Blender mesh")
        ),
        description = "Import as Blender curve or mesh",
        default = "curve"
    )
    
    ####################################
    # Settings for the realistic 3D mode
    ####################################
    setupScript: bpy.props.StringProperty(
        name = "Setup script",
        subtype = 'FILE_PATH',
        description = "Path to a setup script. Leave blank for default."
    )
    
    assetsDir: bpy.props.StringProperty(
        name = "Directory with assets",
        subtype = 'DIR_PATH',
        description = "Directory with assets (building_materials.blend, vegetation.blend). "+
            "Overrides the one from the addon settings"
    )
    
    treeDensity: bpy.props.IntProperty(
        name = "Trees per hectare",
        description = "Number of trees per hectare (10,000 square meters, " +
            "e.g. a plot 100m x 100m) for forests",
        min = 1,
        subtype = 'UNSIGNED',
        default = 500
    )
    
    makeRealisticLayer: bpy.props.EnumProperty(
        name = "\"Make realistic\" layer",
        items = (
            ("water", "water", "water"),
            ("forest", "forest", "forest"),
            ("meadow", "meadow", "meadow")
        ),
        description = "A layer for the operator \"Make realistic\"",
        default = "water"
    )
    
    litWindows: bpy.props.IntProperty(
        name = "Percentage of lit windows",
        description = "Percentage of lit windows for a building",
        min = 0,
        max = 100,
        subtype = 'UNSIGNED',
        default = 0,
        update = FacadeWithColor.updateLitWindows if _has3dRealistic else None
    )
    
    assetPackage: bpy.props.EnumProperty(
        name = "Asset package",
        items = getAssetPackages,
        description = "Asset package to use for a scene import"
    )
    
    #    
    # A group of properties for Blender material utilities
    #
    
    materialType: bpy.props.EnumProperty(
        name = "Material type",
        items = (
            # <fo> stands for 'facade with overlays'
            ("fo", "facades with window overlays", 
                "Seamless textures for wall cladding, windows are placed on walls via overlay textures"),
            # <fs> stands for 'facade seamless'
            ("fs", "seamless facades",
                "Seamless facade textures with windows"),
            # <ms> stands for 'material seamless'
            ("ms", "seamless cladding",
                "Seamless cladding textures for walls and roofs")
            #("custom", "custom script", "Custom script")
        ),
        description = "Type of Blender materials to create",
        default = "fo"
    )
    
    blenderMaterials: bpy.props.EnumProperty(
        name = "Blender materials",
        items = getBlenderMaterials,
        description = "A group of Blender materials to create"
    )
    
    wallTexture: bpy.props.StringProperty(
        name = "Wall texture",
        subtype = 'FILE_PATH',
        description = "Path to a wall texture, that must be listed in the Blender text data-block \"wall_textures\""
    )
    
    listOfTextures: bpy.props.StringProperty(
        name = "List of textures",
        description = "A list of textures to download from textures.com"
    )
    
    materialScript: bpy.props.StringProperty(
        name = "Script",
        description = "A Python script to generate materials with selected textures"
    )
    
    #
    # Properties for a Geometry Nodes setup applied to building footprints
    #
    gnBlendFile2d: bpy.props.StringProperty(
        name = "File with Geometry Nodes",
        subtype = 'FILE_PATH',
        description = "Path to a Blender file with a Geometry Nodes setup applied to building footprints",
        update = updateGnBlendFile2d
    )
    
    gnSetup2d: bpy.props.EnumProperty(
        name = "Geometry Nodes setup",
        items = getGnSetups2d,
        description = "A Geometry Nodes setup applied to building footprints"
    )


_classes = (
    BLOSM_UL_DefaultLevels,
    BlosmDefaultLevelsEntry,
    BLOSM_OT_ReloadAssetPackageList,
    BLOSM_OT_SelectExtent,
    BLOSM_OT_PasteExtent,
    BLOSM_OT_ExtentFromActive,
    BLOSM_OT_ReplaceMaterials,
    BLOSM_OT_Gn2d_Info,
    BLOSM_OT_LevelsAdd,
    BLOSM_OT_LevelsDelete,
    BLOSM_PT_Copyright,
    BLOSM_PT_Extent,
    BLOSM_PT_Settings,
    BLOSM_PT_Tools,
    BLOSM_PT_BpyProj,
    BlosmProperties
)

def register():
    for c in _classes:
        bpy.utils.register_class(c)
    # a group for all GUI attributes related to Blosm
    bpy.types.Scene.blosm = bpy.props.PointerProperty(type=BlosmProperties)
    bpy.app.timers.register(_onRegister280, first_interval=2)
    # see the notes near the code for <_onFileLoaded>
    bpy.app.handlers.load_post.append(_onFileLoaded)

def unregister():
    for c in _classes:
        bpy.utils.unregister_class(c)
    del bpy.types.Scene.blosm
    bpy.app.handlers.load_post.remove(_onFileLoaded)