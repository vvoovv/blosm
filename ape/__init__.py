"""
APE stands for Asset Package Editor
"""

import os, json
from copy import deepcopy
from distutils.dir_util import copy_tree
from shutil import copyfile
import bpy
import bpy.utils.previews

from app import app


# the maximum file name length for a texture file
maxFileNameLength = 27


assetPackages = []
# the content of the current asset package
assetPackage = [0]
assetPackagesLookup = {}
imagePreviews = [0]
# a mapping between an asset attribute in a JSON file and the attribute of <BlosmApeProperties>
assetAttr2ApeAttr = {
    "category": "assetCategory",
    "type": "type",
    "name": "name",
    "path": "path",
    "part": "part",
    "featureWidthM": "featureWidthM",
    "featureLpx": "featureLpx",
    "featureRpx": "featureRpx",
    "numTilesU": "numTilesU",
    "numTilesV": "numTilesV",
    # it's required to perform conversion: false->0, true->0: int(false)=0, int(true)=1
    "cladding": ("useCladdingTexture", int),
    "material": "claddingMaterial",
    "textureWidthM": "textureWidthM",
    "part": "buildingPart"
}

defaults = dict(
    texture = dict(
        part = dict(
            category = "part",
            part = "level",
            type = "texture",
            featureWidthM = 1.,
            featureLpx = 0,
            featureRpx = 100,
            numTilesU = 2,
            numTilesV = 2,
            cladding = 0
        ),
        cladding = dict(
            category = "cladding",
            type = "texture",
            material = "concrete",
            textureWidthM = 1.
        )
    ),
    mesh = dict()
)


# values for <_changed>:
_edited = 1
_new = 2


# There are no edits if we simply change the selected building assets collection.
# See <updateBuilding(..)>
_ignoreEdits = False


def getAssetsDir(context):
    return app.getAssetsDir(context)


def getBuildingEntry(context):
    return assetPackage[0]["buildings"][int(context.scene.blosmApe.building)]


def getAssetInfo(context):
    return getBuildingEntry(context)["assets"][int(context.scene.blosmApe.buildingAsset)]


def _markBuildingEdited(buildingEntry):
    if _ignoreEdits:
        return
    if not buildingEntry["_changed"]:
        buildingEntry["_changed"] = _edited
    _markAssetPackageChanged()
    global _updateEnumBuildings
    _updateEnumBuildings = True


def _markAssetPackageChanged():
    if not assetPackage[0]["_changed"]:
        assetPackage[0]["_changed"] = 1
    

def updateAttributes(ape, assetInfo):
    category = assetInfo["category"]
    ape.assetCategory = category
    if category == "part":
        ape.buildingPart = assetInfo["part"]
        ape.featureWidthM = assetInfo["featureWidthM"]
        ape.featureLpx = assetInfo["featureLpx"]
        ape.featureRpx = assetInfo["featureRpx"]
        ape.numTilesU = assetInfo["numTilesU"]
        ape.numTilesV = assetInfo["numTilesV"]
        ape.useCladdingTexture = assetInfo["cladding"]
    elif category == "cladding":
        ape.claddingMaterial = assetInfo["material"]
        ape.textureWidthM = assetInfo["textureWidthM"]


_enumBuildings = []
_updateEnumBuildings = True
def getBuildings(self, context):
    global _updateEnumBuildings
    if _updateEnumBuildings:
        _enumBuildings.clear()
        _enumBuildings.extend(
            _getBuildingTuple(bldgIndex, bldg, context) for bldgIndex,bldg in enumerate(assetPackage[0]["buildings"])
        )
        _updateEnumBuildings = False
    return _enumBuildings

def _getBuildingTuple(bldgIndex, bldg, context):
    loadImagePreviews(bldg["assets"], context)
    # pick up the first asset
    assetInfo = bldg["assets"][0]
    return (
        str(bldgIndex),
        
        "%s%s%s%s" % (
            "[edit] " if bldg["_changed"]==_edited else ("[new] " if bldg["_changed"]==_new else ''),
            bldg["use"],
            " %s" % assetInfo["name"] if assetInfo["name"] else '',
            " (%s assets)" % len(bldg["assets"]) if len(bldg["assets"]) > 1 else ''
        ),
        
        "%s%s" % (bldg["use"], " %s" % assetInfo["name"] if assetInfo["name"] else ''),
        
        imagePreviews[0].get(os.path.join(assetInfo["path"], assetInfo["name"])).icon_id if assetInfo["name"] else 'BLANK1',
        
        # index is required to show the icons
        bldgIndex
    )


_enumBuildingAssets = []
def getBuildingAssets(self, context):
    _enumBuildingAssets.clear()
    buildingEntry = getBuildingEntry(context)
    
    # add assets
    loadImagePreviews(buildingEntry["assets"], context)
    _enumBuildingAssets.extend(
        (
            str(assetIndex),
            assetInfo["name"],
            assetInfo["name"],
            imagePreviews[0].get(os.path.join(assetInfo["path"], assetInfo["name"])).icon_id if assetInfo["name"] else 'BLANK1',
            # index is required to show the icons
            assetIndex
        ) for assetIndex, assetInfo in enumerate(buildingEntry["assets"])
    )
    return _enumBuildingAssets


class AssetPackageEditor:
    
    def drawApe(self, context):
        layout = self.layout
        ape = context.scene.blosmApe
        if not assetPackages:
            layout.operator("blosm.ape_load_ap_list")
            return
        
        if ape.state == "apNameEditor":
            self.drawApNameEditor(context)
        elif ape.state == "apSelection":
            self.drawApSelection(context)
        elif ape.state == "apEditor":
            self.drawApEditor(context)
    
    def drawApSelection(self, context):
        layout = self.layout
        ape = context.scene.blosmApe
        
        #layout.operator("blosm.ape_install_asset_package")
        row = layout.row()
        row.prop(ape, "assetPackage")
        row.operator("blosm.ape_edit_ap", text="Edit package")
        row.operator("blosm.ape_copy_ap", text="Copy")
        #row.operator("blosm.ape_update_asset_package", text="Update") # TODO
        row.operator("blosm.ape_edit_ap_name", text="Edit name")
        row.operator("blosm.ape_remove_ap", text="Remove")
        
        #layout.operator("blosm.ape_select_building")
    
    def drawApNameEditor(self, context):
        layout = self.layout
        ape = context.scene.blosmApe
        
        layout.prop(ape, "apDirName")
        layout.prop(ape, "apName")
        layout.prop(ape, "apDescription")
        
        row = layout.row()
        row.operator("blosm.ape_apply_ap_name")
        row.operator("blosm.ape_cancel")
    
    def drawApEditor(self, context):
        layout = self.layout
        ape = context.scene.blosmApe
        
        row = layout.row()
        row.label(
            text = "Asset package: %s%s" %
            (
                assetPackagesLookup[ape.assetPackage][1],
                " [edited]" if assetPackage[0]["_changed"] else ''
            )
        )
        row.operator("blosm.ape_save_ap")
        row.operator("blosm.ape_cancel")
        
        row = layout.row()
        row.prop(ape, "building")
        row2 = row.row(align=True)
        row2.operator("blosm.ape_add_building", text='', icon='FILE_NEW')
        row2.operator("blosm.ape_delete_building", text='', icon='PANEL_CLOSE')
        
        layout.separator(factor=3.0)
        
        layout.prop(ape, "buildingUse")
        
        assetInfo = getAssetInfo(context)
        
        #layout.prop(ape, "buildingAsset")
        box = layout.box()
        
        box.prop(ape, "showAdvancedOptions")
        
        assetIconBox = box.box()
        row = assetIconBox.row()
        row.template_icon_view(ape, "buildingAsset", show_labels=True)
        if ape.showAdvancedOptions:
            column = row.column(align=True)
            column.operator("blosm.ape_add_bldg_asset", text='', icon='ADD')
            column.operator("blosm.ape_delete_bldg_asset", text='', icon='REMOVE')
        self.drawPath(None, assetIconBox.row(), assetInfo, "path", "name")
        
        box.prop(ape, "assetCategory")
        
        if ape.assetCategory == "part":
            box.prop(ape, "buildingPart")
            box.prop(ape, "featureWidthM")
            box.prop(ape, "featureLpx")
            box.prop(ape, "featureRpx")
            box.prop(ape, "numTilesU")
            box.prop(ape, "numTilesV")
            box.prop(ape, "useCladdingTexture")
            
            if ape.showAdvancedOptions:
                self.drawPath("Specular map", box.row(), assetInfo, "specularMapPath", "specularMapName")
        elif ape.assetCategory == "cladding":
            box.prop(ape, "claddingMaterial")
            box.prop(ape, "textureWidthM")
    
    def drawPath(self, textureName, rowPath, assetInfo, pathAttr, nameAttr):
        if textureName:
            rowPath.label(text = "%s:" % textureName)
            rowPath.label(text = "%s/%s" % (assetInfo[pathAttr], assetInfo[nameAttr])\
                if assetInfo.get(nameAttr) else\
                "Select an asset:"
            )
        else:
            rowPath.label(text = "Path: %s/%s" % (assetInfo[pathAttr], assetInfo[nameAttr])\
                    if assetInfo.get(nameAttr) else\
                    rowPath.label(text = "Select an asset:")
            )
        op = rowPath.operator("blosm.ape_set_asset_path", icon='FILE_FOLDER')
        op.pathAttr = pathAttr
        op.nameAttr = nameAttr


class BLOSM_PT_DevApePanel(bpy.types.Panel, AssetPackageEditor):
    bl_label = "blosm"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "blosm ape"
    
    @classmethod
    def poll(cls, context):
        return not app.addonName in context.preferences.addons
    
    def draw(self, context):
        self.drawApe(context)


_enumAssetPackages = []
def getAssetPackages(self, context):
    _enumAssetPackages.clear()
    _enumAssetPackages.extend(
        (assetPackage[0], assetPackage[1], assetPackage[2]) for assetPackage in assetPackages
    )
    return _enumAssetPackages


def loadImagePreviews(imageList, context):
    for imageEntry in imageList:
        if not imageEntry["name"]:
            return
        # generates a thumbnail preview for a file.
        name = os.path.join(imageEntry["path"], imageEntry["name"])
        filepath = os.path.join(getAssetsDir(context), imageEntry["path"][1:], imageEntry["name"])
        if not imagePreviews[0].get(name) and os.path.isfile(filepath):
            imagePreviews[0].load(name, filepath, 'IMAGE')


#
# Update functions for <bpy.props.EnumProperty> fields
#

def updateBuilding(self, context):
    global _ignoreEdits
    _ignoreEdits = True
    
    buildingEntry = getBuildingEntry(context)
    self.buildingUse = buildingEntry["use"]
    self.buildingAsset = "0"
    
    _ignoreEdits = False
    #updateBuildingAsset(self, context)


def updateBuildingAsset(self, context):
    updateAttributes(
        self,
        getAssetInfo(context)
    )


def _updateAttribute(attr, self, context):
    assetInfo = getAssetInfo(context)
    
    attrValue = assetAttr2ApeAttr[attr][1]( getattr(self, assetAttr2ApeAttr[attr][0]) )\
        if isinstance(assetAttr2ApeAttr[attr], tuple) else\
        getattr(self, assetAttr2ApeAttr[attr])
    if attrValue != assetInfo[attr]:
        assetInfo[attr] = attrValue
        _markBuildingEdited( getBuildingEntry(context) )


def updateBuildingUse(self, context):
    buildingEntry = getBuildingEntry(context)
    
    if self.buildingUse != buildingEntry["use"]:
        buildingEntry["use"] = self.buildingUse
        _markBuildingEdited(buildingEntry)


def updateAssetCategory(self, context):
    assetInfo = getAssetInfo(context)
    
    category = self.assetCategory
    if category != assetInfo["category"]:
        path = assetInfo["path"]
        name = assetInfo["name"]
        assetInfo.clear()
        assetInfo.update(path=path, name=name)
        for a in defaults["texture"][category]:
            value = defaults["texture"][category][a]
            assetInfo[a] = value
            setattr(context.scene.blosmApe, assetAttr2ApeAttr[a], value)
        _markBuildingEdited( getBuildingEntry(context) )


def updateBuildingPart(self, context):
    _updateAttribute("part", self, context)

def updateFeatureWidthM(self, context):
    _updateAttribute("featureWidthM", self, context)

def updateFeatureLpx(self, context):
    _updateAttribute("featureLpx", self, context)

def updateFeatureRpx(self, context):
    _updateAttribute("featureRpx", self, context)

def updateUseCladdingTexture(self, context):
    _updateAttribute("cladding", self, context)

def updateNumTilesU(self, context):
    _updateAttribute("numTilesU", self, context)

def updateNumTilesV(self, context):
    _updateAttribute("numTilesV", self, context)

def updateTextureWidthM(self, context):
    _updateAttribute("textureWidthM", self, context)

def updateCladdingMaterial(self, context):
    _updateAttribute("material", self, context)


class BlosmApeProperties(bpy.types.PropertyGroup):
    
    assetPackage: bpy.props.EnumProperty(
        name = "Asset package",
        items = getAssetPackages,
        description = "Asset package for editing"
    )
    
    state: bpy.props.EnumProperty(
        name = "State",
        items = (
            ("apSelection", "asset package selection", "asset package selection"),
            ("apNameEditor", "asset package name editor", "asset package name editor"),
            ("apEditor", "asset package editor", "asset package editor")
        ),
        description = "Asset manager state",
        default = "apEditor" 
    )
    
    #
    # The properties for the asset package name editor
    #
    apDirName: bpy.props.StringProperty(
        name = "Folder",
        description = "Folder name for the asset package, it must be unique among the asset packages"
    )
    
    apName: bpy.props.StringProperty(
        name = "Name",
        description = "Name for the asset package"
    )
    
    apDescription: bpy.props.StringProperty(
        name = "Description",
        description = "Description for the asset package"
    )
    
    showAdvancedOptions: bpy.props.BoolProperty(
        name = "Show advanced options",
        description = "Show advanced options, for example to add an asset for the building asset collection",
        default = False
    )
    
    building: bpy.props.EnumProperty(
        name = "Building asset collection",
        items = getBuildings,
        description = "Building asset collection for editing",
        update = updateBuilding
    )
    
    buildingAsset: bpy.props.EnumProperty(
        name = "Asset entry",
        items = getBuildingAssets,
        description = "Asset entry for the selected building",
        update = updateBuildingAsset
    )
    
    #
    # The properties for editing a building asset collection
    #
    buildingUse: bpy.props.EnumProperty(
        name = "Building use",
        items = (
            ("apartments", "apartments", "Apartments"),
            ("single_family", "single family house", "Single family house"),
            ("office", "office", "Office building"),
            ("mall", "mall", "Mall"),
            ("retail", "retail", "Retail building"),
            ("hotel", "hotel", "Hotel"),
            ("school", "school", "School"),
            ("university", "university", "University"),
            ("any", "any building type", "Any building type")
        ),
        description = "Building usage",
        update = updateBuildingUse
    )
    
    assetCategory: bpy.props.EnumProperty(
        name = "Asset category",
        items = (
            ("part", "building part", "Building part"),
            ("cladding", "cladding", "Facade or roof cladding")
        ),
        description = "Asset category (building part or cladding)",
        update = updateAssetCategory
    )
    
    featureWidthM: bpy.props.FloatProperty(
        name = "Feature width in meters",
        unit = 'LENGTH',
        subtype = 'UNSIGNED',
        default = 1.,
        description = "The width in meters of the texture feature (for example, a window)",
        update = updateFeatureWidthM
    )
    
    featureLpx: bpy.props.IntProperty(
        name = "Feature left coordinate in pixels",
        subtype = 'PIXEL',
        description = "The left coordinate in pixels of the texture feature (for example, a window)",
        update = updateFeatureLpx
    )
    
    featureRpx: bpy.props.IntProperty(
        name = "Feature right coordinate in pixels",
        subtype = 'PIXEL',
        description = "The right coordinate in pixels of the texture feature (for example, a window)",
        update = updateFeatureRpx
    )
    
    useCladdingTexture: bpy.props.BoolProperty(
        name = "Has transparent parts to use cladding texture",
        default = False,
        update = updateUseCladdingTexture
    )
    
    numTilesU: bpy.props.IntProperty(
        name = "Number of tiles horizontally",
        subtype = 'UNSIGNED',
        description = "The number of tiles in the texture in the horizontal direction",
        min = 1,
        update = updateNumTilesU
    )
    
    numTilesV: bpy.props.IntProperty(
        name = "Number of tiles vertically",
        subtype = 'UNSIGNED',
        description = "The number of tiles in the texture in the vertical direction",
        min = 1,
        update = updateNumTilesV
    )
    
    claddingMaterial: bpy.props.EnumProperty(
        name = "Material",
        items = (
            ("brick", "brick", "brick"),
            ("plaster", "plaster", "plaster"),
            ("concrete", "concrete", "concrete"),
            ("metal", "metal", "metal"),
            ("glass", "glass", "glass"),
            ("gravel", "gravel", "gravel"),
            ("roof_tiles", "roof tiles", "roof tiles")
        ),
        description = "Material for cladding",
        update = updateCladdingMaterial
    )
    
    textureWidthM: bpy.props.FloatProperty(
        name = "Texture width in meters",
        unit = 'LENGTH',
        subtype = 'UNSIGNED',
        default = 1.,
        description = "The texture width in meters",
        update = updateTextureWidthM
    )
    
    buildingPart: bpy.props.EnumProperty(
        name = "Building part",
        items = (
            ("level", "level", "level"),
            ("curtain_wall", "curtain wall", "curtain wall")
        ),
        description = "Building part",
        update = updateBuildingPart
    )


###################################################
# Operators
###################################################

def writeJson(jsonObj, jsonFilepath):
    with open(jsonFilepath, 'w', encoding='utf-8') as jsonFile:
        json.dump(jsonObj, jsonFile, ensure_ascii=False, indent=4)

def getApListFilepath(context):
    return os.path.join(getAssetsDir(context), "asset_packages.json")


class BLOSM_OT_ApeLoadApList(bpy.types.Operator):
    bl_idname = "blosm.ape_load_ap_list"
    bl_label = "Load asset package list"
    bl_description = "Load the list of asset packages"
    bl_options = {'INTERNAL'}
    
    def execute(self, context):
        # check if we have a valid directory with assets
        try:
            app.validateAssetsDir(getAssetsDir(context))
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        
        assetPackages.clear()
        assetPackagesLookup.clear()
        
        assetPackages.extend( self.getApListJson(context)["assetPackages"] )
        assetPackagesLookup.update( (assetPackage[0],assetPackage) for assetPackage in assetPackages )
        
        context.scene.blosmApe.state = "apSelection"
        return {'FINISHED'}
    
    def getApListJson(self, context):
        apListFilepath = getApListFilepath(context)
        
        # check if the file with the list of asset packages exists
        if not os.path.isfile(apListFilepath):
            # create a JSON file with the default list of asset packages
            writeJson(
                dict(assetPackages = [("default", "default", "default asset package")]),
                apListFilepath
            )
        
        with open(apListFilepath, 'r') as jsonFile:
            apListJson = json.load(jsonFile)
        
        return apListJson


class BLOSM_OT_ApeEditAp(bpy.types.Operator):
    bl_idname = "blosm.ape_edit_ap"
    bl_label = "Edit asset package"
    bl_options = {'INTERNAL'}

    @classmethod
    def poll(cls, context):
        return assetPackagesLookup[context.scene.blosmApe.assetPackage][0] != "default"
    
    def execute(self, context):
        ape = context.scene.blosmApe
        
        with open(
            os.path.join(getAssetsDir(context), ape.assetPackage, "asset_info", "asset_info.json"),
            'r'
        ) as jsonFile:
            assetPackage[0] = json.load(jsonFile)
        
        assetPackage[0]["_changed"] = 0
        
        # mark all building asset collection as NOT changed
        for buildingEntry in assetPackage[0]["buildings"]:
            buildingEntry["_changed"] = 0
            if not "use" in buildingEntry:
                buildingEntry["use"] = "any"
        
        # pick up the building asset collection with the index 0
        buildingEntry = assetPackage[0]["buildings"][0]
        # the line <getBuildings(ape, context)> is required otherwise there is an error
        getBuildings(ape, context)
        # set the active building asset collection to element with the index 0
        ape.building = "0"
        # pick up the asset info with the index 0
        assetInfo = buildingEntry["assets"][0]
        ape.buildingUse = buildingEntry["use"]
        updateAttributes(ape, assetInfo)
        
        context.scene.blosmApe.state = "apEditor"
        
        global _updateEnumBuildings
        _updateEnumBuildings = True
        return {'FINISHED'}


class BLOSM_OT_ApeEditApName(bpy.types.Operator):
    bl_idname = "blosm.ape_edit_ap_name"
    bl_label = "Edit asset package name"
    bl_options = {'INTERNAL'}
    
    @classmethod
    def poll(cls, context):
        return assetPackagesLookup[context.scene.blosmApe.assetPackage][0] != "default"
    
    def execute(self, context):
        assetPackage = context.scene.blosmApe.assetPackage
        
        apInfo = assetPackagesLookup[assetPackage]
        context.scene.blosmApe.apDirName = assetPackage
        context.scene.blosmApe.apName = apInfo[1]
        context.scene.blosmApe.apDescription = apInfo[2]
        
        context.scene.blosmApe.state = "apNameEditor"
        return {'FINISHED'}


class BLOSM_OT_ApeCopyAp(bpy.types.Operator):
    bl_idname = "blosm.ape_copy_ap"
    bl_label = "Copy asset package"
    bl_options = {'INTERNAL'}
    
    def execute(self, context):
        # 'ap' stands for 'asset package'
        apDirName = context.scene.blosmApe.assetPackage
        assetsDir = getAssetsDir(context)
        sourceDir = os.path.join(assetsDir, apDirName)
        # find a name for the target directory
        counter = 1
        while True:
            apDirNameTarget = "%s_%s" % (apDirName, counter)
            targetDir = os.path.realpath( os.path.join(assetsDir, apDirNameTarget) ) 
            if os.path.isdir(targetDir):
                counter += 1
            else:
                break
        apInfo = assetPackagesLookup[apDirName]
        assetPackages.append([apDirNameTarget, "%s (copy)" % apInfo[1], "%s (copy)" % apInfo[2]])
        assetPackagesLookup[apDirNameTarget] = assetPackages[-1]
        writeJson( dict(assetPackages = assetPackages), getApListFilepath(context) )
        context.scene.blosmApe.assetPackage = apDirNameTarget
        # create a directory for the copy of the asset package <assetPackage>
        os.makedirs(targetDir)
        
        self.copyStyle(sourceDir, targetDir)
        
        self.copyAssetInfos(sourceDir, targetDir, apDirName)
        
        context.scene.blosmApe.apDirName = apDirNameTarget
        context.scene.blosmApe.apName = assetPackages[-1][1]
        context.scene.blosmApe.apDescription = assetPackages[-1][2]
        
        context.scene.blosmApe.state = "apNameEditor"
        
        return {'FINISHED'}
    
    def copyStyle(self, sourceDir, targetDir):
        copy_tree(
            os.path.join(sourceDir, "style"),
            os.path.join(targetDir, "style")
        )
    
    def copyAssetInfos(self, sourceDir, targetDir, apDirName):
        os.makedirs( os.path.join(targetDir, "asset_info") )
        
        # actually we copy only the <asset_info.json>
        
        # 'ai' stands for 'asset info'
        aiFilepathSource = os.path.join(sourceDir, "asset_info", "asset_info.json")
        aiFilepathTarget = os.path.join(targetDir, "asset_info", "asset_info.json")
        # open the source asset info file
        with open(aiFilepathSource, 'r') as aiFile:
            assetInfos = json.load(aiFile)
        self.processAssetInfos(assetInfos, apDirName)
        # write the target asset info file
        writeJson(assetInfos, aiFilepathTarget)
        
        # The old code to copy all JSON files with asset info is commented out:
        # iterate through JSON files in the sub-directory "asset_info" of <sourceDir>
        #for fileName in os.listdir( os.path.join(sourceDir, "asset_info") ):
        #    if os.path.splitext(fileName)[1] == ".json":
    
    def processAssetInfos(self, assetInfos, apDirName):
        """
        The method checks every building entry in <assetInfo> and
        then every 'part' in the building entry.
        If the field 'path' doesn't start with /, the prefix apDirName/ is added to the field 'path'
        """
        for bldgEntry in assetInfos["buildings"]:
            for assetInfo in bldgEntry["assets"]:
                path = assetInfo["path"]
                if path[0] != '/':
                    assetInfo["path"] = "/%s/%s" % (apDirName, path)


class BLOSM_OT_ApeInstallAssetPackage(bpy.types.Operator):
    bl_idname = "blosm.ape_install_asset_package"
    bl_label = "Install"
    bl_description = "Install asset package from a zip-file"
    bl_options = {'INTERNAL'}

    def execute(self, context):
        print("The asset package has been installed")
        return {'FINISHED'}


class BLOSM_OT_ApeUpdateAssetPackage(bpy.types.Operator):
    bl_idname = "blosm.ape_update_asset_package"
    bl_label = "Update asset package"
    bl_options = {'INTERNAL'}

    def execute(self, context):
        print("The asset package has been updated")
        return {'FINISHED'}


class BLOSM_OT_ApeCancel(bpy.types.Operator):
    bl_idname = "blosm.ape_cancel"
    bl_label = "Cancel"
    bl_description = "A generic operator for canceling"
    bl_options = {'INTERNAL'}

    def execute(self, context):
        context.scene.blosmApe.state = "apSelection"
        return {'FINISHED'}
    

class BLOSM_OT_ApeApplyApName(bpy.types.Operator):
    bl_idname = "blosm.ape_apply_ap_name"
    bl_label = "Apply"
    bl_description = "Apply asset package name"
    bl_options = {'INTERNAL'}

    def execute(self, context):
        blosmApe = context.scene.blosmApe
        apDirName = blosmApe.assetPackage
        apInfo = assetPackagesLookup[apDirName]
        
        isDirty = False
        if blosmApe.apDirName != apInfo[0]:
            if blosmApe.apDirName in assetPackagesLookup:
                self.report({'ERROR'}, "The folder '%s' for the asset package already exists" % blosmApe.apDirName)
                return {'CANCELLED'}
            try:
                assetsDir = getAssetsDir(context)
                os.rename(
                    os.path.join(assetsDir, apDirName),
                    os.path.join(assetsDir, blosmApe.apDirName)
                )
            except Exception as _:
                self.report({'ERROR'}, "Unable to create the folder '%s' for the asset package" % blosmApe.apDirName)
                return {'CANCELLED'}
            apInfo[0] = blosmApe.apDirName
            assetPackagesLookup[blosmApe.apDirName] = apInfo
            del assetPackagesLookup[apDirName]
            blosmApe.assetPackage = blosmApe.apDirName
            isDirty = True
        if apInfo[1] != blosmApe.apName:
            apInfo[1] = blosmApe.apName
            isDirty = True
        if apInfo[2] != blosmApe.apDescription:
            apInfo[2] = blosmApe.apDescription
            isDirty = True
        
        if isDirty:
            writeJson (dict(assetPackages = assetPackages), getApListFilepath(context) )
        
        context.scene.blosmApe.state = "apSelection"
        return {'FINISHED'}


class BLOSM_OT_ApeRemoveAp(bpy.types.Operator):
    bl_idname = "blosm.ape_remove_ap"
    bl_label = "Remove the asset package"
    bl_description = "Remove the asset package from the list. Its folder will remain intact"
    bl_options = {'INTERNAL'}

    @classmethod
    def poll(cls, context):
        return assetPackagesLookup[context.scene.blosmApe.assetPackage][0] != "default"

    def execute(self, context):
        # the directory name of an asset package serves as its id
        apDirName = context.scene.blosmApe.assetPackage
        apInfo = assetPackagesLookup[apDirName]
        del assetPackagesLookup[apDirName]
        assetPackages.remove(apInfo)
        # the asset package <default> is write protected
        context.scene.blosmApe.assetPackage = "default"
        self.report({'INFO'},
            "The asset package \"%s\" has been deleted from the list. Its directory remained intact" % apInfo[1]
        )
        writeJson( dict(assetPackages = assetPackages), getApListFilepath(context) )
        return {'FINISHED'}
    
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)


class BLOSM_OT_ApeSaveAp(bpy.types.Operator):
    bl_idname = "blosm.ape_save_ap"
    bl_label = "Save"
    bl_description = "Save the asset package"
    bl_options = {'INTERNAL'}
    
    def execute(self, context):
        if not self.validate():
            return {'FINISHED'}
        ap = deepcopy(assetPackage[0])
        self.cleanup(ap, True)
        path = os.path.join(getAssetsDir(context), context.scene.blosmApe.assetPackage, "asset_info", "asset_info.json")
        writeJson(
            ap,
            path
        )
        self.report({'INFO'}, "The asset package has been successfully saved to %s" % path)
        self.cleanup(assetPackage[0], False)
        global _updateEnumBuildings
        _updateEnumBuildings = True
        return {'FINISHED'}
    
    def validate(self):
        ap = assetPackage[0]
        for buildingEntry in ap["buildings"]:
            for assetInfo in buildingEntry["assets"]:
                if not (assetInfo["path"] and assetInfo["name"]):
                    self.report({'ERROR'},
                        "Unable to save: there is at least one asset without a valid path"
                    )
                    return False
        return True
    
    def cleanup(self, ap, deleteChanged):
        if "buildings" in ap:
            for buildingEntry in ap["buildings"]:
                if deleteChanged:
                    del buildingEntry["_changed"]
                else:
                    # just reset it
                    buildingEntry["_changed"] = 0
        if deleteChanged:
            del ap["_changed"]
        else:
            # just reset it
            ap["_changed"] = 0


class BLOSM_OT_ApeSelectBuilding(bpy.types.Operator):
    bl_idname = "blosm.ape_select_building"
    bl_label = "Select building entry"
    bl_options = {'INTERNAL'}
    bl_property = "buildingList"
    bl_options = {'INTERNAL'}

    buildingList: bpy.props.EnumProperty(
        name = "Building list",
        items = [('one', 'Any', "", 'PRESET', 1), ('two', 'PropertyGroup', "", 'PRESET', 2), ('three', 'type', "", 'PRESET', 3)]
    )
    
    def execute(self, context):
        print(self.buildingList)
        return {'FINISHED'}
    
    def invoke(self, context, event):
        context.window_manager.invoke_search_popup(self)
        return {'FINISHED'}


class BLOSM_OT_ApeAddBuilding(bpy.types.Operator):
    bl_idname = "blosm.ape_add_building"
    bl_label = "New"
    bl_description = "Add a new building asset collection"
    bl_options = {'INTERNAL'}
    
    def execute(self, context):
        ape = context.scene.blosmApe
        bldgIndex = len(assetPackage[0]["buildings"])
        # Create a building asset collection using the current values of <ape.buildingUse>
        assetInfo = defaults["texture"][ape.assetCategory].copy()
        assetInfo.update(name = '', path = '')
        buildingEntry = dict(
            use = ape.buildingUse,
            assets = [ assetInfo ],
            _changed = _new
        )
        assetPackage[0]["buildings"].append(buildingEntry)
        _enumBuildings.append( _getBuildingTuple(bldgIndex, buildingEntry, context) )
        ape.building = str(bldgIndex)
        
        _markAssetPackageChanged()
        return {'FINISHED'}


class BLOSM_OT_ApeDeleteBuilding(bpy.types.Operator):
    bl_idname = "blosm.ape_delete_building"
    bl_label = "Delete the building asset collection"
    bl_description = "Delete the building asset collection"
    bl_options = {'INTERNAL'}
    
    showConfirmatioDialog: bpy.props.BoolProperty(
        name = "Show this dialog",
        description = "Show this dialog to confirm the deletion of a building asset collection",
        default = True
    )
    
    def execute(self, context):
        buildingIndex = int(context.scene.blosmApe.building)
        del assetPackage[0]["buildings"][buildingIndex]
        context.scene.blosmApe.building = str(buildingIndex-1)\
            if len(assetPackage[0]["buildings"]) == buildingIndex else\
            context.scene.blosmApe.building
        #updateBuilding(context.scene.blosmApe, context)
        _markAssetPackageChanged()
        global _updateEnumBuildings
        _updateEnumBuildings = True
        return {'FINISHED'}
    
    def invoke(self, context, event):
        if self.showConfirmatioDialog:
            return context.window_manager.invoke_props_dialog(self)
        else:
            return self.execute(context)
    
    def draw(self, context):
        layout = self.layout
        layout.prop(self, "showConfirmatioDialog")


class BLOSM_OT_ApeAddBldgAsset(bpy.types.Operator):
    bl_idname = "blosm.ape_add_bldg_asset"
    bl_label = "Add"
    bl_description = "Add a building asset"
    bl_options = {'INTERNAL'}
    
    def execute(self, context):
        ape = context.scene.blosmApe
        buildingEntry = getBuildingEntry(context)
        
        assetIndex = len(buildingEntry["assets"])
        assetInfo = defaults["texture"][ape.assetCategory].copy()
        assetInfo.update(name = '', path = '')
        buildingEntry["assets"].append(assetInfo)
        
        _enumBuildingAssets.append( (str(assetIndex), '', '', 'BLANK1', assetIndex) )
        ape.buildingAsset = str(assetIndex)
        
        _markBuildingEdited(buildingEntry)
        return {'FINISHED'}


class BLOSM_OT_ApeDeleteBldgAsset(bpy.types.Operator):
    bl_idname = "blosm.ape_delete_bldg_asset"
    bl_label = "Delete"
    bl_description = "Delete the building asset"
    bl_options = {'INTERNAL'}
    
    @classmethod
    def poll(cls, context):
        return len(getBuildingEntry(context)["assets"]) > 1
    
    def execute(self, context):
        buildingEntry = getBuildingEntry(context)
        assetIndex = int(context.scene.blosmApe.buildingAsset)
        
        del buildingEntry["assets"][assetIndex]
        
        context.scene.blosmApe.buildingAsset = "0"
            
        _markBuildingEdited(buildingEntry)
        return {'FINISHED'}


class BLOSM_OT_ApeSetAssetPath(bpy.types.Operator):
    bl_idname = "blosm.ape_set_asset_path"
    bl_label = "Set path..."
    bl_description = "Set path to the asset"
    bl_options = {'INTERNAL'}
    
    filename: bpy.props.StringProperty()
    
    directory: bpy.props.StringProperty(
        subtype = 'FILE_PATH'
    )
    
    pathAttr: bpy.props.StringProperty()
    
    nameAttr: bpy.props.StringProperty()
    
    def execute(self, context):
        assetsDir = os.path.normpath(getAssetsDir(context))
        directory = os.path.normpath(self.directory)
        
        name = self.filename
        
        if len(name) > maxFileNameLength:
            self.report({'ERROR'}, "The maximum file name length is %s characters" % maxFileNameLength)
            return {'CANCELLED'}
        
        if directory.startswith(assetsDir):
            lenAssetsDir = len(assetsDir)
            if lenAssetsDir == len(directory):
                self.report({'ERROR'}, "The asset must be located in the folder of an asset package")
                return {'CANCELLED'}
            else:
                self.setAssetPath(
                    context,
                    "/".join( directory[lenAssetsDir:].split(os.sep) ),
                    name
                )    
        else:
            path = os.path.join(
                getAssetsDir(context),
                context.scene.blosmApe.assetPackage,
                "assets"
            )
            # The asset will be moved to the directory <path>
            if os.path.isfile( os.path.join(path, name) ):
                self.report({'INFO'},
                    ("The existing asset %s in the sub-bolder \"%s\" in your directory for assets " +\
                    "will be used instead of the selected asset.") % (name, path)
                )
            else:
                if not os.path.isdir(path):
                    os.makedirs(path)
                copyfile(
                    os.path.join(directory, name),
                    os.path.join(path, name)
                )
                self.report({'INFO'},
                    "The asset has been copied to the sub-folder \"%s\" in your directory for assets" %
                    os.path.join(context.scene.blosmApe.assetPackage, "assets")
                )
            self.setAssetPath(
                context,
                "/%s" % '/'.join( (context.scene.blosmApe.assetPackage, "assets") ),
                name
            )
            
        return {'FINISHED'}
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
    def setAssetPath(self, context, path, name):
        assetInfo = getAssetInfo(context)
        if path != assetInfo.get(self.pathAttr) or name != assetInfo.get(self.nameAttr):
            assetInfo[self.pathAttr] = path
            assetInfo[self.nameAttr] = name
            _markBuildingEdited(getBuildingEntry(context))


###################################################
# Registration
###################################################

_classes = (
    BLOSM_PT_DevApePanel,
    BlosmApeProperties,
    BLOSM_OT_ApeLoadApList,
    BLOSM_OT_ApeEditAp,
    BLOSM_OT_ApeEditApName,
    BLOSM_OT_ApeCopyAp,
    BLOSM_OT_ApeInstallAssetPackage,
    BLOSM_OT_ApeUpdateAssetPackage,
    BLOSM_OT_ApeCancel,
    BLOSM_OT_ApeApplyApName,
    BLOSM_OT_ApeRemoveAp,
    BLOSM_OT_ApeSaveAp,
    BLOSM_OT_ApeSelectBuilding,
    BLOSM_OT_ApeAddBuilding,
    BLOSM_OT_ApeDeleteBuilding,
    BLOSM_OT_ApeAddBldgAsset,
    BLOSM_OT_ApeDeleteBldgAsset,
    BLOSM_OT_ApeSetAssetPath
)

def register():
    for _class in _classes:
        bpy.utils.register_class(_class)
    
    bpy.types.Scene.blosmApe = bpy.props.PointerProperty(type=BlosmApeProperties)
    
    imagePreviews[0] = bpy.utils.previews.new()


def unregister():
    for _class in _classes:
        bpy.utils.unregister_class(_class)
    
    del bpy.types.Scene.blosmApe
    
    imagePreviews[0].close()
    imagePreviews.clear()