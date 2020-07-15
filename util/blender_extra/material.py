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
from app import app
from defs import Keys

_isBlender280 = bpy.app.version[1] >= 80


# <FO> means 'facade with overlay'
_materialFamilyFO = (
    "",
    "_emission",
    "_ground_level",
    "_ground_level_emission"
)

# the name of the custom node for a Blender material for a facade with overlay;
# the order of values in the Python tuple <_customNodeFO> corresponds to the one in
# the Python tuple <_materialFamilyFO>
_customNodeFO = 2*("FacadeOverlay",) + 2*("FacadeOverlayGroundLevel",)


# <MS> means 'simple seamless material'
_materialFamilyMS = (
    "",
    "_color",
    "_scaled",
    "_scaled_color"
)


def createMaterialFromTemplate(materialTemplate, materialName):
    m = materialTemplate.copy()
    m.name = materialName
    m.use_fake_user = True
    
    return m.node_tree.nodes


def createFacadeMaterialsForSeamlessTextures(
        files,
        directory,
        listOfTextures,
        materialBaseName1,
        materialBaseName2,
        materialTemplate1,
        materialTemplate2,
        # additional image textures as kwargs
        **kwargs
    ):
    def createMaterials(materialBaseName, materialTemplate):
        materialName = "%s.%s" % (materialBaseName, (i+1))
        if not materialName in bpy.data.materials:
            nodes = createMaterialFromTemplate(materialTemplate, materialName)
            
            setImage(fileName, directory, nodes, "Image Texture")
            # set additional image textures
            for suffix in kwargs:
                setImage(fileName, directory, nodes, kwargs[suffix], suffix)
            node = nodes["FacadePart"]
            setCustomNodeValue(node, "Number of Tiles U", textureDataEntry[0])
            setCustomNodeValue(node, "Number of Tiles V", textureDataEntry[1])
            setCustomNodeValue(node, "Tile Size U Default", textureDataEntry[2])
    
    textureData = readTextures(listOfTextures)
    
    materialTemplate1 = bpy.data.materials.get(materialTemplate1)
    materialTemplate2 = bpy.data.materials.get(materialTemplate2)
    
    for i,fileName in enumerate(files):
        fileName = fileName.name
        textureDataEntry = textureData.get(fileName)
        if textureDataEntry:
            if materialTemplate1:
                createMaterials(materialBaseName1, materialTemplate1)
            if materialTemplate2:
                createMaterials(materialBaseName2, materialTemplate2)
        else:
            print(
                ("Information about the image texture \"%s\" isn't available " +
                "in the list of textures \"%s\"") % (fileName, listOfTextures)
            )


def createMaterialsForFacadesOverlay(
        files, directory, materialBaseName, listOfTextures, baseMaterialTemplate,
        wallMaterial, wallTexturePath, wallTextureWidthM, wallTextureHeightM
    ):
    
    textureData = readTextures(listOfTextures)
    
    materialTemplates = tuple("%s%s_template" % (baseMaterialTemplate, m) for m in _materialFamilyFO)
    
    for i,fileName in enumerate(files):
        fileName = fileName.name
        textureDataEntry = textureData.get(fileName)
        if textureDataEntry:
            fractionalTileNumber = len(textureDataEntry) == 12
            # Calculate parameters out of input values.
            if fractionalTileNumber:
                # <windowCentralIndex> starts from zero
                textureWidthPx,\
                textureHeightPx,\
                windowWidthM,\
                windowCentralLpx,\
                windowCentralRpx,\
                windowCentralIndex,\
                windowCental2Lpx,\
                numberOfTilesU,\
                levelCentralTpx,\
                levelCentralBpx,\
                textureVoffsetPx,\
                numberOfTilesV\
                    = textureDataEntry
                
                tileWidthPx = windowCental2Lpx-windowCentralLpx
            else:
                textureWidthPx,\
                textureHeightPx,\
                windowWidthM,\
                windowCentralLpx,\
                windowCentralRpx,\
                numberOfTilesU,\
                numberOfTilesV\
                    = textureDataEntry
                
                tileWidthPx = textureWidthPx/numberOfTilesU

            windowWidthPx = windowCentralRpx-windowCentralLpx
            factor = windowWidthM/windowWidthPx
            
            textureWidthM = factor*textureWidthPx
            tileSizeUdefaultM = factor*tileWidthPx
            textureUoffsetM =\
                factor*(windowCentralLpx - windowCentralIndex*tileWidthPx - (tileWidthPx - windowWidthPx)/2.)\
                if fractionalTileNumber else\
                0.
            
            textureLevelHeightM =\
                factor*(levelCentralBpx-levelCentralTpx)\
                if fractionalTileNumber else\
                factor*textureHeightPx/numberOfTilesV
            textureHeightM = factor*textureHeightPx
            textureVoffsetM =\
                factor*(textureHeightPx - textureVoffsetPx)\
                if fractionalTileNumber else\
                0.
            
            for m, materialTemplate, customNode in\
                zip(_materialFamilyFO, materialTemplates, _customNodeFO):
                materialTemplate = bpy.data.materials.get(materialTemplate)
                if not materialTemplate:
                    print("Template \"%s\" for materials not found!" % materialTemplate)
                    continue
                materialName = "%s%s.%s" % (materialBaseName, m, (i+1))
                if not materialName in bpy.data.materials:
                    nodes = createMaterialFromTemplate(materialTemplate, materialName)
                    # the overlay texture
                    setImage(fileName, directory, nodes, "Overlay")
                    # The wall material (i.e. background) texture,
                    # set it just in case
                    setImage(wallTexturePath, None, nodes, "Wall Material")
                    nodes["Mapping"].scale[0] = 1./wallTextureWidthM
                    nodes["Mapping"].scale[1] = 1./wallTextureHeightM
                    # the mask for the emission
                    setImage(fileName, directory, nodes, "Emission Mask", "emissive")
                    # setting nodes
                    n = nodes[customNode]
                    setCustomNodeValue(n, "Texture Width", textureWidthM)
                    setCustomNodeValue(n, "Number of Tiles U", numberOfTilesU)
                    setCustomNodeValue(n, "Tile Size U Default", tileSizeUdefaultM)
                    setCustomNodeValue(n, "Texture U-Offset", textureUoffsetM)
                    setCustomNodeValue(n, "Number of Tiles V", numberOfTilesV)
                    setCustomNodeValue(n, "Texture Level Height", textureLevelHeightM)
                    setCustomNodeValue(n, "Texture Height", textureHeightM)
                    setCustomNodeValue(n, "Texture V-Offset", textureVoffsetM)
            # additionally create material for walls
            materialTemplate = bpy.data.materials.get("tiles_color_template")
            if not materialTemplate:
                print("Template \"%s\" for materials not found!" % materialTemplate)
                continue
            materialName = "%s_color" % wallMaterial
            if not materialName in bpy.data.materials:
                nodes = createMaterialFromTemplate(materialTemplate, materialName)
                setImage(wallTexturePath, None, nodes, "Image Texture")
                nodes["Mapping"].scale[0] = 1./wallTextureWidthM
                nodes["Mapping"].scale[1] = 1./wallTextureHeightM
        else:
            print(
                ("Information about the image texture \"%s\" isn't available " +
                "in the list of textures \"%s\"") % (fileName, listOfTextures)
            )


def createMaterialsForSeamlessTextures(files, directory, materialBaseName, listOfTextures, baseMaterialTemplate):
    textureData = readTextures(listOfTextures)
    
    materialTemplates = tuple("%s%s_template" % (baseMaterialTemplate, m) for m in _materialFamilyMS)
    
    for i,fileName in enumerate(files):
        fileName = fileName.name
        textureDataEntry = textureData.get(fileName)
        if textureDataEntry:
            for m, materialTemplate in zip(_materialFamilyMS, materialTemplates):
                materialTemplate = bpy.data.materials.get(materialTemplate)
                if not materialTemplate:
                    print("Template \"%s\" for materials not found!" % materialTemplate)
                    continue
                materialName = "%s%s.%s" % (materialBaseName, m, (i+1))
                if not materialName in bpy.data.materials:
                    nodes = createMaterialFromTemplate(materialTemplate, materialName)
                    
                    setImage(fileName, directory, nodes, "Image Texture")
                    nodes["Mapping"].scale[0] = 1./textureDataEntry[0]
                    nodes["Mapping"].scale[1] = 1./textureDataEntry[1]
        else:
            print(
                ("Information about the image texture \"%s\" isn't available " +
                "in the list of textures \"%s\"") % (fileName, listOfTextures)
            )


def readTextures(listOfTextures):
    from ..osm import parseNumber
    textureData = {}
    for line in bpy.data.texts[listOfTextures].lines:
        entry = line.body.split(',')
        if len(entry) < 3:
            continue
        textureData[entry[0]] = tuple(parseNumber(entry[i], entry[i]) for i in range(1, len(entry)))
    return textureData


def setImage(fileName, directory, nodes, nodeName, imageSuffix=None):
    node = nodes.get(nodeName)
    if node:
        if imageSuffix:
            fileName = "%s_%s.png" % (fileName[:-4], imageSuffix)
        image = bpy.data.images.get(fileName if directory else os.path.basename(fileName))
        if not image:
            # absolute path!
            imagePath = os.path.join(directory, fileName) if directory else fileName
            try:
                image = bpy.data.images.load(imagePath)
            except Exception:
                print("Unable to load the image %s" % imagePath)
        node.image = image
    else:
        image = None
    return image


def setCustomNodeValue(node, inputName, value):
    node.inputs[inputName].default_value = value


"""
class OperatorDownloadTextures(bpy.types.Operator):
    bl_idname = "blosm.download_textures"
    bl_label = "download textures"
    bl_description = "Open multiple webbrowser tabs to download textures from textures.com"
    bl_options = {'INTERNAL'}
    
    url = "https://www.textures.com/search?q="
    
    pauseDuration = 1.
    
    def invoke(self, context, event):
        for line in bpy.data.texts[context.scene.blender_osm.listOfTextures].lines:
            name = line.body.split(',')[0]
            index1 = name.find('_')
            if index1 > 0:
                index1 += 1
                index2 = name.find('_', index1)
                if index2 > 0:
                    webbrowser.open_new_tab(
                        "%s%s" % (self.url, name[index1:index2])
                    )
                    time.sleep(self.pauseDuration)
        return {'FINISHED'}


class OperatorCreateMaterials(bpy.types.Operator):
    bl_idname = "blosm.create_materials"
    bl_label = "Create materials..."
    bl_description = "Create Blender materials with the selected Python script"
    bl_options = {"REGISTER", "UNDO"}
    
    directory = bpy.props.StringProperty(subtype='DIR_PATH')
    
    files = bpy.props.CollectionProperty(type=bpy.types.OperatorFileListElement)
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}
    
    def execute(self, context):
        addon = context.scene.blender_osm
        module = imp.new_module("module")
        exec(
            bpy.data.texts[addon.materialScript].as_string(),
            module.__dict__
        )
        module.main(self.files, self.directory)
        return {'FINISHED'}
"""


class OperatorCreateMaterials(bpy.types.Operator):
    bl_idname = "blosm.create_materials"
    bl_label = "Create materials..."
    bl_description = "Create Blender materials for the chosen material type"
    bl_options = {"REGISTER", "UNDO"}
    
    directory = bpy.props.StringProperty(subtype='DIR_PATH')
    
    files = bpy.props.CollectionProperty(type=bpy.types.OperatorFileListElement)
    
    def invoke(self, context, event):
        if not bpy.data.is_saved:
            self.report({'ERROR'}, "Save the Blender file before creating materials!")
            return {'CANCELLED'}
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
    def execute(self, context):
        if not bpy.data.is_saved:
            self.report({'ERROR'}, "Save the Blender file before creating materials!")
            return {'CANCELLED'}
        addon = context.scene.blender_osm
        materialType = addon.materialType
        # <ms> means 'seamless materials'
        if materialType == "ms":
            createMaterialsForSeamlessTextures(
                self.files,
                self.directory,
                addon.blenderMaterials,
                addon.listOfTextures,
                "tiles"
            )
        # <fo> means 'facades with window overlays'
        elif materialType == "fo":
            wallTexturePath = os.path.realpath(bpy.path.abspath(addon.wallTexture))
            if not os.path.isfile(wallTexturePath):
                self.report({'ERROR'}, "Set a valid wall texture listed in in the Blender text data-block \"textures_cladding\"")
                return {'CANCELLED'}
            
            wallTextureFileName = os.path.basename(wallTexturePath)
            textBlock = bpy.data.texts.get("textures_cladding")
            if not textBlock:
                self.report({'ERROR'}, "The Blender text data-block \"textures_cladding\" is not found!")
                return {'CANCELLED'}
            
            for line in textBlock.lines:
                fileName, width, height, wallMaterial =  map(lambda s: s.strip(), line.body.split(','))
                if fileName == wallTextureFileName:
                    width = float(width)
                    height = float(height)
                    break
            else:
                self.report({'ERROR'}, "Unable to find a valid entry for the texture %s in the Blender text data-block \"textures_cladding\"!" % wallTextureFileName)
                return {'CANCELLED'}
            createMaterialsForFacadesOverlay(
                self.files,
                self.directory,
                addon.blenderMaterials,
                addon.listOfTextures,
                "facade_overlay",
                wallMaterial,
                wallTexturePath,
                width,
                height
            )
        # <fs> means 'facades seamless'
        elif materialType == "fs":
            if addon.blenderMaterials == "glass":
                # no emission
                createFacadeMaterialsForSeamlessTextures(
                    self.files,
                    self.directory,
                    addon.listOfTextures,
                    "glass",
                    "glass_ground_level",
                    "glass_template",
                    "glass_ground_level_template",
                    # additional image textures as kwargs
                    diffuse = "Diffuse Mask"
                )
                # emission
                createFacadeMaterialsForSeamlessTextures(
                    self.files,
                    self.directory,
                    addon.listOfTextures,
                    "glass_emission",
                    "glass_ground_level_emission",
                    "glass_emission_template",
                    "glass_ground_level_emission_template",
                    # additional image textures as kwargs
                    emission = "Emission Texture",
                    diffuse = "Diffuse Mask"
                )
            else:
                # no emission
                createFacadeMaterialsForSeamlessTextures(
                    self.files,
                    self.directory,
                    addon.listOfTextures,
                    addon.blenderMaterials,
                    "%s_ground_level" % addon.blenderMaterials,
                    "facade_seamless_template",
                    "facade_seamless_ground_level_template"
                )
                # emission
                createFacadeMaterialsForSeamlessTextures(
                    self.files,
                    self.directory,
                    addon.listOfTextures,
                    "%s_emission" % addon.blenderMaterials,
                    "%s_ground_level_emission" % addon.blenderMaterials,
                    "facade_seamless_emission_template",
                    "facade_seamless_ground_level_emission_template",
                    # additional image textures as kwargs
                    emissive = "Emission Mask"
                )
        return {'FINISHED'}


class OperatorDeleteMaterials(bpy.types.Operator):
    bl_idname = "blosm.delete_materials"
    bl_label = "Delete materials..."
    bl_description = "Delete a family of Blender materials for the chosen OSM material"
    bl_options = {"REGISTER", "UNDO"}
        
    def execute(self, context):
        addon = context.scene.blender_osm
        materialFamily =\
            _materialFamilyFO\
            if addon.materialType=="fo" or addon.materialType=="fs"\
            else _materialFamilyMS
        for i in range(100):
            for m in materialFamily:
                materialName = "%s%s.%s" % (addon.blenderMaterials, m, (i+1))
                material = bpy.data.materials.get(materialName)
                if material:
                    bpy.data.materials.remove(material, do_unlink=True)
        
        if addon.materialType == "ms":
            # also delete wall materials used by facade overlays
            wallMaterial = bpy.data.materials.get("%s_color" % addon.blenderMaterials)
            if wallMaterial:
                bpy.data.materials.remove(wallMaterial, do_unlink=True)
        return {'FINISHED'}


class PanelMaterialCreate(bpy.types.Panel):
    bl_label = "Material Utilities"
    bl_space_type = "NODE_EDITOR"
    bl_region_type = "UI" if _isBlender280 else "TOOLS"
    bl_context = "objectmode"
    bl_category = "osm"
    #bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        addon = context.scene.blender_osm
        return app.has(Keys.mode3dRealistic) and addon.dataType == "osm"\
            and addon.mode == "3Drealistic"
    
    def draw(self, context):
        addon = context.scene.blender_osm
        layout = self.layout
        
        box = layout.box()
        box.label(text="Material type:")
        box.prop(addon, "materialType", text='')
        
        layout.prop(addon, "blenderMaterials")
        
        if addon.materialType == "fo":
            # wall texture (i.e. background material)
            layout.prop(addon, "wallTexture")
        
        layout.prop_search(addon, "listOfTextures", bpy.data, "texts")
        
        # <ms> stands for 'material seamless'
        #if addon.materialType == "ms":
        #    layout.operator("blosm.download_textures")

        #layout.prop_search(addon, "materialScript", bpy.data, "texts")
        row = layout.row(align=True)
        row.operator("blosm.create_materials")
        row.operator("blosm.delete_materials")
        

_classes = (
    OperatorCreateMaterials,
    OperatorDeleteMaterials,
    PanelMaterialCreate
)

def register():
    for c in _classes:
        bpy.utils.register_class(c)

def unregister():
    for c in _classes:
        bpy.utils.unregister_class(c)