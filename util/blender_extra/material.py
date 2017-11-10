import webbrowser, time, os
import bpy


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
_customNodeFO = (
    "FacadeOverlay",
    "FacadeOverlay",
    "FacadeOverlayGroundLevel",
    "FacadeOverlayGroundLevel"
)

# <MS> means 'simple seamless material'
_materialFamilyMS = (
    "",
    "_color",
    "_scaled",
    "_scaled_color"
)


def createFacadeMaterialsForSeamlessTextures(files, directory, listOfTextures, materialTemplate1, materialTemplate2):
    def createMaterials(materialBaseName, materialTemplate):
        materialName = "%s.%s" % (materialBaseName, (i+1))
        if not materialName in bpy.data.materials:
            m = materialTemplate.copy()
            m.name = materialName
            m.use_fake_user = True
            
            nodes = m.node_tree.nodes
            setImage(fileName, directory, nodes["Image Texture"])
            setCustomNodeValue(nodes["FacadePart"], "Number of Tiles U", textureDataEntry[0])
            setCustomNodeValue(nodes["FacadePart"], "Number of Tiles V", textureDataEntry[1])
            setCustomNodeValue(nodes["FacadePart"], "Tile Size U Default", textureDataEntry[2])
    
    textureData = readTextures(listOfTextures)
    
    # strip off the suffix '_template'
    materialBaseName1 = materialTemplate1[:-9]
    materialBaseName2 = materialTemplate2[:-9]
    
    materialTemplate1 = bpy.data.materials[materialTemplate1]
    materialTemplate2 = bpy.data.materials[materialTemplate2]
    
    for i,fileName in enumerate(files):
        fileName = fileName.name
        textureDataEntry = textureData.get(fileName)
        if textureDataEntry:
            createMaterials(materialBaseName1, materialTemplate1)
            createMaterials(materialBaseName2, materialTemplate2)
        else:
            print(
                ("Information about the image texture \"%s\" isn't available " +
                "in the list of textures \"%s\"") % (fileName, listOfTextures)
            )


def createMaterialsForFacadesOverlay(files, directory, materialBaseName, listOfTextures, baseMaterialTemplate):
    def setNode(n):
        setCustomNodeValue(n, "Texture Width", textureWidthM)
        setCustomNodeValue(n, "Number of Tiles U", numberOfTilesU)
        setCustomNodeValue(n, "Tile Size U Default", tileSizeUdefaultM)
        setCustomNodeValue(n, "Texture U-Offset", textureUoffsetM)
        setCustomNodeValue(n, "Number of Tiles V", numberOfTilesV)
        setCustomNodeValue(n, "Texture Level Height", textureLevelHeightM)
        setCustomNodeValue(n, "Texture Height", textureHeightM)
        setCustomNodeValue(n, "Texture V-Offset", textureVoffsetM)
    
    textureData = readTextures(listOfTextures)
    
    materialTemplates = tuple("%s%s_template" % (baseMaterialTemplate, m) for m in _materialFamilyFO)
    
    for i,fileName in enumerate(files):
        fileName = fileName.name
        textureDataEntry = textureData.get(fileName)
        if textureDataEntry:
            # calculate parameters out of input values
            textureWidthPx,\
            textureHeightPx,\
            windowWidthM,\
            windowCentalLpx,\
            windowCentalRpx,\
            windowCentralIndex,\
            windowCental2Lpx,\
            numberOfTilesU,\
            levelCentralTpx,\
            levelCentralBpx,\
            textureVoffsetPx,\
            numberOfTilesV\
                = textureDataEntry
            
            windowWidthPx = windowCentalRpx-windowCentalLpx
            tileWidthPx = windowCental2Lpx-windowCentalLpx
            factor = windowWidthM/windowWidthPx
            
            textureWidthM = factor*textureWidthPx
            tileSizeUdefaultM = factor*tileWidthPx
            textureUoffsetM = factor*(windowCentalLpx - (windowCentralIndex-1)*tileWidthPx - (tileWidthPx - windowWidthPx)/2.)
            
            textureLevelHeightM = factor*(levelCentralBpx-levelCentralTpx)
            textureHeightM = factor*textureHeightPx
            textureVoffsetM = factor*(textureHeightPx - textureVoffsetPx)
            
            for index, m, materialTemplate, customNode in\
                zip(range(len(materialTemplates)), _materialFamilyFO, materialTemplates, _customNodeFO):
                materialTemplate = bpy.data.materials[materialTemplate]
                if not materialTemplate:
                    print("Template \"%s\" for materials not found!" % materialTemplate)
                    continue
                materialName = "%s%s.%s" % (materialBaseName, m, (i+1))
                if not materialName in bpy.data.materials:
                    m = materialTemplate.copy()
                    m.name = materialName
                    m.use_fake_user = True
                    
                    nodes = m.node_tree.nodes
                    # the base image
                    setImage(fileName, directory, nodes["Overlay"])
                    # the masks for the overlay and for the emission
                    setImage("%s_masks.png" % fileName[:-4], directory, nodes["Masks"])
                    setNode(nodes[customNode])
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
                materialTemplate = bpy.data.materials[materialTemplate]
                if not materialTemplate:
                    print("Template \"%s\" for materials not found!" % materialTemplate)
                    continue
                materialName = "%s%s.%s" % (materialBaseName, m, (i+1))
                if not materialName in bpy.data.materials:
                    m = materialTemplate.copy()
                    m.name = materialName
                    m.use_fake_user = True
                    
                    nodes = m.node_tree.nodes
                    setImage(fileName, directory, nodes["Image Texture"])
                    nodes["Mapping"].scale[0] = 1./textureDataEntry[0]
                    nodes["Mapping"].scale[1] = 1./textureDataEntry[1]
        else:
            print(
                ("Information about the image texture \"%s\" isn't available " +
                "in the list of textures \"%s\"") % (fileName, listOfTextures)
            )


def readTextures(listOfTextures):
    textureData = {}
    for line in bpy.data.texts[listOfTextures].lines:
        entry = line.body.split(',')
        textureData[entry[0]] = tuple(float(entry[i]) for i in range(1, len(entry)))
    return textureData


def setImage(fileName, directory, node):
    image = bpy.data.images.get(
        fileName,
        bpy.data.images.load(os.path.join(directory, fileName))
    )
    node.image = image


def setCustomNodeValue(node, inputName, value):
    node.inputs[inputName].default_value = value


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


"""
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
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}
    
    def execute(self, context):
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
            createMaterialsForFacadesOverlay(
                self.files,
                self.directory,
                addon.blenderMaterials,
                addon.listOfTextures,
                "facade_overlay"
            )
        return {'FINISHED'}


class OperatorDeleteMaterials(bpy.types.Operator):
    bl_idname = "blosm.delete_materials"
    bl_label = "Delete materials..."
    bl_description = "Delete a family of Blender materials for the chosen OSM material"
    bl_options = {"REGISTER", "UNDO"}
        
    def execute(self, context):
        addon = context.scene.blender_osm
        materialFamily = _materialFamilyFO if addon.materialType=="fo" else _materialFamilyMS
        for i in range(100):
            for m in materialFamily:
                materialName = "%s%s.%s" % (addon.blenderMaterials, m, (i+1))
                material = bpy.data.materials.get(materialName)
                if material:
                    bpy.data.materials.remove(material, True)
        return {'FINISHED'}


class PanelMaterialCreate(bpy.types.Panel):
    bl_label = "Material Utils"
    bl_space_type = "VIEW_3D"
    bl_region_type = "TOOLS"
    bl_context = "objectmode"
    bl_category = "osm"
    
    def draw(self, context):
        addon = context.scene.blender_osm
        layout = self.layout
        
        box = layout.box()
        box.label("Material type:")
        box.prop(addon, "materialType", text='')
        
        layout.prop(addon, "blenderMaterials")
        
        layout.prop_search(addon, "listOfTextures", bpy.data, "texts")
        
        # <ms> stands for 'material seamless'
        if addon.materialType == "ms":
            layout.operator("blosm.download_textures")

        #layout.prop_search(addon, "materialScript", bpy.data, "texts")
        row = layout.row(align=True)
        row.operator("blosm.create_materials")
        row.operator("blosm.delete_materials")


def register():
    bpy.utils.register_class(OperatorDownloadTextures)
    bpy.utils.register_class(OperatorCreateMaterials)
    bpy.utils.register_class(OperatorDeleteMaterials)
    bpy.utils.register_class(PanelMaterialCreate)

def unregister():
    bpy.utils.unregister_class(OperatorDownloadTextures)
    bpy.utils.unregister_class(OperatorCreateMaterials)
    bpy.utils.unregister_class(OperatorDeleteMaterials)
    bpy.utils.unregister_class(PanelMaterialCreate)