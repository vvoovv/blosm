import webbrowser, time, imp, os
import bpy


materialFamily = (
    "",
    "_with_color",
    "_scaled",
    "_scaled_with_color"
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


def createMaterialsForSeamlessTextures(files, directory, materialBaseName, listOfTextures, baseMaterialTemplate):
    textureData = readTextures(listOfTextures)
    
    materialTemplates = tuple("%s%s_template" % (baseMaterialTemplate, m) for m in materialFamily)
    
    for i,fileName in enumerate(files):
        fileName = fileName.name
        textureDataEntry = textureData.get(fileName)
        if textureDataEntry:
            for m, materialTemplate in zip(materialFamily, materialTemplates):
                materialName = "%s%s.%s" % (materialBaseName, m, (i+1))
                materialTemplate = bpy.data.materials[materialTemplate]
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


class OperatorCreateMaterialsSeamless(bpy.types.Operator):
    bl_idname = "blosm.create_materials_seamless"
    bl_label = "Create materials..."
    bl_description = "Create Blender materials for the selected seamless textures"
    bl_options = {"REGISTER", "UNDO"}
    
    directory = bpy.props.StringProperty(subtype='DIR_PATH')
    
    files = bpy.props.CollectionProperty(type=bpy.types.OperatorFileListElement)
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}
    
    def execute(self, context):
        addon = context.scene.blender_osm
        createMaterialsForSeamlessTextures(
            self.files,
            self.directory,
            addon.osmMaterial,
            addon.listOfTextures,
            "tiles"
        )
        return {'FINISHED'}


class OperatorDeleteMaterials(bpy.types.Operator):
    bl_idname = "blosm.delete_materials"
    bl_label = "Delete materials..."
    bl_description = "Delete a family of Blender materials for the chosen OSM material"
    bl_options = {"REGISTER", "UNDO"}
        
    def execute(self, context):
        addon = context.scene.blender_osm
        for i in range(100):
            for m in materialFamily:
                materialName = "%s%s.%s" % (addon.osmMaterial, m, (i+1))
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
        
        layout.prop_search(addon, "listOfTextures", bpy.data, "texts")
        layout.operator("blosm.download_textures")

        #layout.prop_search(addon, "materialScript", bpy.data, "texts")
        layout.prop(addon, "osmMaterial")
        layout.operator("blosm.create_materials_seamless")
        layout.operator("blosm.delete_materials")


def register():
    bpy.utils.register_class(OperatorDownloadTextures)
    bpy.utils.register_class(OperatorCreateMaterials)
    bpy.utils.register_class(OperatorCreateMaterialsSeamless)
    bpy.utils.register_class(OperatorDeleteMaterials)
    bpy.utils.register_class(PanelMaterialCreate)

def unregister():
    bpy.utils.unregister_class(OperatorDownloadTextures)
    bpy.utils.unregister_class(OperatorCreateMaterials)
    bpy.utils.unregister_class(OperatorCreateMaterialsSeamless)
    bpy.utils.unregister_class(OperatorDeleteMaterials)
    bpy.utils.unregister_class(PanelMaterialCreate)