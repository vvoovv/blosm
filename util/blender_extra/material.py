import webbrowser, time, imp, os
import bpy


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
        #module = imp.new_module("module")
        #exec(
        #    bpy.data.texts[addon.materialScript].as_string(),
        #    module.__dict__
        #)
        #module.main("testing")
        textures = "textures_apartments"
        material1 = "apartments_template"
        material2 = "apartments_with_ground_level_template"

        textureData = readTextures(textures)
        
        materialTemplate1 = bpy.data.materials[material1]
        materialTemplate2 = bpy.data.materials[material2]
        # strip off the suffix '_template'
        materialBaseName1 = material1[:-9]
        materialBaseName2 = material2[:-9]
        
        for i,fileName in enumerate(self.files):
            fileName = fileName.name
            textureDataEntry = textureData[fileName]
            
            materialName1 = "%s.%s" % (materialBaseName1, (i+1))
            materialName2 = "%s.%s" % (materialBaseName2, (i+1))
            
            if not materialName1 in bpy.data.materials:
                m = materialTemplate1.copy()
                m.name = materialName1
                m.use_fake_user = True
                
                nodes = m.node_tree.nodes
                setImage(fileName, self.directory, nodes["Image Texture"])
                setCustomNodeValue(nodes["FacadePart"], "Number of Tiles U", textureDataEntry[0])
                setCustomNodeValue(nodes["FacadePart"], "Number of Tiles V", textureDataEntry[1])
                setCustomNodeValue(nodes["FacadePart"], "Tile Size U Default", textureDataEntry[2])
                
            if not materialName2 in bpy.data.materials:
                m = materialTemplate2.copy()
                m.name = materialName2
                m.use_fake_user = True
                
                nodes = m.node_tree.nodes
                setImage(fileName, self.directory, nodes["Image Texture"])
                setCustomNodeValue(nodes["FacadePart"], "Number of Tiles U", textureDataEntry[0])
                setCustomNodeValue(nodes["FacadePart"], "Number of Tiles V", textureDataEntry[1])
                setCustomNodeValue(nodes["FacadePart"], "Tile Size U Default", textureDataEntry[2])
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

        layout.prop_search(addon, "materialScript", bpy.data, "texts")
        layout.operator("blosm.create_materials")


def register():
    bpy.utils.register_class(OperatorDownloadTextures)
    bpy.utils.register_class(OperatorCreateMaterials)
    bpy.utils.register_class(PanelMaterialCreate)

def unregister():
    bpy.utils.unregister_class(OperatorDownloadTextures)
    bpy.utils.unregister_class(OperatorCreateMaterials)
    bpy.utils.unregister_class(PanelMaterialCreate)