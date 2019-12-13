import os
import bpy
from manager import Manager
from util.blender import loadSceneFromFile
from util.blender_extra.material import createMaterialFromTemplate, setImage


_exportTemplateFilename = "building_material_templates.blend"


class Exporter:
    
    def __init__(self, bldgMaterialsDirectory, sceneName):
        self.bldgMaterialsDirectory = bldgMaterialsDirectory
        exportTemplateFilename = os.path.join(bldgMaterialsDirectory, _exportTemplateFilename)
        self.setTemplateScene(exportTemplateFilename, sceneName)
    
    def cleanup(self):
        #bpy.data.scenes.remove(self.scene)
        self.scene = None
    
    def setTemplateScene(self, exportTemplateFilename, sceneName):
        scene = None
        scenes = bpy.data.scenes
        scene = scenes.get(sceneName)
        if scene:
            # perform a quick sanity check here
            if not scene.use_nodes:
                scene = None
        if not scene:
            scene = loadSceneFromFile(exportTemplateFilename, sceneName)
        self.scene = scene
    
    def verifyPath(self, textureDir):
        if not os.path.exists(textureDir):
            os.makedirs(textureDir)
    
    def setColor(self, textColor, nodes, nodeName):
        color = Manager.getColor(textColor)
        nodes[nodeName].outputs[0].default_value = (color[0], color[1], color[2], 1.)


class FacadeExporter(Exporter):
    
    def makeTexture(self, textureFilename, textureDir, textColor, facadeTextureInfo, claddingTextureInfo):
        self.verifyPath(textureDir)
        nodes = self.scene.node_tree.nodes
        fileOutputNode = nodes["File Output"]
        fileOutputNode.base_path = textureDir
        fileOutputNode.file_slots[0].path = os.path.splitext(textureFilename)[0]
        # facade texture
        setImage(
            facadeTextureInfo["name"],
            os.path.join(self.bldgMaterialsDirectory, facadeTextureInfo["path"]),
            nodes,
            "facade_texture"
        )
        # cladding texture
        setImage(
            claddingTextureInfo["name"],
            os.path.join(self.bldgMaterialsDirectory, claddingTextureInfo["path"]),
            nodes,
            "cladding_texture"
        )
        # scale for the cladding texture
        scaleInputs = nodes["Scale"].inputs
        scaleFactor = claddingTextureInfo["textureWidthM"]/\
        claddingTextureInfo["textureWidthPx"]*\
        (facadeTextureInfo["windowRpx"]-facadeTextureInfo["windowLpx"])/\
        facadeTextureInfo["windowWidthM"]
        scaleInputs[1].default_value = scaleFactor
        scaleInputs[2].default_value = scaleFactor
        # cladding color
        self.setColor(textColor, nodes, "cladding_color")
        # render the resulting texture
        bpy.ops.render.render(scene=self.scene.name)
        


class MaterialExportManager:
    
    def __init__(self, bldgMaterialsDirectory):
        self.init(bldgMaterialsDirectory)
    
    def init(self, bldgMaterialsDirectory):
        self.facadeExporter = FacadeExporter(bldgMaterialsDirectory, "compositing_facade")
        self.claddingExporter = Exporter(bldgMaterialsDirectory, "compositing_cladding")
        self.doorExporter = Exporter(bldgMaterialsDirectory, "compositing_door")
    
    def cleanup(self):
        self.facadeExporter.cleanup()
        self.facadeExporter = None
        self.claddingExporter.cleanup()
        self.claddingExporter = None
        self.doorExporter.cleanup()
        self.doorExporter = None