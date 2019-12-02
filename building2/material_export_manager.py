import os
import bpy
from util.blender import loadSceneFromFile


_exportTemplateFilename = "building_material_templates.blend"


class Exporter:
    
    def __init__(self, exportTemplateFilename, sceneName):
        self.setTemplateScene(exportTemplateFilename, sceneName)
    
    def cleanup(self):
        bpy.data.scenes.remove(self.scene)
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


class MaterialExportManager:
    
    def __init__(self, bldgMaterialsDirectory):
        self.init(bldgMaterialsDirectory)
    
    def init(self, bldgMaterialsDirectory):
        exportTemplateFilename = os.path.join(bldgMaterialsDirectory, _exportTemplateFilename)
        self.facadeExporter = Exporter(exportTemplateFilename, "compositing_facade")
        self.claddingExporter = Exporter(exportTemplateFilename, "compositing_cladding")
        self.doorExporter = Exporter(exportTemplateFilename, "compositing_door")
    
    def cleanup(self):
        self.facadeExporter.cleanup()
        self.facadeExporter = None
        self.claddingExporter.cleanup()
        self.claddingExporter = None
        self.doorExporter.cleanup()
        self.doorExporter = None