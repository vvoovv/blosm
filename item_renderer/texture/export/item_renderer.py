import os
import bpy
from manager import Manager
from util.blender import loadSceneFromFile
from util.blender_extra.material import createMaterialFromTemplate, setImage


_textureDir = "texture"
_materialTemplateFilename = "building_material_templates.blend"
_materialTemplateName = "export"


class ItemRendererMixin:
    """
    A mixin class
    """
    
    def getTemplateScene(self, sceneName):
        scene = bpy.data.scenes.get(sceneName)
        if scene:
            # perform a quick sanity check here
            if not scene.use_nodes:
                scene = None
        if not scene:
            scene = loadSceneFromFile(self.r.textureExporter.exportTemplateFilename, sceneName)
        return scene
    
    def getCladdingMaterialId(self, item, claddingTextureInfo):
        color = self.getCladdingColor(item)
        return "%s_%s%s" % (color, claddingTextureInfo["material"], os.path.splitext(claddingTextureInfo["name"])[1])\
            if claddingTextureInfo and color\
            else claddingTextureInfo["name"]
    
    def createCladdingMaterial(self, materialName, claddingTextureInfo):
        if not materialName in bpy.data.materials:
            # check if have texture in the data directory
            textureFilepath = self.getTextureFilepath(materialName)
            if not os.path.isfile(textureFilepath):
                self.makeCladdingTexture(
                    materialName,
                    os.path.join(self.r.app.dataDir, _textureDir),
                    claddingTextureInfo
                )
            
            self.createMaterialFromTemplate(materialName, textureFilepath)
        return True
    
    def makeCladdingTexture(self, textureFilename, textureDir, claddingTextureInfo):
        textureExporter = self.r.textureExporter
        scene = self.getTemplateScene("compositing_cladding_color")
        nodes = textureExporter.makeCommonPreparations(
            scene,
            textureFilename,
            textureDir
        )
        # cladding texture
        setImage(
            claddingTextureInfo["name"],
            os.path.join(textureExporter.bldgMaterialsDirectory, claddingTextureInfo["path"]),
            nodes,
            "cladding_texture"
        )
        # cladding color
        self.setColor(self.claddingColor, nodes, "cladding_color")
        # render the resulting texture
        textureExporter.renderTexture(scene, textureFilename, textureDir)
    
    def getCladdingColor(self, item):
        color = Manager.normalizeColor(item.getStyleBlockAttrDeep("claddingColor"))
        # remember the color for a future use in the next funtion call
        self.claddingColor = color
        return color
    
    def setColor(self, textColor, nodes, nodeName):
        color = Manager.getColor(textColor)
        nodes[nodeName].outputs[0].default_value = (color[0], color[1], color[2], 1.)
    
    def getTextureFilepath(self, materialName):
        return os.path.join(self.r.app.dataDir, _textureDir, materialName)
    
    def createMaterialFromTemplate(self, materialName, textureFilepath):
        materialTemplate = self.getMaterialTemplate(
            _materialTemplateFilename,
            _materialTemplateName
        )
        nodes = createMaterialFromTemplate(materialTemplate, materialName)
        # the overlay texture
        setImage(
            textureFilepath,
            None,
            nodes,
            "Image Texture"
        )