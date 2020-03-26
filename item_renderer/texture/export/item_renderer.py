import os
import bpy
from manager import Manager
from util.blender_extra.material import createMaterialFromTemplate, setImage

_textureDir = "texture"
_materialTemplateFilename = "building_material_templates.blend"
_materialTemplateName = "export"


class ItemRendererMixin:
    """
    A mixin class
    """
    
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
                self.r.materialExportManager.claddingExporter.makeTexture(
                    materialName, # the file name of the texture
                    os.path.join(self.r.app.dataDir, _textureDir),
                    self.claddingColor,
                    claddingTextureInfo
                )
            
            self.createMaterialFromTemplate(materialName, textureFilepath)
        return True
    
    def getCladdingColor(self, item):
        color = Manager.normalizeColor(item.getStyleBlockAttrDeep("claddingColor"))
        # remember the color for a future use in the next funtion call
        self.claddingColor = color
        return color
    
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
    
    def setVertexColor(self, parentItem, face):
        # do nothing here
        pass