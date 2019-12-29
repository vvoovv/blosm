import os
import bpy
from manager import Manager
from ..container import Container as ContainerBase
from ...geometry.rectangle import Rectangle
from util.blender_extra.material import createMaterialFromTemplate, setImage


_textureDir = "texture"
_facadeMaterialTemplateFilename = "building_material_templates.blend"
_facadeMaterialTemplateName = "export_template"


class Container(ContainerBase):
    """
    The base class for the item renderers Facade, Div, Layer, Basement
    """
    
    def __init__(self, exportMaterials):
        super().__init__(exportMaterials)
        # The following variable is used to cache the cladding color as as string:
        # either a base colors (e.g. red, green) or a hex string
        self.claddingColor = None
    
    def init(self, itemRenderers, globalRenderer):
        super().init(itemRenderers, globalRenderer)
        self.exporter = globalRenderer.materialExportManager.facadeExporter
    
    def setVertexColor(self, parentItem, face):
        # do nothing here
        pass
    
    def getCladdingColor(self, item):
        color = Manager.normalizeColor(item.getStyleBlockAttrDeep("claddingColor"))
        # remember the color for a future use in the next funtion call
        self.claddingColor = color
        return color
    
    def getTextureFilepath(self, materialName):
        return os.path.join(self.r.app.dataDir, _textureDir, materialName)

    def getFacadeMaterialId(self, item, facadeTextureInfo, claddingTextureInfo):
        color = self.getCladdingColor(item)
        return "%s_%s_%s" % (claddingTextureInfo["material"], color, facadeTextureInfo["name"])\
            if claddingTextureInfo and color\
            else facadeTextureInfo["name"]
    
    def getCladdingMaterialId(self, item, claddingTextureInfo):
        color = self.getCladdingColor(item)
        return "%s_%s%s" % (color, claddingTextureInfo["material"], os.path.splitext(claddingTextureInfo["name"])[1])\
            if claddingTextureInfo and color\
            else claddingTextureInfo["name"]
    
    def createMaterialFromTemplate(self, materialName, textureFilepath):
        materialTemplate = self.getMaterialTemplate(
            _facadeMaterialTemplateFilename,
            _facadeMaterialTemplateName
        )
        nodes = createMaterialFromTemplate(materialTemplate, materialName)
        # the overlay texture
        setImage(
            textureFilepath,
            None,
            nodes,
            "Image Texture"
        )
    
    def createFacadeMaterial(self, materialName, facadeTextureInfo, claddingTextureInfo, uvs):
        if not materialName in bpy.data.materials:
            # check if have texture in the data directory
            textureFilepath = self.getTextureFilepath(materialName)
            if not os.path.isfile(textureFilepath):
                self.exporter.makeTexture(
                    materialName, # the file name of the texture
                    os.path.join(self.r.app.dataDir, _textureDir),
                    self.claddingColor,
                    facadeTextureInfo,
                    claddingTextureInfo,
                    uvs
                )
            
            self.createMaterialFromTemplate(materialName, textureFilepath)
        return True
    
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