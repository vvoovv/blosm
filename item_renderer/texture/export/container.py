import os
import bpy
from .item_renderer import ItemRenderer, _textureDir
from ..container import Container as ContainerBase


class Container(ContainerBase, ItemRenderer):
    """
    The base class for the item renderers Facade, Div, Layer, Bottom
    """
    
    def __init__(self, exportMaterials):
        super().__init__(exportMaterials)
        # The following variable is used to cache the cladding color as as string:
        # either a base colors (e.g. red, green) or a hex string
        self.claddingColor = None
    
    def init(self, itemRenderers, globalRenderer):
        super().init(itemRenderers, globalRenderer)
        self.exporter = globalRenderer.materialExportManager.facadeExporter

    def getFacadeMaterialId(self, item, facadeTextureInfo, claddingTextureInfo):
        color = self.getCladdingColor(item)
        return "%s_%s_%s" % (claddingTextureInfo["material"], color, facadeTextureInfo["name"])\
            if claddingTextureInfo and color\
            else facadeTextureInfo["name"]
    
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