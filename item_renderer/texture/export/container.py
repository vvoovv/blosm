import os
import bpy
from manager import Manager
from ..container import Container as ContainerBase
from util.blender_extra.material import createMaterialFromTemplate, setImage


_textureDir = "texture"
_facadeMaterialTemplateFilename = "building_material_templates.blend"
_facadeMaterialTemplateName = "export_template"


class Container(ContainerBase):
    """
    The base class for the item renderers Facade, Div, Layer, Basement
    """
    
    def __init__(self, exportMaterials):
        # The following variable is used to cache the cladding color as as string:
        # either a base colors (e.g. red, green) or a hex string
        self.claddingColor = None
        super().__init__(exportMaterials)
    
    def renderCladding(self, building, parentItem, face, uvs):
        self.setCladdingUvs(face, uvs)
        super().renderCladding(building, parentItem, face, uvs)
    
    def setVertexColor(self, parentItem, face):
        # do nothing here
        pass

    def setCladdingUvs(self, face, uvs):
        self.r.setUvs(
            face,
            uvs,
            self.r.layer.uvLayerNameFacade
        )

    def getFacadeMaterialId(self, item, facadeTextureInfo, claddingTextureInfo):
        color = Manager.normalizeColor(item.getStyleBlockAttrDeep("claddingColor"))
        # remember the color for a future use
        self.claddingColor = color
        return "%s_%s_%s" % (claddingTextureInfo["material"], color, facadeTextureInfo["name"])\
            if claddingTextureInfo and color\
            else facadeTextureInfo["name"]

    def createFacadeMaterial(self, materialName, facadeTextureInfo, claddingTextureInfo):
        if not materialName in bpy.data.materials:
            # check if have texture in the data directory
            textureFilepath = os.path.join(self.r.app.dataDir, _textureDir, materialName)
            if not os.path.isfile(textureFilepath):
                self.r.materialExportManager.facadeExporter.makeTexture(
                    materialName, # the file name of the texture
                    os.path.join(self.r.app.dataDir, _textureDir),
                    self.claddingColor,
                    facadeTextureInfo,
                    claddingTextureInfo
                )
            
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
        return True