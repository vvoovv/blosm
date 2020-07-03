import os
import bpy
from util.blender_extra.material import createMaterialFromTemplate, setImage

from .container import Container
from ..level import CurtainWall as CurtainWallBase
from ...util import setTextureSize, setTextureSize2


class CurtainWall(CurtainWallBase, Container):
    
    def __init__(self):
        # a reference to the Container class used in the parent classes
        self.Container = Container
        Container.__init__(self, exportMaterials=False)
        CurtainWallBase.__init__(self)
    
    def createFacadeMaterial(self, materialName, facadeTextureInfo, claddingTextureInfo, uvs):
        if not materialName in bpy.data.materials:
            materialTemplate = self.getFacadeMaterialTemplate(
                facadeTextureInfo,
                None,
                self.materialTemplateFilename
            )
            nodes = createMaterialFromTemplate(materialTemplate, materialName)
            # the overlay texture
            image = setImage(
                facadeTextureInfo["name"],
                os.path.join(self.r.assetStore.baseDir, facadeTextureInfo["path"]),
                nodes,
                "Image Texture"
            )
            setTextureSize(facadeTextureInfo, image)
            # specular map
            if facadeTextureInfo.get("specularMapName"):
                setImage(
                    facadeTextureInfo["specularMapName"],
                    os.path.join(self.r.assetStore.baseDir, facadeTextureInfo["path"]),
                    nodes,
                    "Specular Map"
                )
        
        setTextureSize2(facadeTextureInfo, materialName, "Image Texture")
        return True
    
    def getFacadeMaterialTemplate(self, facadeTextureInfo, claddingTextureInfo, materialTemplateFilename):
        useCladdingColor = self.r.useCladdingColor and not facadeTextureInfo.get("noCladdingColor")
        useSpecularMap = facadeTextureInfo.get("specularMapName")
        
        if useSpecularMap and useCladdingColor:
            materialTemplateName = "facade_specular_color"
        elif useSpecularMap:
            materialTemplateName = "facade_specular"
        elif useCladdingColor:
            materialTemplateName = "facade_color"
        else:
            materialTemplateName = "export"
        
        return self.getMaterialTemplate(materialTemplateFilename, materialTemplateName)