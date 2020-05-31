import os
import bpy
from util.blender_extra.material import createMaterialFromTemplate, setImage

from .container import Container
from ..level import CurtainWall as CurtainWallBase


class CurtainWall(CurtainWallBase, Container):
    
    def __init__(self):
        # a reference to the Container class used in the parent classes
        self.Container = Container
        Container.__init__(self, exportMaterials=False)
        CurtainWallBase.__init__(self)
    
    def createFacadeMaterial(self, materialName, facadeTextureInfo, claddingTextureInfo, uvs):
        materialTemplate = self.getFacadeMaterialTemplate(
            facadeTextureInfo,
            None,
            self.facadeMaterialTemplateFilename
        )
        if not materialName in bpy.data.materials:
            nodes = createMaterialFromTemplate(materialTemplate, materialName)
            # the overlay texture
            setImage(
                facadeTextureInfo["name"],
                os.path.join(self.r.assetsDir, facadeTextureInfo["path"]),
                nodes,
                "Image Texture"
            )
            # specular map
            if facadeTextureInfo.get("specularMapName"):
                setImage(
                    facadeTextureInfo["specularMapName"],
                    os.path.join(self.r.assetsDir, facadeTextureInfo["path"]),
                    nodes,
                    "Specular Map"
                )
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