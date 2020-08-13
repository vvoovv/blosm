import os
import bpy
from util.blender_extra.material import createMaterialFromTemplate, setImage

from .container import Container
from ..level import CurtainWall as CurtainWallBase
from ...util import setTextureSize, setTextureSize2, getPath


class CurtainWall(CurtainWallBase, Container):
    
    def __init__(self):
        # a reference to the Container class used in the parent classes
        self.Container = Container
        Container.__init__(self, exportMaterials=False)
        CurtainWallBase.__init__(self)
    
    def createFacadeMaterial(self, item, materialName, facadeTextureInfo, claddingTextureInfo, uvs):
        if not materialName in bpy.data.materials:
            materialTemplate = self.getFacadeMaterialTemplate(
                facadeTextureInfo,
                None
            )
            nodes = createMaterialFromTemplate(materialTemplate, materialName)
            # the overlay texture
            image = setImage(
                facadeTextureInfo["name"],
                getPath(self.r, facadeTextureInfo["path"]),
                nodes,
                "Main"
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
        
        setTextureSize2(facadeTextureInfo, materialName, "Main")
        return True
    
    def getFacadeMaterialTemplate(self, facadeTextureInfo, claddingTextureInfo):
        useCladdingColor = self.r.useCladdingColor
        useSpecularMap = facadeTextureInfo.get("specularMapName")
        
        if useSpecularMap and useCladdingColor:
            materialTemplateName = "facade_specular_color"
        elif useSpecularMap:
            materialTemplateName = "facade_specular"
        elif useCladdingColor:
            materialTemplateName = "facade_color"
        else:
            materialTemplateName = "export"
        
        return self.getMaterialTemplate(materialTemplateName)