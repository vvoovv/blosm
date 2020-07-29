import os
import bpy
from util.blender_extra.material import createMaterialFromTemplate, setImage
from ...util import setTextureSize, setTextureSize2


_claddingMaterialTemplateName = "tiles_color"


class ItemRendererMixin:
    """
    A mixin class
    """
    
    def getCladdingMaterialId(self, item, claddingTextureInfo):
        return claddingTextureInfo["name"]

    def createCladdingMaterial(self, materialName, claddingTextureInfo):
        materialTemplate = self.getMaterialTemplate(
            _claddingMaterialTemplateName
        )
        if not materialName in bpy.data.materials:
            nodes = createMaterialFromTemplate(materialTemplate, materialName)
            # The wall material (i.e. background) texture,
            # set it just in case
            image = setImage(
                claddingTextureInfo["name"],
                os.path.join(self.r.assetPackageDir, claddingTextureInfo["path"]),
                nodes,
                "Cladding"
            )
            setTextureSize(claddingTextureInfo, image)
        
        setTextureSize2(claddingTextureInfo, materialName, "Cladding")
        # return True for consistency with <self.getFacadeMaterialId(..)>
        return True
    
    def setVertexColor(self, item, face):
        color = item.getCladdingColor()
        if color:
            self.r.setVertexColor(face, color, self.r.layer.vertexColorLayerNameCladding)
    
    def getCladdingTextureInfo(self, item):
        return self._getCladdingTextureInfo(item)
    
    def createFacadeMaterial(self, item, materialName, facadeTextureInfo, claddingTextureInfo, uvs):
        if not materialName in bpy.data.materials:
            materialTemplate = self.getFacadeMaterialTemplate(
                facadeTextureInfo,
                claddingTextureInfo
            )
            nodes = createMaterialFromTemplate(materialTemplate, materialName)
            # the overlay texture
            image = setImage(
                facadeTextureInfo["name"],
                os.path.join(self.r.assetStore.baseDir, facadeTextureInfo["path"]),
                nodes,
                "Main"
            )
            setTextureSize(facadeTextureInfo, image)
            
            if claddingTextureInfo:
                # The wall material (i.e. background) texture,
                # set it just in case
                image = setImage(
                    claddingTextureInfo["name"],
                    os.path.join(self.r.assetPackageDir, claddingTextureInfo["path"]),
                    nodes,
                    "Cladding"
                )
                setTextureSize(claddingTextureInfo, image)
        
        setTextureSize2(facadeTextureInfo, materialName, "Main")
        if claddingTextureInfo:
            setTextureSize2(claddingTextureInfo, materialName, "Cladding")
        return True
    
    def renderExtra(self, item, face, facadeTextureInfo, claddingTextureInfo, uvs):
        # set UV-coordinates for the cladding texture
        if claddingTextureInfo:
            self.setCladdingUvs(item, face, claddingTextureInfo, uvs)
            self.setVertexColor(item, face)
        elif self.r.useCladdingColor and facadeTextureInfo.get("claddingColor"):
            self.setVertexColor(item, face)