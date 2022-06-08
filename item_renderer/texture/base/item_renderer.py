import os
import bpy
from util.blender_extra.material import createMaterialFromTemplate, setImage
from ...util import setTextureSize, setTextureSize2, getPath


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
                getPath(self.r, claddingTextureInfo["path"]),
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
            self.r.setVertexColor(face, color, item.footprint.element.l, item.footprint.element.l.vertexColorLayerNameCladding)
    
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
                getPath(self.r, facadeTextureInfo["path"]),
                nodes,
                "Main"
            )
            setTextureSize(facadeTextureInfo, image)
            
            if claddingTextureInfo:
                # The wall material (i.e. background) texture,
                # set it just in case
                image = setImage(
                    claddingTextureInfo["name"],
                    getPath(self.r, claddingTextureInfo["path"]),
                    nodes,
                    "Cladding"
                )
                setTextureSize(claddingTextureInfo, image)
            elif facadeTextureInfo.get("specularMapName"):
                # specular map
                setImage(
                    facadeTextureInfo["specularMapName"],
                    getPath(self.r, facadeTextureInfo["path"]),
                    nodes,
                    "Specular Map"
                )
        
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