import os
import bpy
from .item_renderer import ItemRendererMixin
from ..container import Container as ContainerBase

from util.blender_extra.material import createMaterialFromTemplate, setImage
from ...util import setTextureSize, setTextureSize2


class Container(ContainerBase, ItemRendererMixin):
    """
    The base class for the item renderers Facade, Div, Layer, Bottom
    """
    
    def renderLevelGroupExtra(self, item, face, facadeTextureInfo, claddingTextureInfo, uvs):
        # set UV-coordinates for the cladding texture
        if claddingTextureInfo:
            self.setCladdingUvs(item, face, claddingTextureInfo, uvs)
        if self.r.useCladdingColor and not facadeTextureInfo.get("noCladdingColor"):
            self.setVertexColor(item, face)
    
    def createFacadeMaterial(self, materialName, facadeTextureInfo, claddingTextureInfo, uvs):
        if not materialName in bpy.data.materials:
            materialTemplate = self.getFacadeMaterialTemplate(
                facadeTextureInfo,
                claddingTextureInfo,
                self.materialTemplateFilename
            )
            nodes = createMaterialFromTemplate(materialTemplate, materialName)
            # the overlay texture
            image = setImage(
                facadeTextureInfo["name"],
                os.path.join(self.r.assetStore.baseDir, facadeTextureInfo["path"]),
                nodes,
                "Overlay"
            )
            setTextureSize(facadeTextureInfo, image)
            
            if claddingTextureInfo:
                # The wall material (i.e. background) texture,
                # set it just in case
                image = setImage(
                    claddingTextureInfo["name"],
                    os.path.join(self.r.assetsDir, claddingTextureInfo["path"]),
                    nodes,
                    "Wall Material"
                )
                setTextureSize(claddingTextureInfo, image)
        
        setTextureSize2(facadeTextureInfo, materialName, "Overlay")
        setTextureSize2(claddingTextureInfo, materialName, "Wall Material")
        return True

    def getFacadeMaterialId(self, item, facadeTextureInfo, claddingTextureInfo):
        return "%s_%s" % (facadeTextureInfo["name"], claddingTextureInfo["name"])\
            if claddingTextureInfo\
            else facadeTextureInfo["name"]