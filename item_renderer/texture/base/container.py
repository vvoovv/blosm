import os
import bpy
from .item_renderer import ItemRendererMixin
from ..container import Container as ContainerBase

from util.blender_extra.material import createMaterialFromTemplate, setImage


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
        materialTemplate = self.getFacadeMaterialTemplate(
            facadeTextureInfo,
            claddingTextureInfo,
            self.facadeMaterialTemplateFilename
        )
        if not materialName in bpy.data.materials:
            nodes = createMaterialFromTemplate(materialTemplate, materialName)
            # the overlay texture
            setImage(
                facadeTextureInfo["name"],
                os.path.join(self.r.assetsDir, facadeTextureInfo["path"]),
                nodes,
                "Overlay"
            )
            if claddingTextureInfo:
                # The wall material (i.e. background) texture,
                # set it just in case
                setImage(
                    claddingTextureInfo["name"],
                    os.path.join(self.r.assetsDir, claddingTextureInfo["path"]),
                    nodes,
                    "Wall Material"
                )
        return True

    def getFacadeMaterialId(self, item, facadeTextureInfo, claddingTextureInfo):
        return "%s_%s" % (facadeTextureInfo["name"], claddingTextureInfo["name"])\
            if claddingTextureInfo\
            else facadeTextureInfo["name"]