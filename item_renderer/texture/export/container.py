import os
import bpy
from .item_renderer import ItemRendererMixin, _textureDir
from ..container import Container as ContainerBase
from ...util import setTextureSize, getPath


class Container(ContainerBase, ItemRendererMixin):
    """
    The base class for the item renderers Facade, Div, Layer, Bottom
    """
    
    def __init__(self, exportMaterials):
        super().__init__(exportMaterials)
        # The following variable is used to cache the cladding color as as string:
        # either a base colors (e.g. red, green) or a hex string
        self.claddingColor = None

    def getFacadeMaterialId(self, item, facadeTextureInfo, claddingTextureInfo):
        # set UV-coordinates for the cladding texture
        if claddingTextureInfo:
            color = self.getCladdingColorHex(item)
            return "%s_%s_%s" % (claddingTextureInfo["name"], color, facadeTextureInfo["name"])
        elif self.r.useCladdingColor and facadeTextureInfo.get("claddingColor"):
            color = self.getCladdingColorHex(item)
            return "%s_%s" % (color, facadeTextureInfo["name"])
        else:
            return facadeTextureInfo["name"]
    
    def makeTexture(self, item, textureFilename, textureDir, textureFilepath, textColor, facadeTextureInfo, claddingTextureInfo, uvs):
        textureExporter = self.r.textureExporter
        scene = textureExporter.getTemplateScene("compositing_facade_cladding_color")
        nodes = textureExporter.makeCommonPreparations(
            scene,
            textureFilename,
            textureDir
        )
        # facade texture
        image = textureExporter.setImage(
            facadeTextureInfo["name"],
            getPath(self.r, facadeTextureInfo["path"]),
            nodes,
            "facade_texture"
        )
        setTextureSize(facadeTextureInfo, image)
        
        if claddingTextureInfo:
            # cladding texture
            image = textureExporter.setImage(
                claddingTextureInfo["name"],
                getPath(self.r, claddingTextureInfo["path"]),
                nodes,
                "cladding_texture"
            )
            setTextureSize(claddingTextureInfo, image)
            # scale for the cladding texture
            scaleFactor = claddingTextureInfo["textureWidthM"]/\
                claddingTextureInfo["textureSize"][0]*\
                (
                    facadeTextureInfo["textureSize"][0]/item.width
                    if "class" in facadeTextureInfo else
                    (facadeTextureInfo["featureRpx"]-facadeTextureInfo["featureLpx"])/facadeTextureInfo["featureWidthM"]
                )
            textureExporter.setScaleNode(nodes, "Scale", scaleFactor, scaleFactor)
        # cladding color
        textureExporter.setColor(textColor, nodes, "cladding_color")
        # render the resulting texture
        textureExporter.renderTexture(scene, textureFilepath)