import os
import bpy
from .item_renderer import ItemRendererMixin, _textureDir
from ..container import Container as ContainerBase
from ...util import setTextureSize


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
        color = self.getCladdingColorHex(item)
        return "%s_%s_%s" % (claddingTextureInfo["name"], color, facadeTextureInfo["name"])\
            if claddingTextureInfo and color else\
            ("%s_%s" % (color, facadeTextureInfo["name"]) if color else facadeTextureInfo["name"])
    
    def createFacadeMaterial(self, materialName, facadeTextureInfo, claddingTextureInfo, uvs):
        if not materialName in bpy.data.materials:
            if facadeTextureInfo.get("noCladdingTexture") and (not self.r.useCladdingColor or facadeTextureInfo.get("noCladdingColor")):
                # use the diffuse texture as is
                textureFilepath = os.path.join(
                    self.r.assetStore.baseDir,
                    facadeTextureInfo["path"],
                    facadeTextureInfo["name"]
                )
            else:
                # check if have texture in the data directory
                textureFilename, textureDir, textureFilepath = self.getTextureFilepath(materialName)
                if not os.path.isfile(textureFilepath):
                    self.makeTexture(
                        textureFilename,
                        textureDir,
                        textureFilepath,
                        self.claddingColor,
                        facadeTextureInfo,
                        claddingTextureInfo,
                        uvs
                    )
            
            image = self.createMaterialFromTemplate(materialName, textureFilepath)
            if not "textureSize" in facadeTextureInfo:
                setTextureSize(facadeTextureInfo, image)
        return True
    
    def makeTexture(self, textureFilename, textureDir, textureFilepath, textColor, facadeTextureInfo, claddingTextureInfo, uvs):
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
            facadeTextureInfo["path"],
            nodes,
            "facade_texture"
        )
        setTextureSize(facadeTextureInfo, image)
        
        if claddingTextureInfo:
            # cladding texture
            image = textureExporter.setImage(
                claddingTextureInfo["name"],
                claddingTextureInfo["path"],
                nodes,
                "cladding_texture"
            )
            setTextureSize(claddingTextureInfo, image)
            # scale for the cladding texture
            scaleFactor = claddingTextureInfo["textureWidthM"]/\
                claddingTextureInfo["textureSize"][0]*\
                (facadeTextureInfo["featureRpx"]-facadeTextureInfo["featureLpx"])/\
                facadeTextureInfo["featureWidthM"]
            textureExporter.setScaleNode(nodes, "Scale", scaleFactor, scaleFactor)
        # cladding color
        textureExporter.setColor(textColor, nodes, "cladding_color")
        # render the resulting texture
        textureExporter.renderTexture(scene, textureFilepath)
    
    def renderLevelGroupExtra(self, item, face, facadeTextureInfo, claddingTextureInfo, uvs):
        # do nothing here
        return