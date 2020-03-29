import os
import bpy
from .item_renderer import ItemRendererMixin, _textureDir
from ..container import Container as ContainerBase
from manager import Manager
from util.blender_extra.material import setImage


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
        color = self.getCladdingColor(item)
        return "%s_%s_%s" % (claddingTextureInfo["material"], color, facadeTextureInfo["name"])\
            if claddingTextureInfo and color\
            else facadeTextureInfo["name"]
    
    def createFacadeMaterial(self, materialName, facadeTextureInfo, claddingTextureInfo, uvs):
        if not materialName in bpy.data.materials:
            if facadeTextureInfo.get("noCladdingTexture") and (not self.r.useCladdingColor or facadeTextureInfo.get("noCladdingColor")):
                # use the diffuse texture as is
                textureFilepath = os.path.join(
                    self.r.bldgMaterialsDirectory,
                    facadeTextureInfo["path"],
                    facadeTextureInfo["name"]
                )
            else:
                # check if have texture in the data directory
                textureFilepath = self.getTextureFilepath(materialName)
                if not os.path.isfile(textureFilepath):
                    self.makeTexture(
                        materialName, # the file name of the texture
                        os.path.join(self.r.app.dataDir, _textureDir),
                        self.claddingColor,
                        facadeTextureInfo,
                        claddingTextureInfo,
                        uvs
                    )
            
            self.createMaterialFromTemplate(materialName, textureFilepath)
        return True
    
    def makeTexture(self, textureFilename, textureDir, textColor, facadeTextureInfo, claddingTextureInfo, uvs):
        textureExporter = self.r.textureExporter
        scene = textureExporter.getTemplateScene("compositing_facade_cladding_color")
        nodes = textureExporter.makeCommonPreparations(
            scene,
            textureFilename,
            textureDir
        )
        # facade texture
        setImage(
            facadeTextureInfo["name"],
            os.path.join(textureExporter.bldgMaterialsDirectory, facadeTextureInfo["path"]),
            nodes,
            "facade_texture"
        )
        if claddingTextureInfo:
            # cladding texture
            setImage(
                claddingTextureInfo["name"],
                os.path.join(textureExporter.bldgMaterialsDirectory, claddingTextureInfo["path"]),
                nodes,
                "cladding_texture"
            )
            # scale for the cladding texture
            scaleFactor = claddingTextureInfo["textureWidthM"]/\
                claddingTextureInfo["textureWidthPx"]*\
                (facadeTextureInfo["featureRpx"]-facadeTextureInfo["featureLpx"])/\
                facadeTextureInfo["featureWidthM"]
            self.setScaleNode(nodes, "Scale", scaleFactor, scaleFactor)
        # cladding color
        self.setColor(textColor, nodes, "cladding_color")
        # render the resulting texture
        textureExporter.renderTexture(scene, textureFilename, textureDir)
    
    def setScaleNode(self, nodes, nodeName, scaleX, scaleY):
        scaleInputs = nodes[nodeName].inputs
        scaleInputs[1].default_value = scaleX
        scaleInputs[2].default_value = scaleY
    
    def setTranslateNode(self, nodes, nodeName, translateX, translateY):
        translateInputs = nodes[nodeName].inputs
        translateInputs[1].default_value = translateX
        translateInputs[2].default_value = translateY
    
    def renderLevelGroupExtra(self, item, face, facadeTextureInfo, claddingTextureInfo, uvs):
        # do nothing here
        return