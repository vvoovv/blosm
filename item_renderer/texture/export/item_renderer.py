import os
import bpy
from util.blender_extra.material import createMaterialFromTemplate, setImage
from ...util import setTextureSize, setTextureSize2


_textureDir = "texture"
_materialTemplateName = "export"


class ItemRendererMixin:
    """
    A mixin class
    """
    
    def getCladdingMaterialId(self, item, claddingTextureInfo):
        color = self.getCladdingColorHex(item)
        return "%s_%s%s" % (color, claddingTextureInfo["material"], os.path.splitext(claddingTextureInfo["name"])[1])\
            if claddingTextureInfo and color\
            else claddingTextureInfo["name"]
    
    def createCladdingMaterial(self, materialName, claddingTextureInfo):
        if not materialName in bpy.data.materials:
            # check if have texture in the data directory
            textureFileName, textureDir, textureFilepath = self.getTextureFilepath(materialName)
            if not os.path.isfile(textureFilepath):
                self.makeCladdingTexture(
                    textureFileName,
                    textureDir,
                    textureFilepath,
                    claddingTextureInfo
                )
            
            self.createMaterialFromTemplate(materialName, textureFilepath)
        return True
    
    def makeCladdingTexture(self, textureFilename, textureDir, textureFilepath, claddingTextureInfo):
        textureExporter = self.r.textureExporter
        scene = textureExporter.getTemplateScene("compositing_cladding_color")
        nodes = textureExporter.makeCommonPreparations(
            scene,
            textureFilename,
            textureDir
        )
        # cladding texture
        image = textureExporter.setImage(
            claddingTextureInfo["name"],
            claddingTextureInfo["path"],
            nodes,
            "cladding_texture"
        )
        setTextureSize(claddingTextureInfo, image)
        # cladding color
        textureExporter.setColor(self.claddingColor, nodes, "cladding_color")
        # render the resulting texture
        textureExporter.renderTexture(scene, textureFilepath)
    
    def getCladdingColorHex(self, item):
        color = item.getCladdingColor()
        # remember the color for a future use in the next funtion call
        self.claddingColor = color
        # return a hex string
        return "{:02x}{:02x}{:02x}".format(round(255*color[0]), round(255*color[1]), round(255*color[2]))
    
    def getTextureFilepath(self, materialName):
        textureFilename = "baked_%s" % materialName
        textureDir = os.path.join(self.r.app.dataDir, _textureDir)
        return textureFilename, textureDir, os.path.join(textureDir, textureFilename)
    
    def createMaterialFromTemplate(self, materialName, textureFilepath):
        materialTemplate = self.getMaterialTemplate(_materialTemplateName)
        
        nodes = createMaterialFromTemplate(materialTemplate, materialName)
        # the texture
        image = setImage(
            textureFilepath,
            None,
            nodes,
            "Main"
        )
        return image
    
    def getCladdingTextureInfo(self, item):
        if self.r.cacheCladdingTextureInfo:
            _cache = self.r._cache
            claddingMaterial = item.getStyleBlockAttrDeep("claddingMaterial")
            if not claddingMaterial in _cache:
                _cache[claddingMaterial] = self._getCladdingTextureInfo(item)
            return _cache[claddingMaterial]
        else:
            return self._getCladdingTextureInfo(item)
    
    def createFacadeMaterial(self, item, materialName, facadeTextureInfo, claddingTextureInfo, uvs):
        if not materialName in bpy.data.materials:
            if not facadeTextureInfo.get("cladding"):
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
                        item,
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
        
        setTextureSize2(facadeTextureInfo, materialName, "Main")
        return True
    
    def renderExtra(self, item, face, facadeTextureInfo, claddingTextureInfo, uvs):
        # do nothing here
        return