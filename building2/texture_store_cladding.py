import os, json
from app import app


class CladdingTextureStore:
    
    textureInfoFilename = "texture_info_cladding.json"
    
    def __init__(self, exportMaterials):
        byMaterial = {}
        self.byMaterial = byMaterial
        
        with open(
            os.path.join(
                os.path.dirname(os.path.abspath(app.bldgMaterialsFilepath)),
                "%s_256.json" % self.textureInfoFilename[:-5] if exportMaterials else self.textureInfoFilename
            ),
        'r') as jsonFile:
            textures = json.load(jsonFile)["textures"]
        
        for texture in textures:
            material = texture["material"]
            if not material in byMaterial:
                byMaterial[material] = TextureBundle()
            byMaterial[material].addTextureInfo(texture)
    
    def getTextureInfo(self, material):
        textureBundle = self.byMaterial.get(material)
        return textureBundle.getTextureInfo() if textureBundle else None


class TextureBundle:
    """
    A structure to store textures for buildings that are similar in look and feel
    """
    
    def __init__(self):
        self.index = 0
        # the largest index in <self.textures>
        self.largestIndex = -1
        self.textures = []
    
    def addTextureInfo(self, textureInfo):
        self.textures.append(textureInfo)
        self.largestIndex += 1

    def getTextureInfo(self):
        index = self.index
        if self.largestIndex:
            if index == self.largestIndex:
                self.index = 0
            else:
                self.index += 1
        return self.textures[index]