import os, json
import bpy
from app import app


_parts = ("level", "groundlevel", "entrance-door")

_uses = (
    "residential", "single-family", "office", "mall", "retail", "hotel", "school", "university"
)

# laf stands for "look and feel"
_lafs = (
    "modern", "neoclassical"
)


def loadTextFromFile(filepath, name):
    """
    Loads a Blender collection with the given <name> from the .blend file with the given <filepath>
    """
    with bpy.data.libraries.load(filepath) as (data_from, data_to):
        # a Python list (not a Python tuple!) must be set to <data_to.meshes>
        data_to.texts = [name]
    return data_to.texts[0]


def _getContentKeyWithNumbers(content, sortedContentKeys):
    return "".join(key[0]+str(content[key]) for key in sortedContentKeys)

def _getContentKeyWithoutNumbers(content, sortedContentKeys):
    return "".join(key[0] for key in sortedContentKeys if content[key])


class TextureStore:
    
    textureInfoFilename = "texture_info.blend"
    
    def __init__(self):
        self.byPart = {}
        byTextureFamily = []
        self.byTextureFamily = byTextureFamily
        
        for _part in _parts:
            self.initPart(_part)
        
        textures = json.loads(
            loadTextFromFile(
                os.path.join(
                    os.path.dirname(os.path.abspath(app.bldgMaterialsFilepath)),
                    self.textureInfoFilename
                ),
                "textures_facade"
            ).as_string()
        )["textures"]
        
        # To simplify the code start the count of <byTextureFamily> from
        # 1 instead 0, i.e. <if textureFamily> instead of <if not textureFamily is None>
        byTextureFamily.append(None)
        textureFamily = 1
        
        for texture in textures:
            if isinstance(texture, dict):
                self.initTextureInfo(texture, None)
            else: # Python list
                # we have a texture family
                textureFamilyData = {}
                byTextureFamily.append(textureFamilyData)
                for textureInfo in texture:
                    
                    buildingPart = textureInfo["part"]
                    if not buildingPart in textureFamilyData:
                        textureFamilyData[buildingPart] = {}
                    part = textureFamilyData[buildingPart]
                    
                    buildingLaf = textureInfo["laf"]
                    if not buildingLaf in part:
                        part[buildingLaf] = {}
                    
                    self.initTextureInfo(textureInfo, textureFamily)
                textureFamily += 1
    
    def initTextureInfo(self, textureInfo, textureFamily):
        byPart = self.byPart
        byTextureFamily = self.byTextureFamily
        
        buildingPart = textureInfo["part"]
        if not buildingPart in byPart:
            byPart[buildingPart] = {}
        part = byPart[buildingPart]
        
        buildingUse = textureInfo["use"]
        if not buildingUse in part:
            part[buildingUse] = {}
        use = part[buildingUse]
        
        # laf stands for "look and feel"
        buildingLaf = textureInfo["laf"]
        if not buildingLaf in use:
            use[buildingLaf] = {}
        laf = use[buildingLaf]
        
        # create keys
        content = textureInfo["content"]
        sortedContentKeys = sorted(content)
        
        # content with numbers
        key = _getContentKeyWithNumbers(content, sortedContentKeys)
        if key not in laf:
            laf[key] = TextureBundle()
        laf[key].addTextureInfo(
            textureInfo,
            textureFamily,
            key,
            byTextureFamily[textureFamily][buildingPart][buildingLaf] if textureFamily else None
        )
        
        # content without numbers
        key = _getContentKeyWithoutNumbers(content, sortedContentKeys)
        if key not in laf:
            laf[key] = TextureBundle()
        laf[key].addTextureInfo(
            textureInfo,
            textureFamily,
            key,
            byTextureFamily[textureFamily][buildingPart][buildingLaf] if textureFamily else None
        )
    
    def initPart(self, buildingPart):
        part = {}
        self.byPart[buildingPart] = part
        for _use in _uses:
            self.initUse(_use, part)
    
    def initUse(self, buildingUse, part):
        use = {}
        part[buildingUse] = use
        for _laf in _lafs:
            self.initLaf(_laf, use)
    
    def initLaf(self, buildingLaf, use):
        use[buildingLaf] = {}
    
    def getTextureInfo(self, building, buildingPart, facadePatternInfo):
        buildingAttrs = building.metaStyleBlock.attrs
        buildingPart = self.byPart.get(buildingPart)
        if not buildingPart:
            return None
        buildingUse = buildingPart.get(buildingAttrs.get("buildingUse"))
        if not buildingUse:
            return None
        buildingLaf = buildingUse.get(buildingAttrs.get("buildingLaf"))
        if not buildingLaf:
            return None
        textureBundle = buildingLaf.get(_getContentKeyWithoutNumbers(facadePatternInfo, sorted(facadePatternInfo)))
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
    
    def addTextureInfo(self, textureInfo, textureFamily, key, textureFamilyData):
        if textureFamily:
            lastTexture = self.textures[self.largestIndex] if self.textures else None
            if lastTexture and lastTexture.textureFamily == textureFamily:
                # we have a similar texture from the same texture family
                if isinstance(lastTexture, TextureSingle):
                    # replace it with an instance of TextureFamily
                    self.textures[self.largestIndex] = TextureFamily(lastTexture.textureInfo, textureInfo, textureFamily)
                    textureFamilyData[key] = self.textures[self.largestIndex]
                else:
                    lastTexture.addTextureInfo(textureInfo)
            else:
                # A new texture family is arrived
                # We don't know yet if there will be another similar texture
                # from the same texture family; so we use the <TextureSingle> wrapper for now
                self.textures.append(TextureSingle(textureInfo, textureFamily))
                self.largestIndex += 1
                textureFamilyData[key] = self.textures[self.largestIndex]
        else:
            self.textures.append(TextureSingle(textureInfo, textureFamily))
            self.largestIndex += 1
    
    def getTextureInfo(self):
        index = self.index
        if self.largestIndex:
            if index == self.largestIndex:
                self.index = 0
            else:
                self.index += 1
        return self.textures[index].getTextureInfo()


class TextureSingle:
    """
    A wrapper for a single texture to store it in the texture bundle
    """
    def __init__(self, textureInfo, textureFamily):
        self.textureInfo = textureInfo
        self.textureFamily = textureFamily
    
    def getTextureInfo(self):
        return self.textureInfo


class TextureFamily:
    """
    A wrapper for a family of textures to store them in the texture bundle
    """
    def __init__(self, textureInfo1, textureInfo2, textureFamily):
        self.index = 0
        self.textures = [textureInfo1, textureInfo2]
        # the largest index in <self.textures>
        self.largestIndex = 1
        self.textureFamily = textureFamily

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