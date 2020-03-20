import os, json
from app import app
from util.blender import loadTextFromFile


_parts = ("level", "groundlevel", "door")

_uses = (
    "residential", "single_family", "office", "mall", "retail", "hotel", "school", "university"
)

# laf stands for "look and feel"
_lafs = (
    "modern", "neoclassical", "curtain_wall"
)


def _getContentKeyWithNumbers(content, sortedContentKeys):
    return ''.join(key[0]+str(content[key]) for key in sortedContentKeys)

def _getContentKeyWithoutNumbers(content, sortedContentKeys):
    return ''.join(key[0] for key in sortedContentKeys if content[key])

def _any(content, sortedContentKeys):
    return ''

# generators of keys from the most detailed one to the least detailed one
_keyGenerators = (
    _getContentKeyWithNumbers,
    _getContentKeyWithoutNumbers,
    _any
)

_numKeyGenerators = len(_keyGenerators)


class FacadeTextureStore:
    
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
        # an auxiliary Python list to store keys in the method <self.getTextureInfo(..)>
        self._keys = []
    
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
        
        #
        # create keys
        #
        
        # the key of all available textures
        key = ''
        if key not in laf:
            laf[key] = TextureBundle()
        laf[key].addTextureInfo(
            textureInfo,
            textureFamily,
            key,
            byTextureFamily[textureFamily][buildingPart][buildingLaf] if textureFamily else None
        )
        
        content = textureInfo.get("content")
        if content:
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
        
        cache = building._cache
        if facadePatternInfo:
            _keys = self._keys
            _keys.clear()
            _sorted = sorted(facadePatternInfo)
            # try the keys from the most detailed one to the least detailed one
            for keyIndex, key in enumerate(_keyGenerators):
                key = key(facadePatternInfo, _sorted)
                if key in cache:
                    textureInfo = cache[key]
                    # set the cache for more detailed keys
                    if keyIndex:
                        for _keyIndex in range(keyIndex):
                            cache[_keys[_keyIndex]] = textureInfo
                    break
                else:
                    textureBundle = buildingLaf.get(key)
                    if textureBundle:
                        textureInfo = textureBundle.getTextureInfo()
                        cache[key] = textureInfo
                        # set the cache for more detailed keys
                        if keyIndex:
                            for _keyIndex in range(keyIndex):
                                cache[_keys[_keyIndex]] = textureInfo
                        # check if need to set the cache for less detailed keys
                        for _keyIndex in range(keyIndex+1, _numKeyGenerators):
                            key = _keyGenerators[_keyIndex](facadePatternInfo, _sorted)
                            if not key in cache:
                                cache[key] = textureInfo
                        break
                    else:
                        _keys.append(key)
        else:
            key = ''
            if key in cache:
                textureInfo = cache[key]
            else:
                # that key must be present in <buildingLaf>
                textureInfo = buildingLaf.get(key).getTextureInfo()
                cache[key] = textureInfo
        return textureInfo


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