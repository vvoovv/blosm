import os, json


_parts = ("level", "groundlevel", "door")

_uses = (
    "appartments", "single_family", "office", "mall", "retail", "hotel", "school", "university"
)

# laf stands for "look and feel"
_lafs = (
    "modern", "neoclassical", "curtain_wall"
)


def _getContentKeyWithNumbers(content, buildingPart, sortedContentKeys):
    return "%s_%s" % ( buildingPart, ''.join(key[0]+str(content[key]) for key in sortedContentKeys) )

def _getContentKeyWithoutNumbers(content, buildingPart, sortedContentKeys):
    return "%s_%s" % ( buildingPart, ''.join(key[0] for key in sortedContentKeys if content[key]) )

def _any(content, buildingPart, sortedContentKeys):
    return buildingPart

# generators of keys from the most detailed one to the least detailed one
_keyGenerators = (
    _getContentKeyWithNumbers,
    _getContentKeyWithoutNumbers,
    _any
)

_numKeyGenerators = len(_keyGenerators)


class FacadeTextureStore:
    
    def __init__(self, assetInfoFilepath):
        self.baseDir = os.path.dirname(assetInfoFilepath)
        # The following Python dictionary is used to calculated the number of windows and balconies
        # in the Level pattern
        self.facadePatternInfo = dict(Window=0, Balcony=0, Door=0)
        
        self.byPart = {}
        byTextureFamily = []
        self.byTextureFamily = byTextureFamily
        
        for _part in _parts:
            self.initPart(_part)
        
        with open(assetInfoFilepath, 'r') as jsonFile:
            textures = json.load(jsonFile)["textures"]
        
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
        key = buildingPart
        if not key in laf:
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
            key = _getContentKeyWithNumbers(content, buildingPart, sortedContentKeys)
            if key not in laf:
                laf[key] = TextureBundle()
            laf[key].addTextureInfo(
                textureInfo,
                textureFamily,
                key,
                byTextureFamily[textureFamily][buildingPart][buildingLaf] if textureFamily else None
            )
            
            # content without numbers
            key = _getContentKeyWithoutNumbers(content, buildingPart, sortedContentKeys)
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
    
    def getTextureInfo(self, building, buildingPart, item, itemRenderer):
        if not building.metaStyleBlock:
            return None
        buildingAttrs = building.metaStyleBlock.attrs
        _buildingPart = self.byPart.get(buildingPart)
        if not _buildingPart:
            return None
        buildingUse = _buildingPart.get(buildingAttrs.get("buildingUse"))
        if not buildingUse:
            return None
        buildingLaf = buildingUse.get(item.laf or buildingAttrs.get("buildingLaf"))
        if not buildingLaf:
            return None
        
        facadePatternInfo = self.getFacadePatternInfo(item, itemRenderer)
        
        cache = building._cache
        if facadePatternInfo:
            _keys = self._keys
            _keys.clear()
            _sorted = sorted(facadePatternInfo)
            # try the keys from the most detailed one to the least detailed one
            for keyIndex, key in enumerate(_keyGenerators):
                key = key(facadePatternInfo, buildingPart, _sorted)
                if key in cache:
                    texture = cache[key]
                    # set the cache for more detailed keys
                    if keyIndex:
                        for _keyIndex in range(keyIndex):
                            cache[_keys[_keyIndex]] = texture
                    break
                else:
                    textureBundle = buildingLaf.get(key)
                    if textureBundle:
                        texture = textureBundle.getTexture()
                        cache[key] = texture
                        # set the cache for more detailed keys
                        if keyIndex:
                            for _keyIndex in range(keyIndex):
                                cache[_keys[_keyIndex]] = texture
                        # check if need to set the cache for less detailed keys
                        for _keyIndex in range(keyIndex+1, _numKeyGenerators):
                            key = _keyGenerators[_keyIndex](facadePatternInfo, _sorted)
                            if not key in cache:
                                cache[key] = texture
                        break
                    else:
                        _keys.append(key)
        else:
            key = buildingPart
            if key in cache:
                texture = cache[key]
            else:
                # that key must be present in <buildingLaf>
                texture = buildingLaf.get(key).getTexture()
                cache[key] = texture
        return texture.getTextureInfo()
    
    def getFacadePatternInfo(self, item, itemRenderer):
        if itemRenderer.facadePatternInfo:
            # it's the special case for the door
            return itemRenderer.facadePatternInfo
        elif not item.markup or not item.hasFacadePatternInfo:
            return None
        
        facadePatternInfo = self.facadePatternInfo
        # reset <facadePatternInfo>
        for key in facadePatternInfo:
            facadePatternInfo[key] = 0
        # initalize <facadePatternInfo>
        for _item in item.markup:
            className = _item.__class__.__name__
            if className in facadePatternInfo:
                facadePatternInfo[className] += 1
        return facadePatternInfo


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
    
    def getTexture(self):
        index = self.index
        if self.largestIndex:
            if index == self.largestIndex:
                self.index = 0
            else:
                self.index += 1
        return self.textures[index]


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