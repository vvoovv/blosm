import os, json


_parts = ("facade", "level", "groundlevel", "door")

_uses = (
    "appartments", "single_family", "office", "mall", "retail", "hotel", "school", "university"
)

_assetTypes = (
    "texture", "mesh"
)


class AssetStore:
    
    def __init__(self, assetInfoFilepath):
        self.baseDir = os.path.dirname(assetInfoFilepath)
        
        self.byUse = {}
        for _use in _uses:
            self.initUse(_use)
        
        self.byBuilding = []
        
        with open(assetInfoFilepath, 'r') as jsonFile:
            # getting asset entries for buildings
            buildings = json.load(jsonFile)["buildings"]
        
        for bldgIndex,building in enumerate(buildings):
            buildingInfo = {}
            self.byBuilding.append(buildingInfo)
            
            _use = building.get("use")
            if not _use in self.byUse:
                self.initUse(_use)
            
            byBldgClass = self.byUse[_use]["byBldgClass"]
            byPart = self.byUse[_use]["byPart"]
            
            _bldgClass = building.get("class")
            if _bldgClass:
                if not _bldgClass in byBldgClass:
                    byBldgClass[_bldgClass] = EntryList()
                byBldgClass[_bldgClass].addEntry(building)
            
            for partInfo in building.get("parts"):
                # inject <bldgIndex> into partInfo
                partInfo["_bldgIndex"] = bldgIndex
                _part = partInfo.get("part")
                if not _part in byPart:
                    self.initPart(_part, byPart)
                byType = byPart[_part]
                if not _part in buildingInfo:
                    self.initPart(_part, buildingInfo)
                
                _assetType = partInfo.get("type")
                if not _assetType in byType:
                    self.initAssetType(_assetType, byType)
                if not _assetType in buildingInfo[_part]:
                    self.initAssetType(_assetType, buildingInfo[_part])
                
                byClass = byType[_assetType]["byClass"]
                
                _class = partInfo.get("class")
                if _class:
                    if not _class in byClass:
                        byClass[_class] = EntryList()
                    byClass[_class].addEntry(partInfo)
                    if not _class in buildingInfo[_part][_assetType]["byClass"]:
                        buildingInfo[_part][_assetType]["byClass"][_class] = EntryList()
                    buildingInfo[_part][_assetType]["byClass"][_class].addEntry(partInfo)
                else:
                    byType[_assetType]["other"].addEntry(partInfo)
                    buildingInfo[_part][_assetType]["other"].addEntry(partInfo)
    
    def initUse(self, buildingUse):
        byPart = {}
        self.byUse[buildingUse] = dict(
            byBldgClass = {},
            byPart = byPart
        )
        for _part in _parts:
            self.initPart(_part, byPart)
    
    def initPart(self, buildingPart, byPart):
        byType = {}
        byPart[buildingPart] = byType
        for _assetType in _assetTypes:
            self.initAssetType(_assetType, byType)
    
    def initAssetType(self, assetType, byType):
        byType[assetType] = dict(
            byClass={},
            other=EntryList()
        )
    
    def getAssetInfoByClass(self, building, buildingPart, assetType, bldgClass, itemClass):
        if not building.metaStyleBlock:
            return None
        
        _use = building.use
        if not _use:
            return None
        
        use = self.byUse.get(_use)
        if not use:
            return None
        
        if bldgClass:
            pass
        else:
            # <itemClass> is given
            byPart = use["byPart"]
            byType = byPart.get(buildingPart)
            if not byType:
                return None
            if not assetType in byType:
                return None
            byClass = byType[assetType]["byClass"]
            assetInfo = byClass[itemClass].getEntry() if itemClass in byClass else byType[assetType]["other"].getEntry()
        return assetInfo
    
    def getAssetInfoByBldgIndexAndClass(self, bldgIndex, buildingPart, assetType, itemClass):
        byPart = self.byBuilding[bldgIndex]
        byType = byPart.get(buildingPart)
        if not byType:
            return None
        if not assetType in byType:
            return None
        byClass = byType[assetType]["byClass"]
        return byClass[itemClass].getEntry() if itemClass in byClass else byType[assetType]["other"].getEntry()


class EntryList:
    
    def __init__(self):
        self.index = -1
        # the largest index in <self.buildings>
        self.largestIndex = -1
        self.entries = []
    
    def addEntry(self, entry):
        self.entries.append(entry)
        self.largestIndex += 1
    
    def getEntry(self):
        index = self.index
        if self.largestIndex:
            if index == self.largestIndex:
                self.index = 0
            else:
                self.index += 1
        return self.entries[index]


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