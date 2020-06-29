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
        self.baseDir = os.path.join(os.path.dirname(assetInfoFilepath), os.pardir)
        
        # For parts with class:
        #   * use -> part -> assetType -> byClass[class]
        #   * building -> part -> assetType -> byClass[class]
        
        # For parts without class:
        #   * use -> part -> assetType -> other
        #   * building -> part -> assetType -> other
        
        # For cladding with use and with class:
        #   * use -> cladding -> assetType -> byClass[class]
        #   * building -> cladding -> assetType -> byClass[class]
        
        # For cladding without use and with class:
        #   * use(None) -> cladding -> assetType -> byClass[class]
        #   * building -> cladding -> assetType -> byClass[class]

        # For cladding with use and without class:
        #   * use -> cladding -> assetType -> other
        #   * building -> cladding -> assetType -> other
        
        # For cladding withiout use and without class:
        #   * use(None) -> cladding -> assetType -> other
        #   * building -> cladding -> assetType -> other
        
        self.byUse = {}
        for _use in _uses:
            self.initUse(_use)
        # the special case for cladding without a specific use
        self.byUse[None] = dict(
            byCladding = {},
            byPart = {}
        )
        
        self.byBuilding = []
        
        with open(assetInfoFilepath, 'r') as jsonFile:
            # getting asset entries for buildings
            buildings = json.load(jsonFile)["buildings"]
        
        for bldgIndex,building in enumerate(buildings):
            buildingInfo = dict(byPart={}, byCladding={})
            self.byBuilding.append(buildingInfo)
            
            parts = building.get("parts") or tuple()
            cladding = building.get("cladding") or tuple()
            
            _use = building.get("use")
            if parts and not _use:
                continue
            if not _use in self.byUse:
                self.initUse(_use)
            
            byPart = self.byUse[_use]["byPart"]
            
            _bldgClass = building.get("class")
            if _bldgClass:
                byBldgClass = self.byUse[_use]["byBldgClass"]
                if not _bldgClass in byBldgClass:
                    byBldgClass[_bldgClass] = EntryList()
                byBldgClass[_bldgClass].addEntry(building)
            
            for partInfo in parts:
                # inject <bldgIndex> into <partInfo>
                partInfo["_bldgIndex"] = bldgIndex
                _part = partInfo.get("part")
                if not _part in byPart:
                    self.initPart(_part, byPart)
                byType = byPart[_part]
                # the same for <buildingInfo>
                if not _part in buildingInfo["byPart"]:
                    self.initPart(_part, buildingInfo["byPart"])
                
                _assetType = partInfo.get("type")
                if not _assetType in byType:
                    self.initAssetType(_assetType, byType)
                # the same for <buildingInfo>
                if not _assetType in buildingInfo["byPart"][_part]:
                    self.initAssetType(_assetType, buildingInfo["byPart"][_part])
                
                byClass = byType[_assetType]["byClass"]
                
                _class = partInfo.get("class")
                if _class:
                    if not _class in byClass:
                        byClass[_class] = EntryList()
                    byClass[_class].addEntry(partInfo)
                    # the same for <buildingInfo>
                    if not _class in buildingInfo["byPart"][_part][_assetType]["byClass"]:
                        buildingInfo["byPart"][_part][_assetType]["byClass"][_class] = EntryList()
                    buildingInfo["byPart"][_part][_assetType]["byClass"][_class].addEntry(partInfo)
                else:
                    byType[_assetType]["other"].addEntry(partInfo)
                    # the same for <buildingInfo>
                    buildingInfo["byPart"][_part][_assetType]["other"].addEntry(partInfo)
            
            for claddingInfo in cladding:
                # inject <bldgIndex> into <claddingInfo>
                claddingInfo["_bldgIndex"] = bldgIndex
                _material = claddingInfo.get("material")
                
                # <_use> can be also equal to None, e.g. not present in <building>
                if not _material in self.byUse[_use]["byCladding"]:
                    byType = {}
                    self.byUse[_use]["byCladding"][_material] = byType
                    for _assetType in _assetTypes:
                        self.initAssetType(_assetType, byType)
                # the same for <buildingInfo>
                if not _material in buildingInfo:
                    byType = {}
                    buildingInfo["byCladding"][_material] = byType
                    for _assetType in _assetTypes:
                        self.initAssetType(_assetType, byType)
                
                _assetType = claddingInfo.get("type")
                _class = claddingInfo.get("class")
                if _class:
                    byClass = self.byUse[_use]["byCladding"][_material][_assetType]["byClass"]
                    if not _class in byClass:
                        byClass[_class] = EntryList()
                    byClass[_class].addEntry(claddingInfo)
                    # the same for <buildingInfo>
                    byClass = buildingInfo["byCladding"][_material][_assetType]["byClass"]
                    if not _class in byClass:
                        byClass[_class] = EntryList()
                    byClass[_class].addEntry(claddingInfo)
                else:
                    self.byUse[_use]["byCladding"][_material][_assetType]["other"].addEntry(claddingInfo)
                    # the same for <buildingInfo>
                    buildingInfo["byCladding"][_material][_assetType]["other"].addEntry(claddingInfo)
    
    def initUse(self, buildingUse):
        byPart = {}
        self.byUse[buildingUse] = dict(
            byBldgClass = {},
            byPart = byPart,
            byCladding = {}
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
        byPart = self.byBuilding[bldgIndex]["byPart"]
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