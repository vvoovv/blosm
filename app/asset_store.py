import os, json


_parts = ("facade", "level", "groundlevel", "entrance", "corner", "bottom")

_uses = (
    "any", "apartments", "single_family", "office", "mall", "retail", "hotel", "school", "university"
)

_facades = (
    "front", "side", "back", "shared", "all"
)

_claddings = (
    "brick",
    "plaster",
    "concrete",
    "metal",
    "glass",
    "gravel",
    "roof_tiles"
)


class AssetStore:
    
    def __init__(self, assetInfoFilepath):
        #
        # For assets representing building parts:
        # collection -> part -> class
        #
        # For cladding assets:
        # collection -> cladding -> class
        # 
        self.baseDir = os.path.dirname(os.path.dirname(assetInfoFilepath))
        
        self.collections = {}
        
        # building parts without collections
        self.textureParts = self.initPartsNoCols()
        # cladding without collections
        self.textureCladdings = self.initCladdingsNoCols()
        
        # building parts without collections
        self.meshParts = self.initPartsNoCols()
        
        self.hasMesh = self.hasTexture = False

        with open(assetInfoFilepath, 'r') as jsonFile:
            # getting asset entries for collections
            collections = json.load(jsonFile)["collections"]
            
            for collection in collections:
                assets = collection["assets"]
                if len(assets) == 1:
                    self.processNoCollectionAsset(assets[0])
                else:
                    collectionName = collection["name"]
                    collection = Collection()
                    if not collectionName in self.collections:
                        self.collections[collectionName] = EntryList()
                    self.collections[collectionName].addEntry(collection)
                    
                    for asset in assets:
                        self.processCollectionAsset(asset, collection)
    
    def processCollectionAsset(self, asset, collection):
        category = asset["category"]
        tp = asset["type"]
        cl = asset.get("class")
        
        if category == "part":
            if not cl:
                return
            parts = collection.meshParts if tp == "mesh" else collection.textureParts
            parts[ asset["part"] ][cl] = asset
        else: # cladding
            cladding = collection.textureCladdings[ asset["cladding"] ]
            # None is allowed for <cl>. The previous value for <cladding[None]> will be overriden
            cladding[cl] = asset
        
        self.processHasMeshOrTexture(tp)
    
    def processNoCollectionAsset(self, asset):
        category = asset["category"]
        tp = asset["type"]
        cl = asset.get("class")
        
        if category == "part":
            parts = self.meshParts if tp == "mesh" else self.textureParts
            part = parts[ asset["part"] ]
            if not cl in part:
                part[cl] = EntryList()
            part[cl].addEntry(asset)
        else: # cladding
            cladding = self.textureCladdings[ asset["cladding"] ]
            if not cl in cladding:
                cladding[cl] = EntryList()
            cladding[cl].addEntry(asset)
        
        self.processHasMeshOrTexture(tp)
    
    def processHasMeshOrTexture(self, tp):
        """
        Args:
            tp (str): An asset type (mesh or texture)
        """
        if tp == "mesh":
            if not self.hasMesh:
                self.hasMesh = True
        else: # texture
            if not self.hasTexture:
                self.hasTexture = True
    
    def initPartsNoCols(self):
        parts = {}
        
        for _part in _parts:
            parts[_part] = {}
        
        return parts
    
    def initCladdingsNoCols(self):
        claddings = {}
        
        for _cladding in _claddings:
            claddings[_cladding] = {}
        
        return claddings
    
    def getCollection(self, collection, key, cache):
        """
        Check if <collection> is available in <cache> using <key>.
        
        Returns:
            The value available in <cache> or a value from <self.collections> (including None).
            In the latter case the value is set in <cache> for <key>.
        """
        
        if key in cache:
            return cache[key]
        else:
            collection = self.getCollection(collection, cache)
            # get an instance of <EntryList>
            collection = self.collections.get(collection)
            if collection:
                collection = collection.getEntry()
            # save the resulting value in <cache>
            cache[key] = collection
        
        return collection
    
    def getAssetInfo(self, meshType, building, collection, buildingPart, cl):
        if not cl:
            return None
        
        cache = building.renderInfo._cache
        
        if collection:
            collection = self.getCollection(collection, "col_"+collection, cache)
            
            if collection:
                assetInfo = ( collection.meshParts[buildingPart] if meshType else collection.textureParts[buildingPart] ).get(cl)
                if assetInfo:
                    return assetInfo
                else:
                    # try to get an asset info without a collection in the code below
                    collection = None
        
        if not collection:
            # Check if an entry is available in <cache> for the given combination of <buildingPart> and <cl>
            # <pcl> stands for "(building) part" and "class"
            key = "pcl_" + buildingPart + cl
            if key in cache:
                assetInfo = cache[key]
            else:
                # get an instance of <EntryList>
                assetInfo = ( self.meshParts[buildingPart] if meshType else self.textureParts[buildingPart] ).get(cl)
                if assetInfo:
                    assetInfo = assetInfo.getEntry()
                # save the resulting value in <cache>
                cache[key] = assetInfo
        
        return assetInfo
    
    def getAssetInfoCladdingTexture(self, building, collection, cladding, cl):
        # <None> is allowed for <cl>
        
        cache = building.renderInfo._cache
        
        if collection:
            collection = self.getCollection(collection, "col_"+collection, cache)
            
            if collection:
                assetInfo = collection.textureCladdings[cladding].get(cl)
                if assetInfo:
                    return assetInfo
                else:
                    # try to get an asset info without a collection in the code below
                    collection = None
        
        if not collection:
            # Check if an entry is available in <cache> for the given combination of <cladding> and <cl>
            # <pcl> stands for "cladding" and "class"
            key = "ccl_" + (cladding + cl if cl else cladding)
            if key in cache:
                assetInfo = cache[key]
            else:
                # get an instance of <EntryList>
                assetInfo = self.textureCladdings[cladding].get(cl)
                if assetInfo:
                    assetInfo = assetInfo.getEntry()
                # save the resulting value in <cache>
                cache[key] = assetInfo
        
        return assetInfo


class Collection:
    
    __slots__ = (
        "textureParts",
        "textureCladdings"
        "meshParts",
        
    )
    
    def __init__(self):
        self.textureParts = {}
        for _part in _parts:
            self.textureParts[_part] = {}
        
        self.textureCladdings = {}
        for _cladding in _claddings:
            self.textureCladdings[_cladding] = {}
        
        self.meshParts = {}
        for _part in _parts:
            self.meshParts[_part] = {}


class EntryList:
    
    def __init__(self):
        self.index = 0
        # the largest index in <self.buildings>
        self.largestIndex = -1
        self.entries = []
    
    def addEntry(self, entry):
        self.entries.append(entry)
        self.largestIndex += 1
    
    def getEntry(self):
        if not self.entries:
            return None
        index = self.index
        if self.largestIndex:
            if index == self.largestIndex:
                self.index = 0
            else:
                self.index += 1
        return self.entries[index]