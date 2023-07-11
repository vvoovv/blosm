import os, json


_parts = ("facade", "level", "groundlevel", "entrance", "corner", "top", "bottom")

"""
# obsolete structures
_uses = (
    "any", "apartments", "single_family", "office", "mall", "retail", "hotel", "school", "university"
)

_facades = (
    "front", "side", "back", "shared", "all"
)
"""

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
        # group -> part -> class
        #
        # For cladding assets:
        # group -> cladding -> class
        # 
        self.baseDir = os.path.dirname(os.path.dirname(assetInfoFilepath))
        
        self.groups = {}
        
        # building parts without groups
        self.textureParts = self.initPartsNoGroups()
        # cladding without groups
        self.textureCladdings = self.initCladdingsNoGroups()
        
        # building parts without groups
        self.meshParts = self.initPartsNoGroups()
        
        self.hasMesh = self.hasTexture = False

        with open(assetInfoFilepath, 'r') as jsonFile:
            # getting asset entries for groups
            groups = json.load(jsonFile)["groups"]
            
            for group in groups:
                assets = group["assets"]
                if len(assets) == 1:
                    self.processNoGroupAsset(assets[0])
                else:
                    groupName = group["name"]
                    group = Group()
                    if not groupName in self.groups:
                        self.groups[groupName] = EntryList()
                    self.groups[groupName].addEntry(group)
                    
                    for asset in assets:
                        self.processGroupAsset(asset, group)
    
    def processGroupAsset(self, asset, group):
        category = asset["category"]
        tp = asset["type"]
        cl = asset.get("class")
        
        if category == "part":
            if not cl:
                return
            parts = group.meshParts if tp == "mesh" else group.textureParts
            parts[ asset["part"] ][cl] = asset
        else: # cladding
            cladding = group.textureCladdings[ asset["cladding"] ]
            # None is allowed for <cl>. The previous value for <cladding[None]> will be overriden
            cladding[cl] = asset
        
        self.processHasMeshOrTexture(tp)
    
    def processNoGroupAsset(self, asset):
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
    
    def initPartsNoGroups(self):
        parts = {}
        
        for _part in _parts:
            parts[_part] = {}
        
        return parts
    
    def initCladdingsNoGroups(self):
        claddings = {}
        
        for _cladding in _claddings:
            claddings[_cladding] = {}
        
        return claddings
    
    def getGroup(self, group, key, cache):
        """
        Check if <group> is available in <cache> using <key>.
        
        Returns:
            The value available in <cache> or a value from <self.groups> (including None).
            In the latter case the value is set in <cache> for <key>.
        """
        
        if key in cache:
            return cache[key]
        else:
            # get an instance of <EntryList>
            group = self.groups.get(group)
            if group:
                group = group.getEntry()
            # Save the resulting value in <cache>. Next time a group with
            # the name <group> is needed for this building, the one saved in <cache[key]> will be used
            cache[key] = group
        
        return group
    
    def getAssetInfo(self, meshType, building, group, buildingPart, cl):
        if not cl:
            return None
        
        cache = building.renderInfo._cache
        
        if group:
            group = self.getGroup(group, "gr_"+group, cache)
            
            if group:
                assetInfo = ( group.meshParts[buildingPart] if meshType else group.textureParts[buildingPart] ).get(cl)
                if assetInfo:
                    return assetInfo
                else:
                    # try to get an asset info without a group in the code below
                    group = None
        
        if not group:
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
    
    def getAssetInfoCladdingTexture(self, building, group, cladding, cl):
        # <None> is allowed for <cl>
        
        cache = building.renderInfo._cache
        
        if group:
            group = self.getGroup(group, "gr_"+group, cache)
            
            if group:
                assetInfo = group.textureCladdings[cladding].get(cl)
                if assetInfo:
                    return assetInfo
                else:
                    # try to get an asset info without a group in the code below
                    group = None
        
        if not group:
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


class Group:
    
    __slots__ = (
        "textureParts",
        "textureCladdings",
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