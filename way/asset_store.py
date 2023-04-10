import os, json
from app.asset_store import EntryList


class AssetType:
    material = 1
    mesh = 2


class AssetPart:
    roadway = "roadway"
    sidewalk = "sidewalk"
    pavement = "pavement"
    ground = "ground"

_parts = [_part for _part in dir(AssetPart) if not _part.startswith('__')]


class AssetStore:
    
    def __init__(self, assetInfoFilepath):
        #
        # collection -> class
        #
        self.baseDir = os.path.dirname(os.path.dirname(assetInfoFilepath))
        
        self.groups = {}
        
        # materials without collections
        self.materials = self.initNoGroups()
        
        self.groupCache = {}
        self.materialCache = {}

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
        entries = group.materials if asset["type"] == "material" else None
        
        entries[asset.get("part")][asset.get("class")] = asset
    
    def processNoGroupAsset(self, asset):
        cl = asset.get("class")
        
        entries = (self.materials if asset["type"] == "material" else None).get(asset["part"])
        
        if not cl in entries:
            entries[cl] = EntryList()
        entries[cl].addEntry(asset)
    
    def getAssetInfo(self, assetType, group, streetPart, cl):
        # <cl> equal to <None> is allowed
        
        if group:
            group = self.getGroup(group)
            
            if group:
                assetInfo = ( group.materials[streetPart] if assetType == AssetType.material else None ).get(cl)
                if assetInfo:
                    return assetInfo
                else:
                    # try to get an asset info without a group in the code below
                    group = None
        
        if not group:
            entries, cache = None, None
            if assetType == AssetType.material:
                entries, cache = self.materials, self.materialCache
            # Check if an entry is available in <cache> for the given combination of <streetPart> and <cl>
            key = streetPart + ("_" + cl if cl else "_None")
            if key in cache:
                assetInfo = cache[key]
            else:
                # get an instance of <EntryList>
                assetInfo = entries[streetPart].get(cl)
                if assetInfo:
                    assetInfo = assetInfo.getEntry()
                # save the resulting value in <cache>
                cache[key] = assetInfo
        
        return assetInfo
    
    def getGroup(self, groupName):
        """
        Check if <groupName> is available in <self.groupCache>.
        
        Returns:
            The value available in <self.groupCache> or a value from <self.groups> (including None).
            In the latter case the value is set in <self.groupCache> for <groupName>.
        """
        
        cache = self.groupCache
        
        # <groupName> is used as a key for <cache>
        if groupName in cache:
            return cache[groupName]
        else:
            # get an instance of <EntryList>
            group = self.groups.get(groupName)
            if group:
                group = group.getEntry()
            # Save the resulting value in <cache>. Next time a group with
            # the name <groupName> is needed, the one saved in <cache[groupName]> will be used
            cache[groupName] = group
        
        return group
    
    def initNoGroups(self):
        entries = {}
        
        for _part in _parts:
            entries[_part] = {}
        
        return entries


class Group:
    
    __slots__ = (
        "materials"
    )
    
    def __init__(self):
        self.materials = {}
        for _part in _parts:
            self.materials[_part] = {}