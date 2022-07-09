from math import sin, radians
from operator import itemgetter

from util.blender import linkCollectionFromFile
from ..util import getFilepath


class Corner:
    
    def processCollection(self, item, assetInfo, indices):
        
        # a quick validity check
        if not item.cornerL and not item.cornerR:
            return
        
        facade = item.facade
        
        leftCorner = item.cornerL
        
        # Create a key as a z-coordinate of the bottom of <item>. Note that strings as keys
        # for Python dictionaries and sets work faster than floats
        cornerKey = str( round(item.building.renderInfo.verts[indices[0]][2], 3) )
        
        if leftCorner:
            if not facade.cornerInfoL:
                facade.cornerInfoL = facade.vector.prev.facade.cornerInfoR = set()
            cornerInfo = facade.cornerInfoL
        else:
            if not facade.cornerInfoR:
                facade.cornerInfoR = facade.vector.next.facade.cornerInfoL = set()
            cornerInfo = facade.cornerInfoR
        
        if cornerKey in cornerInfo:
            # the corner item was already inserted in the adjacent facade
            return
        
        collectionName = assetInfo["collection"]
        # path to a Blender file defined in <assetInfo>
        filepath = getFilepath(self.r, assetInfo)
        collectionKey = filepath + "_" + collectionName
        
        if not collectionKey in self.r.blenderCollections:
            collection = linkCollectionFromFile(filepath, collectionName)
            if not collection:
                return
            # The first Python list is for corner with convex angles, the second one is
            # for the corners with concave angles
            collectionInfo = (
                collection,
                [ ( obj, sin(radians(obj["angle"])) ) for obj in collection.objects if obj["angle"]>0.],
                [ ( obj, sin(radians(obj["angle"])) ) for obj in collection.objects if obj["angle"]<0.]
            )
            collectionInfo[1].sort(key=itemgetter(1))
            collectionInfo[2].sort(key=itemgetter(1))
            self.r.blenderCollections[collectionKey] = collectionInfo
        
        # If the neighbor corner item in the adjacent facade is visited, not actions will be needed,
        # since the corner item has been already inserted here.
        cornerInfo.add(cornerKey)
        
        return