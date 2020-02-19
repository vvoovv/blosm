"""
This file is part of blender-osm (OpenStreetMap importer for Blender).
Copyright (C) 2014-2018 Vladimir Elistratov
prokitektura+support@gmail.com

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import bpy
import os, math
from . import Renderer, assignTags
from util.blender import loadCollectionFromFile, loadGroupFromFile
from util.osm import parseNumber
from util.random import RandomNormal, RandomWeighted

_isBlender280 = bpy.app.version[1] >= 80


class BaseNodeRenderer(Renderer):
    
    def __init__(self, app, path, filename, collectionName, randomizeScale=False, randomizeRotation=False):
        super().__init__(app)
        
        self._initialized = False
        self.app = app
        
        filepath = os.path.realpath( os.path.join(path, filename) )
        if not os.path.isfile(filepath):
            raise Exception("The file %s with assets doesn't exist." % filepath)
        self.filepath = filepath
        self.collectionName = collectionName
        # number of objects in the collection with the name <self.collectionName>
        self.numObjects = 0
        # index of the current object from the colleciton with the name <self.collectionName>
        self.objectIndex = 0
        
        self.randomScale = RandomNormal(1., sigmaRatio=0.2) if randomizeScale else None
        self.randomRotation = RandomWeighted( tuple((math.radians(angle), 1) for angle in range(0, 360)) ) if randomizeRotation else None
    
    def renderNode(self, node, osm):
        tags = node.tags
        layer = node.l
        
        coords = node.getData(osm)
        
        self.init()
        
        collection = bpy.data.collections[self.collectionName] if _isBlender280 else bpy.data.groups[self.collectionName]
        
        # calculate z-coordinate of the object
        z = parseNumber(tags["min_height"], 0.) if "min_height" in tags else 0.
        if self.app.terrain:
            terrainOffset = self.app.terrain.project(coords)
            if terrainOffset is None:
                # the point is outside of the terrain
                return
            z += terrainOffset[2]
        
        obj = self.createBlenderObject(
            self.getName(node),
            (coords[0], coords[1], z),
            collection.objects[self.objectIndex].data,
            collection = layer.getCollection(self.collection) if _isBlender280 else None,
            parent = None if _isBlender280 else layer.getParent()
        )
        
        if obj:
            if self.randomRotation:
                obj.rotation_euler[2] = self.randomRotation.value
            if self.randomScale:
                scale = self.randomScale.value
                obj.scale = (scale, scale, scale)
            # assign OSM tags to the blender object
            assignTags(obj, node.tags)
            
            self.objectIndex += 1
            if self.objectIndex == self.numObjects:
                self.objectIndex = 0
    
    def init(self):
        if self._initialized:
            return
        if _isBlender280:
            collection = loadCollectionFromFile(self.filepath, self.collectionName)
            self.numObjects = len(collection.objects)
        else:
            group = loadGroupFromFile(self.filepath, self.collectionName)
            self.numObjects = len(group.objects)
        self._initialized = True
    
    @classmethod
    def createBlenderObject(self, name, location, objectData, collection=None, parent=None):
        obj = bpy.data.objects.new(name, objectData)
        if location:
            obj.location = location
        if _isBlender280:
            collection.objects.link(obj)
        else:
            bpy.context.scene.objects.link(obj)
        if parent:
            # perform parenting
            obj.parent = parent
        return obj


class SingleTreeRenderer(BaseNodeRenderer):
      
    collectionName = "trees"
    
    def __init__(self, app):
        path, filename = os.path.split(app.vegetationFilepath)
        super().__init__(app, path, filename, self.collectionName, randomizeScale=True, randomizeRotation=True)