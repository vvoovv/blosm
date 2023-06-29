from os.path import basename, join as joinStrings, exists as pathExists
from os import remove as removeFile
from math import radians, atan, sqrt, pi
import re
from operator import itemgetter

import bpy

from util.blender import createEmptyObject, createCollection


class BlenderRenderer:
    
    def __init__(self, threedTilesName, join3dTilesObjects):
        self.threedTilesName = threedTilesName
        self.join3dTilesObjects = join3dTilesObjects
        self.importedObjects = []
        
        self.calculateHeightOffset = False
        self.heightOffset = 0.
        
        self.licenseRePattern = re.compile(b'\"copyright\":\s*\"(\w+)\"')
        self.copyrightHolders = {}
    
    def prepare(self, manager):
        self.collection = createCollection(self.threedTilesName)
    
    def finalize(self, manager):
        if not self.importedObjects:
            self.collection = None
            return
        
        context = bpy.context
        # find avarage height of <self.importedObjects>
        #height = sum(obj.location[2] for obj in self.importedObjects)/len(self.importedObjects)
        centerCoords = manager.fromGeographic(manager.centerLat, manager.centerLon, 0.)
        
        parentObject = createEmptyObject(self.threedTilesName, (0., 0., 0.), collection=self.collection)
        
        for obj in self.importedObjects:
            obj.parent = parentObject
        #    obj.matrix_parent_inverse = parentObject.matrix_world.inverted()
        
        parentObject.rotation_mode = 'ZXY'
        parentObject.rotation_euler[2] = radians(-90. - manager.centerLon)
        lat = atan(centerCoords[2]/sqrt(centerCoords[0]*centerCoords[0] + centerCoords[1]*centerCoords[1]))
        # <radians(manager.centerLat - 90.)> gives incorrect result
        # for the expression below
        parentObject.rotation_euler[0] = lat-pi/2.
        
        bpy.ops.object.select_all(action='DESELECT')
        parentObject.select_set(True)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
        parentObject.select_set(False)
        
        for obj in self.importedObjects:
            obj.parent = None
            obj.select_set(True)
        
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
        
        heightOffset = min(obj.location[2] for obj in self.importedObjects)\
            if self.calculateHeightOffset else\
            self.heightOffset
        if self.calculateHeightOffset:
            self.heightOffset = heightOffset
        
        for obj in self.importedObjects:
            obj.location[2] -= heightOffset
        
        # join the selected objects
        if self.join3dTilesObjects:
            bpy.ops.object.join()
            joinedObject = self.importedObjects[-1]
            joinedObject.name = self.threedTilesName
            bpy.data.objects.remove(parentObject, do_unlink=True)
            context.view_layer.objects.active = joinedObject
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.remove_doubles()
            bpy.ops.object.mode_set(mode='OBJECT')
        else:
            for obj in self.importedObjects:
                obj.parent = parentObject
                obj.select_set(False)
        
        context.scene.blosm.copyright = "; ".join(
            entry[0] for entry in reversed(
                sorted(
                    self.copyrightHolders.items(), key=itemgetter(1)
                )
            )
        )
        
        self.importedObjects.clear()
        self.collection = None
        
        print("Import is finished!")
    
    def renderGlb(self, manager, uri, path, cacheContent):
        context = bpy.context
        
        filePath = joinStrings(
            manager.tilesDir,
            basename(path) if cacheContent else "current_file.glb"
        )
        
        if cacheContent:
            if not pathExists(filePath):
                fileContent = manager.download(uri)
                with open(filePath, 'wb') as f:
                    f.write(fileContent)
            bpy.ops.import_scene.gltf(filepath=filePath)
        else:
            fileContent = manager.download(uri)
            # check if <fileContent> contains copyright information
            match = re.search(self.licenseRePattern, fileContent)
            if match:
                self.processCopyrightInfo(match.group(1).decode('utf-8'))
            with open(filePath, 'wb') as f:
                f.write(fileContent)
            bpy.ops.import_scene.gltf(filepath=filePath)
            removeFile(filePath)
        
        self.collection.objects.link(context.object)
        self.importedObjects.append(context.object)
        context.scene.collection.objects.unlink(context.object)
    
    def processCopyrightInfo(self, info):
        for copyrightHolder in info.split(';'):
            copyrightHolder = copyrightHolder.strip()
            if not copyrightHolder in self.copyrightHolders:
                self.copyrightHolders[copyrightHolder] = 0
            self.copyrightHolders[copyrightHolder] += 1