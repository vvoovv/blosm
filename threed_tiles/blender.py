from os.path import basename, join as joinStrings, exists as pathExists
from os import remove as removeFile
from math import radians, atan, sqrt, pi

import bpy

from util.blender import createEmptyObject, createCollection


class BlenderRenderer:
    
    def __init__(self, threedTilesName, join3dTilesObjects):
        self.threedTilesName = threedTilesName
        self.join3dTilesObjects = join3dTilesObjects
        self.importedObjects = []
    
    def prepare(self, manager):
        self.collection = createCollection(self.threedTilesName)
    
    def finalize(self, manager):
        context = bpy.context
        # find avarage height of <self.importedObjects>
        #height = sum(obj.location[2] for obj in self.importedObjects)/len(self.importedObjects)
        centerCoords = manager.fromGeographic(manager.centerLat, manager.centerLon, 0.)
        
        rotationZ = -90 - manager.centerLon
        
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
        
        zOffset = min(obj.location[2] for obj in self.importedObjects)
        
        for obj in self.importedObjects:
            obj.location[2] -= zOffset
        
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
        
        self.importedObjects.clear()
        self.collection = None
    
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
        else:
            fileContent = manager.download(uri)
            with open(filePath, 'wb') as f:
                f.write(fileContent)
            removeFile(filePath)
        
        bpy.ops.import_scene.gltf(filepath=filePath)
        self.collection.objects.link(context.object)
        self.importedObjects.append(context.object)
        context.scene.collection.objects.unlink(context.object)