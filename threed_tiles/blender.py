from os.path import basename, join as joinStrings, exists as pathExists
from os import remove as removeFile
from math import radians, atan, sqrt, pi

import bpy

from util.blender import createEmptyObject


class BlenderRenderer:
    
    def __init__(self, ):
        self.importedObjects = []
    
    def prepare(self, manager):
        pass
    
    def finalize(self, manager):
        context = bpy.context
        # find avarage height of <self.importedObjects>
        #height = sum(obj.location[2] for obj in self.importedObjects)/len(self.importedObjects)
        centerCoords = manager.fromGeographic(manager.centerLat, manager.centerLon, 0.)
        
        rotationZ = -90 - manager.centerLon
        
        parentObject = createEmptyObject("3D Tiles", (0., 0., 0.))
        
        for obj in self.importedObjects:
            obj.parent = parentObject
        #    obj.matrix_parent_inverse = parentObject.matrix_world.inverted()
        
        parentObject.rotation_mode = 'ZXY'
        parentObject.rotation_euler[2] = radians(-90. - manager.centerLon)
        angle = atan(centerCoords[2]/sqrt(centerCoords[0]*centerCoords[0] + centerCoords[1]*centerCoords[1]))
        # <radians(manager.centerLat - 90.)> gives incorrect result
        # for the expression below
        parentObject.rotation_euler[0] = angle-pi/2.
        
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
        
        bpy.ops.object.join()
        
        self.importedObjects.clear()
    
    def renderGlb(self, manager, uri, path, cacheContent):
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
        self.importedObjects.append(bpy.context.object)