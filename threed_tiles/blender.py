from os.path import basename, join as joinStrings, exists as pathExists
from os import remove as removeFile
from math import radians, atan, sqrt, pi
import re
from operator import itemgetter
import sys

import bpy
from mathutils import Matrix

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
        
        self.gltfImporterPatched = False
        # the original static function <BlenderGlTF.set_convert_functions> of the glTF importer will stored in the attribute below
        self._set_convert_functions = None
    
    def prepare(self, manager):
        self.collection = createCollection(self.threedTilesName)
        
        self.centerCoords = manager.fromGeographic(manager.centerLat, manager.centerLon, 0.)
        
        if "io_scene_gltf2" in sys.modules:
            self.patchGltfImporter()
    
    def finalize(self, manager):
        if not self.importedObjects:
            self.collection = None
            return
        
        #
        # tranformation matrix
        #
        centerCoords = self.centerCoords
        # lat = radians(manager.centerLat - 90.) # gives incorrect result for the expression below
        lat = atan(centerCoords[2]/sqrt(centerCoords[0]*centerCoords[0] + centerCoords[1]*centerCoords[1]))
        # Bring the mesh to the north pole with rotations around Z and X axes,
        # then move it to the center of the coordinate reference system
        matrix = Matrix.Rotation(lat-pi/2., 4, 'X') @ Matrix.Rotation(radians(-90. - manager.centerLon), 4, 'Z')
        
        locationsAtNorthPole = [(matrix @ obj.location) for obj in self.importedObjects]
        
        # find the lowest Z-coordinate if <self.calculateHeightOffset>
        heightOffset = min(location[2] for location in locationsAtNorthPole)\
            if self.calculateHeightOffset else\
            self.heightOffset
        if self.calculateHeightOffset:
            self.heightOffset = heightOffset
        
        bpy.ops.object.select_all(action='DESELECT')
        
        if self.join3dTilesObjects:
            for obj in self.importedObjects:
                obj.select_set(True)
            self.joinObjects()
            
            # set the origin of the resulting Blender object at <centerCoords>
            _cursorLocation = bpy.context.scene.cursor.location.copy()
            bpy.context.scene.cursor.location = (0., 0., 0.) if self.gltfImporterPatched else centerCoords
            bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
            bpy.context.scene.cursor.location = _cursorLocation
            
            joinedObject = self.importedObjects[-1]
            #location = locationsAtNorthPole[-1]
            #location[2] -= heightOffset
            joinedObject.matrix_local = matrix#Matrix.Translation(location) @ matrix
        else:
            for obj, location in zip(self.importedObjects, locationsAtNorthPole):
                location[2] -= heightOffset
                obj.matrix_local = Matrix.Translation(location) @ matrix
                obj.select_set(True)
        
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False) 
        bpy.ops.object.select_all(action='DESELECT')
        
        bpy.context.scene.blosm.copyright = "; ".join(
            entry[0] for entry in reversed(
                sorted(
                    self.copyrightHolders.items(), key=itemgetter(1)
                )
            )
        )
        
        numImportedTiles = len(self.importedObjects)
        
        self.importedObjects.clear()
        self.collection = None
        
        if self.gltfImporterPatched:
            from io_scene_gltf2.io.imp.gltf2_io_gltf import glTFImporter
            from io_scene_gltf2.blender.imp.gltf2_blender_gltf import BlenderGlTF
            
            # clean everything up after patching
            BlenderGlTF.set_convert_functions = self._set_convert_functions
            self._set_convert_functions = None
            delattr(glTFImporter, "_offset")
        
        return numImportedTiles
    
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
        
        importedObject = context.object
        self.collection.objects.link(importedObject)
        self.importedObjects.append(importedObject)
        context.scene.collection.objects.unlink(importedObject)
    
    def processCopyrightInfo(self, info):
        for copyrightHolder in info.split(';'):
            copyrightHolder = copyrightHolder.strip()
            if not copyrightHolder in self.copyrightHolders:
                self.copyrightHolders[copyrightHolder] = 0
            self.copyrightHolders[copyrightHolder] += 1
    
    def joinObjects(self):
        bpy.ops.object.join()
        joinedObject = self.importedObjects[-1]
        joinedObject.name = self.threedTilesName
        bpy.context.view_layer.objects.active = joinedObject
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.remove_doubles()
        bpy.ops.object.mode_set(mode='OBJECT')
    
    def patchGltfImporter(self):
        bv = bpy.app.version
        if bv[0] == 4 and bv[1] == 0:
            from .gltf_patch import set_convert_functions_4_0
            from io_scene_gltf2.io.imp.gltf2_io_gltf import glTFImporter
            from io_scene_gltf2.blender.imp.gltf2_blender_gltf import BlenderGlTF
            
            glTFImporter._offset = self.centerCoords
            self._set_convert_functions = BlenderGlTF.set_convert_functions
            BlenderGlTF.set_convert_functions = set_convert_functions_4_0
            
            self.gltfImporterPatched = True