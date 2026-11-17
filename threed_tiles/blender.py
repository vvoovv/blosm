from os.path import basename
from math import radians, atan, sqrt, pi, isfinite
import sys

import bpy
from mathutils import Vector, Matrix

from ..util.blender import createCollection

from .gltf_import import import_gltf
from .attribution import record_import


def find_layer_collection(layer_col: bpy.types.LayerCollection, target_col: bpy.types.Collection):
    """
    Recursively searches for the LayerCollection that corresponds to the given <target_col> of type <bpy.types.Collection>
    within the hierarchy starting from <layer_col> of type <bpy.types.LayerCollection>
    Returns the found LayerCollection or None if not found.
    """
    if layer_col.collection == target_col:
        return layer_col
    for child in layer_col.children:
        found = find_layer_collection(child, target_col)
        if found:
            return found
    return None


class BlenderRenderer:
    
    def __init__(self, threedTilesName, join3dTilesObjects, instanceName, report=None, *, recordAttribution=True):
        self.threedTilesName = threedTilesName
        self.join3dTilesObjects = join3dTilesObjects
        self.instanceName = instanceName
        self.report = report
        
        self.calculateHeightOffset = False
        self.heightOffset = 0.
        
        self.copyrightHolders = {}
        self.providerCredits = []
        self.recordAttribution = recordAttribution
        
    
    def prepare(self, manager):
        self.num_imported_tiles = 0

        self.collection = createCollection(self.threedTilesName)
        # <self.collection_import> is used to import the objects first, then they are moved to <self.collection>.
        # The reason for this is to isolatede the imported objects from other objects in the scene during the import process.
        self.collection_import = createCollection(self.threedTilesName + "_import")
        # LayerCollection corresponding to <self.collection_import>
        self.layer_collection_import = find_layer_collection(
            bpy.context.view_layer.layer_collection,
            self.collection_import
        )
        # set the active LayerCollection to <self.layer_collection_import>
        bpy.context.view_layer.active_layer_collection = self.layer_collection_import
        
        self.centerCoords = manager.fromGeographic(manager.centerLat, manager.centerLon, 0.)
        
    
    def finalize(self, manager):
        bpy.data.collections.remove(self.collection_import)
        self.collection_import = None
        self.layer_collection_import = None

        if not self.collection.objects:
            bpy.data.collections.remove(self.collection)
            self.collection = None

            return

        if not self.join3dTilesObjects and self._finalize_direct(manager):
            if self.recordAttribution and self.num_imported_tiles:
                self._attribution()
            self.collection = None
            return
        
        bpy.ops.object.select_all(action='DESELECT')

        # check if there are any parent-child relationships among the imported objects
        if sum(1 for obj in self.collection.objects if obj.parent):
            # Out task is to remove all parent-child relationships among the imported objects,
            # i.e. flatten the hierarchy.

            # Build a dictionary Object Name --> Object for the imported Mesh objects
            mesh_objects = {obj.name: obj for obj in self.collection.objects if obj.type == 'MESH'}
            while True:
                num_processed_objects = 0
                for obj in list(mesh_objects.values()):
                    # Only bottom-level Mesh objects with a parent (i.e at the very bottom of a parent-child hierarchy)
                    # are processed in each iteration
                    if obj.parent and not obj.children:
                        # get final world transform (includes all parents)
                        mw = obj.matrix_world.copy()
                        # clear parent
                        obj.parent = None
                        # apply the final world transform to the object
                        obj.matrix_world = mw
                        # remove <obj> from the dictionary <mesh_objects>
                        del mesh_objects[obj.name]
                        num_processed_objects += 1
                if num_processed_objects == 0 or not mesh_objects:
                    # no more Mesh objects with a parent and without children are left
                    break
            # remove imported Blender Empty objects
            for obj in [obj for obj in self.collection.objects if obj.type == 'EMPTY']:
                bpy.data.objects.remove(obj)
        
        # select all imported Mesh objects:
        for obj in self.collection.objects:
            obj.select_set(True)
        
        # apply possible rotation
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

        #
        # tranformation matrix
        #
        centerCoords = self.centerCoords
        # lat = radians(manager.centerLat - 90.) # gives incorrect result for the expression below
        lat = atan(centerCoords[2]/sqrt(centerCoords[0]*centerCoords[0] + centerCoords[1]*centerCoords[1]))
        # Rotate the mesh, so it will point to the north pole. The rotations are around Z and X axes
        matrix = Matrix.Rotation(lat-pi/2., 4, 'X') @ Matrix.Rotation(radians(-90. - manager.centerLon), 4, 'Z')
        
        locationsAfterRotation = [(matrix @ obj.location) for obj in self.collection.objects]
        
        # find the lowest Z-coordinate if <self.calculateHeightOffset>
        heightOffset = min(location[2] for location in locationsAfterRotation)\
            if self.calculateHeightOffset else\
            self.heightOffset
        if self.calculateHeightOffset:
            self.heightOffset = heightOffset
        

        if self.join3dTilesObjects:
            self.joinObjects()
            
            # set the origin of the resulting Blender object at <centerCoords>
            _cursorLocation = bpy.context.scene.cursor.location.copy()
            bpy.context.scene.cursor.location = (0., 0., 0.)
            bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
            bpy.context.scene.cursor.location = _cursorLocation
            
            #location = locationsAfterRotation[-1]
            #location[2] -= heightOffset
            # set the matrix_local of the resulting Blender object
            self.collection.objects[0].matrix_local = matrix#Matrix.Translation(location) @ matrix
        else:
            for obj, location in zip(self.collection.objects, locationsAfterRotation):
                obj.matrix_local = Matrix.Translation(location) @ matrix
                obj.select_set(True)
        
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
        bpy.ops.object.select_all(action='DESELECT')
        
        if self.recordAttribution and self.num_imported_tiles:
            self._attribution()

        self.collection = None

    def _finalize_direct(self, manager):
        """Position ordinary static tiles without selection/transform operators.

        Return False before changing anything when the original operator path
        is needed. Errors after transformation starts propagate to the existing
        tile rollback; never retry a partly transformed tile with operators.
        """
        objects = tuple(self.collection.objects)
        imported = set(objects)
        meshes = tuple(obj for obj in objects if obj.type == 'MESH')
        parents = any(obj.parent for obj in objects)
        if not meshes:
            return False
        if parents:
            # RTC/tile transforms can change a parent's matrix after the glTF
            # update. Match the old first operator's refresh before flattening,
            # then check eligibility after any dependency-graph handlers ran.
            bpy.context.view_layer.update()
        # Restrict the fast path to static, private mesh data and simple
        # hierarchies. In particular, mesh -> empty chains retain the legacy
        # flattening order, and no outside child may inherit a changed basis.
        for obj in objects:
            if (obj.type not in {'MESH', 'EMPTY'} or
                    (obj.type == 'EMPTY' and not parents) or
                    obj.parent_type != 'OBJECT' or
                    (obj.parent is not None and obj.parent not in imported) or
                    any(child not in imported or (obj.type == 'MESH' and child.type != 'MESH')
                        for child in obj.children) or
                    obj.animation_data is not None or obj.constraints or obj.modifiers or
                    obj.library is not None or obj.hide_viewport or obj.hide_select or obj.hide_get() or
                    obj.rotation_mode == 'AXIS_ANGLE' or
                    tuple(obj.delta_location) != (0., 0., 0.) or
                    tuple(obj.delta_rotation_euler) != (0., 0., 0.) or
                    tuple(obj.delta_rotation_quaternion) != (1., 0., 0., 0.) or
                    tuple(obj.delta_scale) != (1., 1., 1.)):
                return False
            if obj.type == 'MESH' and (
                    obj.mode != 'OBJECT' or obj.data.users != 1 or
                    obj.data.library is not None or obj.data.shape_keys is not None or
                    obj.data.animation_data is not None):
                return False
        matrices = tuple((obj, obj.matrix_world.copy()) for obj in meshes)
        for obj, matrix in matrices:
            axes = matrix.to_3x3()
            if (not all(isfinite(value) for row in matrix for value in row) or
                    any(abs(axis.length - 1.) > 1e-5 for axis in axes.col) or
                    any(abs(axes.col[i].dot(axes.col[j])) > 1e-5
                        for i, j in ((0, 1), (0, 2), (1, 2))) or
                    axes.determinant() < 0.):
                return False
        center = self.centerCoords
        latitude = atan(center[2]/sqrt(center[0]*center[0] + center[1]*center[1]))
        rotation = Matrix.Rotation(latitude-pi/2., 4, 'X') @ Matrix.Rotation(
            radians(-90. - manager.centerLon), 4, 'Z')
        locations = []
        for obj, matrix in matrices:
            if obj.parent is not None:
                obj.parent = None
                obj.matrix_world = matrix
            self._apply_rotation(obj)
            location = rotation @ obj.location
            locations.append(location)
            obj.matrix_local = Matrix.Translation(location) @ rotation
            self._apply_rotation(obj)
        for obj in objects:
            if obj.type == 'EMPTY':
                bpy.data.objects.remove(obj)
        if self.calculateHeightOffset:
            self.heightOffset = min(location.z for location in locations)
        # Match the blocking path too; progressive preserve_context restores
        # the user's selection after this synchronous tile operation.
        for obj in bpy.context.selected_objects:
            obj.select_set(False)
        return True

    @staticmethod
    def _apply_rotation(obj):
        # Match Apply Rotation's scale correction, including tiny scale values
        # introduced by matrix decomposition. Leave location and scale intact.
        rotation = (obj.rotation_quaternion.to_matrix() if obj.rotation_mode == 'QUATERNION'
                    else obj.rotation_euler.to_matrix())
        scale = Matrix.Diagonal(obj.scale)
        obj.data.transform((scale.inverted() @ rotation @ scale).to_4x4())
        obj.rotation_euler = (0., 0., 0.)
        obj.rotation_quaternion = (1., 0., 0., 0.)
        obj.rotation_axis_angle = (0., 0., 1., 0.)

    def _attribution(self):
        record_import(bpy.context.scene.blosm, self.collection,
                      self.copyrightHolders, self.providerCredits)
    
    def renderGlb(self, manager, uri, path, transformMatrix, *, content=None):
        # The filename supplies the source identity; the tile stays in memory.
        result = import_gltf(basename(path), content=manager.download(uri) if content is None else content,
                             report=self.report, coordinate_offset=self.centerCoords)
        if self.collection_import.objects or self.collection_import.children:
            self.finalize_glb_import(transformMatrix)
            self.processCopyrightInfo(result.copyright)

    def renderB3dm(self, manager, uri, path, transformMatrix, *, content=None):
        import numpy
        from .py3dtiles.tileset.content.tile_content_reader import read_array

        fileContent = manager.download(uri) if content is None else content
        b3dmContent = read_array(numpy.frombuffer(fileContent, dtype=numpy.uint8))
        if b3dmContent is None or b3dmContent.header is None or not hasattr(b3dmContent.body, 'gltf'):
            raise ValueError("The file doesn't contain valid B3DM data.")

        # Keep B3DM conversion in memory. RTC is separate from the embedded GLB
        # and must survive conversion.
        rtc_center = b3dmContent.body.feature_table.header.data.get("RTC_CENTER")
        hasRtc = rtc_center is not None
        result = import_gltf(basename(path), content=b3dmContent.body.gltf.to_array(),
                             report=self.report, coordinate_offset=(0., 0., 0.) if hasRtc else self.centerCoords)

        if self.collection_import.objects or self.collection_import.children:
            if hasRtc:
                self.process_rtc(rtc_center)
            self.finalize_glb_import(transformMatrix)
            self.processCopyrightInfo(result.copyright)

    def processCopyrightInfo(self, info):
        for copyrightHolder in info.split(';'):
            copyrightHolder = copyrightHolder.strip()
            if not copyrightHolder:
                continue
            if not copyrightHolder in self.copyrightHolders:
                self.copyrightHolders[copyrightHolder] = 0
            self.copyrightHolders[copyrightHolder] += 1
    
    def process_rtc(self, rtc_center):
        rtc_center = Vector(rtc_center)

        # Find Blender collection where the imported objects are located
        collection = self.collection_import
        if self.collection_import.children:
            # Find the first non-excluded child collection using the property <exclude> of <LayerCollection>
            # Why only the first? The other appear to be a phatom collection with the property <exclude> set to True.
            collection = next((c for c in self.layer_collection_import.children if not c.exclude)).collection
        
        for obj in collection.objects:
            # only top level objects are adjusted
            if not obj.parent:
                # Both lines below are identical. However Blender does not immediately
                # refresh <obj.matrix_world> after changing <obj.location> and
                # a call of <bpy.context.view_layer.update()> would be required to force this refresh,
                # so the subsequent updates of <obj.matrix_world> would work correctly.
                # Using matrix operations instead does not require calling <bpy.context.view_layer.update()>
                # obj.location += rtc_center - self.centerCoords
                obj.matrix_world = Matrix.Translation(rtc_center - self.centerCoords) @ obj.matrix_world
                #
                # See also doc/Transforms.md
                #
    
    def finalize_glb_import(self, transformMatrix):

        # Find Blender collection where the imported objects are located
        collection = self.collection_import
        if self.collection_import.children:
            # Find the first non-excluded child collection using the property <exclude> of <LayerCollection>
            # Why only the first? The other appear to be a phatom collection with the property <exclude> set to True.
            collection = next((c for c in self.layer_collection_import.children if not c.exclude)).collection

        # Move all objects from <collection> to the main collection
        for obj in collection.objects:
            collection.objects.unlink(obj)
            self.collection.objects.link(obj)
            if transformMatrix:
                self.apply_transform_matrix(obj, transformMatrix)
        
        if self.collection_import.children:
            # Remove all child collections from <self.collection_import>
            for collection in self.collection_import.children:
                bpy.data.collections.remove(collection)
        
        self.num_imported_tiles += 1
    
    def apply_transform_matrix(self, obj, transformMatrix):
        u = 1.0 / bpy.context.scene.unit_settings.scale_length

        center = u * self.centerCoords
        m = transformMatrix.copy()
        m.translation *= u
        
        # See doc/Transforms.md
        obj.matrix_world = Matrix.Translation(-center) @ m @ Matrix.Translation(center) @ obj.matrix_world


    def joinObjects(self):
        # If a Blender Empty object was imported and it became the active object in the course of the processing,
        # after deleting it, Blender may not have any active object, which is required by <bpy.ops.object.join()>.
        # Therefore, we set the active object to the first object in the collection.
        bpy.context.view_layer.objects.active = self.collection.objects[0]

        if len(self.collection.objects) > 1:
            bpy.ops.object.join()
        
        joinedObject = self.collection.objects[0]
        joinedObject.name = self.threedTilesName

        # Removing doubles produces odd results in some cases, so it is disabled for now.
        #bpy.ops.object.mode_set(mode='EDIT')
        #bpy.ops.mesh.select_all(action='SELECT')
        #bpy.ops.mesh.remove_doubles()
        #bpy.ops.object.mode_set(mode='OBJECT')
