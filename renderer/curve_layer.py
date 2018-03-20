import os
import bpy
from .layer import MeshLayer
from util.blender import appendObjectsFromFile, createDiffuseMaterial


class CurveLayer(MeshLayer):
    
    # Blender layer index to place a way profile
    profileLayerIndex = 1
    
    # Blender file with way profiles
    assetFile = "way_profiles.blend"
    
    def __init__(self, layerId, app):
        super().__init__(layerId, app)
        self.curve = None
        self.assetPath = os.path.join(app.assetPath, self.assetFile)

    def prepare(self, instance):
        instance.curve = instance.obj.data

    def finalizeBlenderObject(self, obj):
        """
        Slice Blender MESH object, add modifiers
        """
        # set a bevel object for the curve
        curve = obj.data
        # the name of the bevel object
        bevelName = "profile_%s" % self.id
        bevelObj = bpy.data.objects.get(bevelName)
        if not (bevelObj and bevelObj.type == 'CURVE'):
            bevelObj = appendObjectsFromFile(self.assetPath, bevelName)[0]
            # move <obj> to the Blender layer with the index <self.profileLayerIndex>
            bevelObj.layers[self.profileLayerIndex] = True
            bevelObj.layers[0] = False
        curve.bevel_object = bevelObj
        # set a material
        # the material name is simply <id> of the layer
        name = self.id
        material = bpy.data.materials.get(name)
        curve.materials.append(
            material or createDiffuseMaterial(name, self.app.colors.get(name, self.app.defaultColor))
        )
        
        terrain = self.app.terrain
        if self.modifiers:
            if not terrain.envelope:
                terrain.createEnvelope()
            self.addShrinkwrapModifier(obj, terrain.terrain, self.swOffset)