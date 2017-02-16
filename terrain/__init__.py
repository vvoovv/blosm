import bmesh
from mathutils.bvhtree import BVHTree
from util import zAxis

direction = -zAxis # downwards


class Terrain:
    
    def __init__(self, context):
        """
        The method performs some initialization, namely checks if a Blender object for the terrain is set.
        """
        terrain = None
        name = context.scene.blender_osm.terrainObject
        if name:
            terrain = context.scene.objects.get(name)
            if not terrain:
                raise Exception("Blender object %s for the terrain doesn't exist. " % name)
            if terrain.type != "MESH":
                raise Exception("Blender object %s for the terrain doesn't exist. " % name)
        self.terrain = terrain
    
    def initBvhTree(self):
        bm = bmesh.new()
        bm.from_mesh(self.terrain.data)
        self.bvhTree = BVHTree.FromBMesh(bm)
        # <bm> is no longer needed
        bm.free()
    
    def project(self, coords):
        # Cast a ray from the point with horizontal coords equal to <coords> and
        # z = 10000. in the direction of <direction>
        return self.bvhTree.ray_cast((coords[0], coords[1], 10000.), direction)[0]