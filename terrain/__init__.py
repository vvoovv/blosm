import bmesh
import math
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
    
    @staticmethod
    def getHgtIntervals(x1, x2):
        """
        Split (x1, x2) into .hgt intervals. Examples:
        (31.2, 32.7) => [ (31.2, 32), (32, 32.7) ]
        (31.2, 32) => [ (31.2, 32) ]
        """
        _x1 = x1
        intervals = []
        while True:
            _x2 = math.floor(_x1 + 1)
            if (_x2>=x2):
                intervals.append((_x1, x2))
                break
            else:
                intervals.append((_x1, _x2))
                _x1 = _x2
        return intervals
    
    @staticmethod
    def getHgtFileName(lat, lon):
        prefixLat = "N" if lat>= 0 else "S"
        prefixLon = "E" if lon>= 0 else "W"
        return "{}{:02d}{}{:03d}.hgt.gz".format(prefixLat, abs(lat), prefixLon, abs(lon))