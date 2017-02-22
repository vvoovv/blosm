import bpy, bmesh
import math
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from util import zAxis
from util.blender import createMeshObject, getBmesh, setBmesh, pointNormalUpward

direction = -zAxis # downwards


class Terrain:
    
    # z-coordinate of the bottom of the envelope
    envZb = -1.
    # offset for the SHRINKWRAP modifier used to project flat meshes on the terrain
    swOffset = 1.
    
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
        self.envelope = None
    
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
    
    def createEnvelope(self):
        terrain = self.terrain
        envelope = createMeshObject("%s_envelope" % terrain.name, (0., 0., self.envZb), terrain.data.copy())
        self.envelope = envelope
        # flatten the terrain envelope
        envelope.scale[2] = 0.
        envelope.select = True 
        bpy.context.scene.objects.active = envelope
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        bm = getBmesh(envelope)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0001)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bmesh.ops.dissolve_limit(bm, angle_limit=math.radians(0.1), verts=bm.verts, edges=bm.edges)
        for f in bm.faces:
            f.smooth = False
            pointNormalUpward(f)
        verts = tuple(
            v for v in bmesh.ops.extrude_face_region(bm, geom=bm.faces)["geom"]
                if isinstance(v, bmesh.types.BMVert)
        )
        envZt = max( terrain.bound_box, key=lambda f:(terrain.matrix_world*Vector(f))[2] )[2]
        bmesh.ops.translate(bm, verts=verts, vec=(0., 0., -self.envZb + envZt + 2*self.swOffset))
        setBmesh(envelope, bm)