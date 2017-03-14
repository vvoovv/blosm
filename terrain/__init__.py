"""
This file is part of blender-osm (OpenStreetMap importer for Blender).
Copyright (C) 2014-2017 Vladimir Elistratov
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

import bpy, bmesh
import math
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
from util import zAxis, zeroVector
from util.blender import createMeshObject, getBmesh, setBmesh, pointNormalUpward

direction = -zAxis # downwards


class Terrain:
    
    # extra offset for the top part of the terrain envelope
    envelopeOffset = 25.
    # Extra offset for z-coordinate of flat layers to be projected on the terrain;
    # used only if a terrain is set. It must be less than <self.envelopeOffset>
    layerOffset = 20.
    # extra offset for the location used to project objects on the terrain
    projectOffset = 1.
    # used for <thickness> parameter of bmesh.ops.inset_region(..)
    envelopeInset = 0.5
    
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
    
    def init(self):
        terrain = self.terrain
        
        # transform <terrain.bound_box> to the world system of coordinates
        bound_box = tuple(terrain.matrix_world*Vector(v) for v in terrain.bound_box)
        self.minZ = min(bound_box, key = lambda v: v[2])[2]
        self.maxZ = max(bound_box, key = lambda v: v[2])[2]
        
        self.minX = min(bound_box, key = lambda v: v[0])[0]
        self.maxX = max(bound_box, key = lambda v: v[0])[0]
        self.minY = min(bound_box, key = lambda v: v[1])[1]
        self.maxY = max(bound_box, key = lambda v: v[1])[1]
        
        self.projectLocation = self.maxZ + self.projectOffset
        
        # An attribute to store the original location of the terrain Blender object,
        # if the terrain isn't located at the origin of the world system of coordinates
        self.location = None
        if terrain.location.length_squared:
            self.location = terrain.location.copy()
            # set origin of the terrain Blender object to zero
            self.setOrigin(zeroVector())
            # execute the line below to get correct results (i.e. update transformation matrix)
            bpy.context.scene.update()
        bm = bmesh.new()
        bm.from_mesh(terrain.data)
        self.bvhTree = BVHTree.FromBMesh(bm)
        # <bm> is no longer needed
        bm.free()
    
    def cleanup(self):
        if not self.location is None:
            self.setOrigin(self.location)
        self.terrain = None
        self.bvhTree = None
    
    def project(self, coords):
        # Cast a ray from the point with horizontal coords equal to <coords> and
        # z = 10000. in the direction of <direction>
        return self.bvhTree.ray_cast((coords[0], coords[1], self.projectLocation), direction)[0]
    
    def setOrigin(self, origin):
        terrain = self.terrain
        # see http://blender.stackexchange.com/questions/35825/changing-object-origin-to-arbitrary-point-without-origin-set
        offset = origin - terrain.matrix_world.translation
        terrain.data.transform(Matrix.Translation(-offset))
        terrain.matrix_world.translation += offset
    
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
        name = "%s_envelope" % terrain.name
        if name in bpy.data.objects:
            # use existing Blender object
            envelope = bpy.data.objects[name]
            # delete modifiers
            for m in reversed(envelope.modifiers):
                envelope.modifiers.remove(m)
        else:
            # create new envelope for the terrain
            envelope = createMeshObject(name, (0., 0., self.minZ), terrain.data.copy())
            # flatten the terrain envelope
            envelope.scale[2] = 0.
            envelope.select = True 
            bpy.context.scene.objects.active = envelope
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='DESELECT')
            bpy.ops.mesh.select_mode(type='FACE')
            bpy.ops.object.mode_set(mode='OBJECT')
            for p in envelope.data.polygons:
                if p.normal[2] < 0.:
                    p.select = True
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.delete(type='FACE')
            bpy.ops.mesh.select_mode(type='VERT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.remove_doubles()
            bpy.ops.mesh.region_to_loop()
            bpy.ops.mesh.select_all(action='INVERT')
            bpy.ops.mesh.dissolve_verts(use_face_split=False, use_boundary_tear=False)
            bpy.ops.mesh.select_all(action='SELECT')
            #bpy.ops.mesh.dissolve_limited(angle_limit=math.radians(0.1))
            bpy.ops.object.mode_set(mode='OBJECT')
            bm = getBmesh(envelope)
            for f in bm.faces:
                f.smooth = False
                # ensure all normals point upward
                pointNormalUpward(f)
            # inset faces to avoid weird results of the BOOLEAN modifier
            insetFaces = bmesh.ops.inset_region(bm, faces=bm.faces,
                use_boundary=True, use_even_offset=True, use_interpolate=True,
                use_relative_offset=False, use_edge_rail=False, use_outset=False,
                thickness=self.envelopeInset, depth=0.
            )['faces']
            bmesh.ops.delete(bm, geom=insetFaces, context=5)
            setBmesh(envelope, bm)
        
        self.envelope = envelope
        envelope.select = False
        envelope.hide_render = True
        # hide <envelope> after all Blender operator
        envelope.hide = True
        # SOLIDIFY modifier instead of BMesh extrude operator
        m = envelope.modifiers.new(name="Solidify", type='SOLIDIFY')
        m.offset = 1.
        m.thickness = self.maxZ - self.minZ + self.envelopeOffset