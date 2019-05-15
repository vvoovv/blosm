"""
This file is part of blender-osm (OpenStreetMap importer for Blender).
Copyright (C) 2014-2018 Vladimir Elistratov
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

import math
import bpy, bmesh
from mathutils import Vector
from renderer import Renderer
from util.blender import createCollection, createEmptyObject, getBmesh, setBmesh

_isBlender280 = bpy.app.version[1] >= 80


class Layer:
    
    def __init__(self, layerId, app):
        self.app = app
        self.id = layerId
        self.singleObject = app.singleObject
        # instance of BMesh
        self.bm = None
        # Blender object
        self.obj = None
        # Blender collection for the layer objects; used only if <not layer.singleObject>
        self.collection = None
        # Blender parent object
        self.parent = None
        # does the layer represents an area (natural or landuse)?
        self.area = False

    @property
    def name(self):
        return "%s_%s" % (Renderer.name, self.id)
    
    def getCollection(self, parentCollection):
        # The method is called currently in the single place of the code:
        # in <Renderer.preRender(..)> if (not layer.singleObject)
        collection = self.collection
        if not collection:
            collection = createCollection(self.name, parent=parentCollection)
            self.collection = collection
        return collection

    def getParent(self, collection=None):
        # The method is called currently in the single place of the code:
        # in <Renderer.preRender(..)> if (not layer.singleObject)
        parent = self.parent
        if not self.parent:
            parent = createEmptyObject(
                self.name,
                self.parentLocation.copy(),
                collection = collection,
                empty_draw_size=0.01
            )
            if not _isBlender280:
                parent.parent = Renderer.parent
            self.parent = parent
        return parent


class MeshLayer(Layer):
    
    def __init__(self, layerId, app):
        super().__init__(layerId, app)
        # does the layer represents an area (natural or landuse)?
        self.area = True
    
    def init(self):
        # The code in this method has been moved from the constructor,
        # since a flat terrain can be set only after OSM parsing if the flat terrain is needed
        # to plant forests. If OSM import is performed from a local file, the area extent becomes
        # known only after OSM parsing, so the flat terrain (if needed) can be created only after
        # OSM parsing
        app = self.app
        layerId = self.id
        
        terrain = app.terrain
        hasTerrain = bool(terrain)
        # apply Blender modifiers (BOOLEAND AND SHRINKWRAP) if a terrain is set
        self.modifiers = hasTerrain
        # slice flat mesh to project it on the terrain correctly
        self.sliceMesh = hasTerrain and app.subdivide
        # set layer offsets <self.location>, <self.meshZ> and <self.parentLocation>
        # <self.location> is used for a Blender object
        # <self.meshZ> is used for vertices of a BMesh
        # <self.parentLocation> is used for an EMPTY Blender object serving
        # as a parent for Blender objects of the layer
        self.parentLocation = None
        meshZ = 0.
        _z = app.layerOffsets.get(layerId, 0.)
        if hasTerrain:
            # here we have <self.singleObject is True>
            location = Vector((0., 0., terrain.maxZ + terrain.layerOffset))
            self.swOffset = _z or self.getDefaultSwOffset(app)
            if not self.singleObject:
                # it's the only case when <self.parentLocation> is needed if a terrain is set
                self.parentLocation = Vector((0., 0., _z))
        elif self.singleObject:
            location = Vector((0., 0., _z or self.getDefaultZ(app)))
        elif not self.singleObject:
            location = None
            # it's the only case when <self.parentLocation> is needed if a terrain isn't set
            self.parentLocation = Vector((0., 0., _z or self.getDefaultZ(app)))
        self.location = location
        self.meshZ = meshZ
    
    def getDefaultZ(self, app):
        return 0.
    
    def getDefaultSwOffset(self, app):
        return app.swOffset
    
    def prepare(self, instance):
        instance.bm = getBmesh(instance.obj)
        instance.materialIndices = {}
    
    def finalizeBlenderObject(self, obj):
        """
        Slice Blender MESH object, add modifiers
        """
        app = self.app
        terrain = app.terrain
        if terrain and self.sliceMesh:
            self.slice(obj, terrain, app)
        if self.modifiers:
            self.addBoolenModifier(obj, terrain.envelope)
            self.addShrinkwrapModifier(obj, terrain.terrain, self.swOffset)
    
    def addShrinkwrapModifier(self, obj, target, offset):
        m = obj.modifiers.new(name="Shrinkwrap", type='SHRINKWRAP')
        m.wrap_method = "PROJECT"
        m.use_positive_direction = False
        m.use_negative_direction = True
        m.use_project_z = True
        m.target = target
        m.offset = offset
    
    def addBoolenModifier(self, obj, operand):
        m = obj.modifiers.new(name="Boolean", type='BOOLEAN')
        m.operation = "INTERSECT"
        m.object = operand
    
    def slice(self, obj, terrain, app):
        sliceSize = app.subdivisionSize
        bm = getBmesh(obj)
        
        def _slice(index, plane_no, terrainMin, terrainMax):
            # min and max value along the axis defined by <index>
            # 1) terrain
            # a simple conversion from the world coordinate system to the local one
            terrainMin = terrainMin - obj.location[index]
            terrainMax = terrainMax - obj.location[index]
            # 2) <bm>, i.e. Blender mesh
            minValue = min(obj.bound_box, key = lambda v: v[index])[index]
            maxValue = max(obj.bound_box, key = lambda v: v[index])[index]
            
            # cut everything off outside the terrain bounding box
            if minValue < terrainMin:
                minValue = terrainMin
                bmesh.ops.bisect_plane(
                    bm,
                    geom=bm.verts[:]+bm.edges[:]+bm.faces[:],
                    plane_co=(0., minValue, 0.) if index else (minValue, 0., 0.),
                    plane_no=plane_no,
                    clear_inner=True
                )
            
            if maxValue > terrainMax:
                maxValue = terrainMax
                bmesh.ops.bisect_plane(
                    bm,
                    geom=bm.verts[:]+bm.edges[:]+bm.faces[:],
                    plane_co=(0., maxValue, 0.) if index else (maxValue, 0., 0.),
                    plane_no=plane_no,
                    clear_outer=True
                )
            
            # now cut the slices
            width = maxValue - minValue
            if width > sliceSize:
                numSlices = math.ceil(width/sliceSize)
                _sliceSize = width/numSlices
                coord = minValue
                sliceIndex = 1
                while sliceIndex < numSlices:
                    coord += _sliceSize
                    bmesh.ops.bisect_plane(
                        bm,
                        geom=bm.verts[:]+bm.edges[:]+bm.faces[:],
                        plane_co=(0., coord, 0.) if index else (coord, 0., 0.),
                        plane_no=plane_no
                    )
                    sliceIndex += 1
        
        _slice(0, (1., 0., 0.), terrain.minX, terrain.maxX)
        _slice(1, (0., 1., 0.), terrain.minY, terrain.maxY)
        setBmesh(obj, bm)