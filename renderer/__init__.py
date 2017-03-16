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

import os, math, bpy, bmesh
from util import zeroVector
from util.blender import createEmptyObject, createDiffuseMaterial, pointNormalUpward,\
    getBmesh, setBmesh
from util.osm import assignTags


class Renderer:
    
    # types of data
    linestring = (1,)
    multilinestring = (1,)
    polygon = (1,)
    multipolygon = (1,)
    
    parent = None
    name = None
    
    def __init__(self, app):
        self.app = app
        # offset for a Blender object created if <layer.singleObject is False>
        self.offset = None
        # offset if a terrain is set (used instead of <self.offset>)
        self.offsetZ = None
    
    @classmethod
    def begin(self, app):
        self.name = os.path.basename(app.osmFilepath)
        
        if app.layered or not app.singleObject:
            self.parent = createEmptyObject(
                self.name,
                zeroVector(),
                empty_draw_size=0.01
            )
        if app.singleObject and not app.layered:
            self.bm = bmesh.new()
            self.obj = self.createBlenderObject(self.name, None, None)
            # cache material indices in <self.materialIndices>
            self.materialIndices = {}

        # store here Blender object that are to be joined
        self.toJoin = {}
    
    def preRender(self, element, layer=None):
        app = self.app
        layer = element.l if layer is None else layer
        self.layer = layer
        
        if layer.singleObject:
            if layer.layered:
                bm = layer.bm
                obj = layer.obj
                materialIndices = layer.materialIndices
                if not bm:
                    bm = bmesh.new()
                    layer.bm = bm
                    obj = self.createBlenderObject(
                        layer.name,
                        layer.location,
                        self.parent
                    )
                    layer.obj = obj
                    materialIndices = {}
                    layer.materialIndices = materialIndices
                self.bm = bm
                self.obj = obj
                self.materialIndices = materialIndices
            else:
                self.bm = Renderer.bm
                self.obj = Renderer.obj
                self.materialIndices = Renderer.materialIndices
        else:
            self.obj = self.createBlenderObject(
                self.getName(element),
                self.offsetZ if self.offsetZ else (self.offset if self.offset else layer.location),
                layer.getParent() if app.layered else Renderer.parent
            )
            self.bm = bmesh.new()
            self.materialIndices = {}
    
    def renderLineString(self, element, data):
        pass
    
    def renderPolygon(self, element, data):
        pass
    
    def renderMultiPolygon(self, element, data):
        pass
    
    def postRender(self, element):
        if not self.layer.singleObject:
            # finalize BMesh
            self.bm.to_mesh(self.obj.data)
            # assign OSM tags to the blender object
            assignTags(self.obj, element.tags)
    
    @classmethod
    def end(self, app):
        terrain = app.terrain
        for layer in app.layers:
            if layer.obj:
                setBmesh(layer.obj, layer.bm)
        if app.singleObject and not app.layered:
            # finalize BMesh
            setBmesh(self.obj, self.bm)
        
        bpy.context.scene.update()
        # Go through <app.layers> once again after <bpy.context.scene.update()>
        # to get correct results for <layer.obj.bound_box>
        for layer in app.layers:
            if layer.obj:
                if terrain and layer.sliceMesh:
                    self.slice(layer.obj, terrain, app)
                if layer.swModifier:
                    if not terrain.envelope:
                        terrain.createEnvelope()
                    self.addBoolenModifier(layer.obj, terrain.envelope)
                    self.addShrinkwrapModifier(layer.obj, terrain.terrain, layer.swOffset)
        
        bpy.ops.object.select_all(action="DESELECT")
        self.join()
        
        if terrain:
            terrain.cleanup()
    
    @classmethod
    def join(self):
        """
        Join Blender object collect during rendering
        """
        join = self.toJoin
        if join:
            for target in join:
                for o in join[target]:
                    o.select = True
                target = bpy.data.objects[target]
                target.select = True
                bpy.context.scene.objects.active = target
                bpy.ops.object.join()
                target.select = False
        join.clear()
    
    @classmethod
    def addForJoin(self, o, target):
        """
        Add a Blender object o to be joined with the Blender object <target>
        """
        join = self.toJoin
        # <name> of the Blender object <target> is used as a key for Python dict <join>
        if not target.name in join:
            join[target.name] = []
        join[target.name].append(o)
    
    @classmethod
    def createBlenderObject(self, name, location, parent):
        mesh = bpy.data.meshes.new(name)
        obj = bpy.data.objects.new(name, mesh)
        if location:
            obj.location = location
        bpy.context.scene.objects.link(obj)
        if parent:
            # perform parenting
            obj.parent = parent
        return obj
    
    def getName(self, element):
        tags = element.tags
        return tags["name"] if "name" in tags else "element"
    
    def getMaterial(self, element):
        # the material name is simply <id> of the layer
        name = self.layer.id
        material = bpy.data.materials.get(name)
        
        if not material:
            # create Blender material
            material = createDiffuseMaterial(name, self.app.colors[name])
        return material
    
    def getElementMaterialIndex(self, element):
        # the material name is simply <id> of the layer
        name = self.layer.id
        materialIndex = self.getMaterialIndexByName(name)
        if materialIndex is None:
            # create Blender material
            materialIndex = self.getMaterialIndex( createDiffuseMaterial(name, self.app.colors[name]) )
        return materialIndex
    
    def getMaterialIndex(self, material):
        obj = self.obj
        materialIndex = len(obj.data.materials)
        obj.data.materials.append(material)
        self.materialIndices[material.name] = materialIndex
        return materialIndex
    
    def getMaterialIndexByName(self, name):
        if name in self.materialIndices:
            materialIndex = self.materialIndices[name]
        elif name in bpy.data.materials:
            materialIndex = len(self.obj.data.materials)
            self.obj.data.materials.append( bpy.data.materials[name] )
            self.materialIndices[name] = materialIndex
        else:
            materialIndex = None
        return materialIndex
    
    @staticmethod
    def addShrinkwrapModifier(obj, target, offset):
        m = obj.modifiers.new(name="Shrinkwrap", type='SHRINKWRAP')
        m.wrap_method = "PROJECT"
        m.use_positive_direction = False
        m.use_negative_direction = True
        m.use_project_z = True
        m.target = target
        m.offset = offset
    
    @staticmethod
    def addBoolenModifier(obj, operand):
        m = obj.modifiers.new(name="Boolean", type='BOOLEAN')
        m.object = operand
    
    @staticmethod
    def slice(obj, terrain, app):
        sliceSize = app.sliceSize
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
        
        #x = terrain.minX
        #while x < terrain.maxX:
        #    bmesh.ops.bisect_plane(bm, geom=bm.verts[:]+bm.edges[:]+bm.faces[:], plane_co=(x,0.,0.), plane_no=plane_no)
        #    x += step


class Renderer2d(Renderer):
    
    def __init__(self, op):
        super().__init__(op)
        # vertical position for polygons and multipolygons
        self.z = 0.
    
    def renderLineString(self, element, data):
        self._renderLineString(element, element.getData(data), element.isClosed())
    
    def _renderLineString(self, element, coords, closed):
        bm = self.bm
        z = self.layer.meshZ
        # previous BMesh vertex
        _v = None
        for coord in coords:
            v = bm.verts.new((coord[0], coord[1], z))
            if _v:
                bm.edges.new((_v, v))
            else:
                v0 = v
            _v = v
        if closed:
            # create the closing edge
            bm.edges.new((v, v0))
    
    def renderMultiLineString(self, element, data):
        for i,l in enumerate( element.getDataMulti(data) ):
            self._renderLineString(element, l, element.isClosed(i))
    
    def renderPolygon(self, element, data):
        bm = self.bm
        z = self.layer.meshZ
        f = bm.faces.new(
            bm.verts.new((coord[0], coord[1], z)) for coord in element.getData(data)
        )
        f.normal_update()
        pointNormalUpward(f)
        # assign material to BMFace <f>
        materialIndex = self.getElementMaterialIndex(element)
        f.material_index = materialIndex
        # Store <materialIndex> since it's returned
        # by the default implementation of <Renderer3d.getSideMaterialIndex(..)>
        self.materialIndex = materialIndex
        return f.edges
    
    def renderMultiPolygon(self, element, data):
        # get both outer and inner polygons via <element.getDataMulti(data)>
        return self.createMultiPolygon(element, element.getDataMulti(data))
    
    def createMultiPolygon(self, element, polygons):
        bm = self.bm
        z = self.layer.meshZ
        # the common list of all edges of all polygons
        edges = []
        for polygon in polygons:
            # previous BMesh vertex
            _v = None
            for coord in polygon:
                v = bm.verts.new((coord[0], coord[1], z))
                if _v:
                    edges.append( bm.edges.new((_v, v)) )
                else:
                    v0 = v
                _v = v
            # create the closing edge
            edges.append( bm.edges.new((v, v0)) )
                
        # finally a magic function that does everything
        geom = bmesh.ops.triangle_fill(bm, use_beauty=True, use_dissolve=True, edges=edges)
        # check the normal direction of the created faces and assign material to all BMFace
        materialIndex = self.getElementMaterialIndex(element)
        for f in geom["geom"]:
            if isinstance(f, bmesh.types.BMFace):
                pointNormalUpward(f)
                f.material_index = materialIndex
        # Store <materialIndex> since it's returned
        # by the default implementation of <Renderer3d.getSideMaterialIndex(..)>
        self.materialIndex = materialIndex
        return edges


class Renderer3d(Renderer2d):
    
    def renderPolygon(self, element, data):
        bm = self.bm
        edges = super().renderPolygon(element, data)
        materialIndex = self.getSideMaterialIndex(element)
        
        # extrude the edges
        # each edge has exactly one BMLoop
        # initial top (suffix 't') and bottom (suffix 'b') vertices
        ut = edges[0].link_loops[0].vert
        ub = bm.verts.new(ut.co - self.h)
        _vt = ut
        _vb = ub
        for i in range(len(edges)-1):
            vt = edges[i].link_loops[0].link_loop_next.vert
            vb = bm.verts.new(vt.co - self.h)
            f = bm.faces.new((vt, _vt, _vb, vb))
            f.material_index = materialIndex
            _vt = vt
            _vb = vb
        # the closing face
        f = bm.faces.new((ut, vt, vb, ub))
        f.material_index = materialIndex
        
    def renderMultiPolygon(self, element, data):
        bm = self.bm
        
        # get both outer and inner polygons
        polygons = element.getDataMulti(data)
        edges = self.createMultiPolygon(element, polygons)
        materialIndex = self.getSideMaterialIndex(element)
        
        # index of the first edge of the related closed linestring in the list <edges>
        index = 0
        for polygon in polygons:
            # initial BMLoop
            # each BMEdge from the list <edges> has only one BMLoop
            if not edges[index].link_loops:
                # something wrong with the topology of the related OSM multipolygon
                # update <index> to switch to the next closed linestring
                index += len(polygon)
                # skip that linestring
                continue
            l = edges[index].link_loops[0]
            # initial top (suffix 't') and bottom (suffix 'b') vertices
            ut = l.vert
            ub = bm.verts.new(ut.co - self.h)
            _vt = ut
            _vb = ub
            # Walk along the closed linestring;
            # it can be either an outer polygon outline or a hole in the multipolygon
            while True:
                l = l.link_loop_next
                vt = l.vert
                if vt == ut:
                    # reached the initial vertex, creating the closing face
                    f = bm.faces.new((ut, _vt, _vb, ub))
                    f.material_index = materialIndex
                    break
                else:
                    if len(l.edge.link_loops) == 2:
                        # internal edge, switch to the neighbor polygon
                        l = vt.link_loops[1] if vt.link_loops[0]==l else vt.link_loops[0]
                    vb = bm.verts.new(vt.co - self.h)
                    f = bm.faces.new((vt, _vt, _vb, vb))
                    f.material_index = materialIndex
                    _vt = vt
                    _vb = vb
            # update <index> to switch to the next closed linestring
            index += len(polygon)

    def getSideMaterialIndex(self):
        # return the material index used for the cap
        return self.materialIndex