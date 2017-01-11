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

import os, bpy, bmesh
from util import zeroVector
from util.blender import createEmptyObject, createDiffuseMaterial
from util.osm import assignTags


class Renderer:
    
    # types of data
    linestring = (1,)
    multilinestring = (1,)
    polygon = (1,)
    multipolygon = (1,)
    
    parent = None
    name = None
    
    def __init__(self, op):
        self.op = op
    
    @classmethod
    def begin(self, op):
        self.name = os.path.basename(op.filepath)
        
        if op.layered or not op.singleObject:
            self.parent = createEmptyObject(
                self.name,
                zeroVector(),
                empty_draw_size=0.01
            )
        if op.singleObject:
            if op.layered:
                self.layerMeshes = [None for _ in op.layerIndices]
                self.layerObjects = [None for _ in op.layerIndices]
                # cache material indices in <self.materialIndices>
                self.materialIndices = [None for _ in op.layerIndices]
            else:
                self.bm = bmesh.new()
                self.obj = self.createBlenderObject(self.name, None)
                # cache material indices in <self.materialIndices>
                self.materialIndices = {}
        else:
            if op.layered:
                self.layerParents = [None for _ in op.layerIndices]

        # store here Blender object that are to be joined
        self.toJoin = {}
    
    def preRender(self, element, layerIndex=None):
        op = self.op
        # <li> stands for 'layer index'
        li = element.li if layerIndex is None else layerIndex
        self.layerIndex = li
        
        if op.singleObject:
            if op.layered:
                mesh = Renderer.layerMeshes[li]
                obj = Renderer.layerObjects[li]
                materialIndices = Renderer.materialIndices[li]
                if not mesh:
                    mesh = bmesh.new()
                    Renderer.layerMeshes[li] = mesh
                    obj = self.createBlenderObject(
                        self.getLayerName(li, op),
                        self.parent
                    )
                    Renderer.layerObjects[li] = obj
                    materialIndices = {}
                    Renderer.materialIndices[li] = materialIndices
                self.bm = mesh
                self.obj = obj
                self.materialIndices = materialIndices
            else:
                self.bm = Renderer.bm
                self.obj = Renderer.obj
                self.materialIndices = Renderer.materialIndices
        else:
            self.obj = self.createBlenderObject(
                self.getName(element),
                self.getLayerParent() if op.layered else Renderer.parent
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
        if not self.op.singleObject:
            # finalize BMesh
            self.bm.to_mesh(self.obj.data)
            # assign OSM tags to the blender object
            assignTags(self.obj, element.tags)
    
    @classmethod
    def end(self, op):
        if op.singleObject:
            if op.layered:
                for bm,obj in zip(self.layerMeshes, self.layerObjects):
                    if bm:
                        bm.to_mesh(obj.data)
            else:
                # finalize BMesh
                self.bm.to_mesh(self.obj.data)
        bpy.context.scene.update()
        self.join()
    
    @classmethod
    def join(self):
        """
        Join Blender object collect during rendering
        """
        join = self.toJoin
        if join:
            bpy.ops.object.select_all(action="DESELECT")
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
    def createBlenderObject(self, name, parent=None):
        mesh = bpy.data.meshes.new(name)
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.scene.objects.link(obj)
        if parent:
            # perform parenting
            obj.parent = parent
        return obj
    
    def getName(self, element):
        tags = element.tags
        return tags["name"] if "name" in tags else "element"
    
    def getLayerParent(self):
        layerIndex = self.layerIndex
        layerParents = self.layerParents
        layerParent = layerParents[layerIndex]
        if not layerParent:
            layerParent = createEmptyObject(
                self.getLayerName(layerIndex, self.op),
                zeroVector(),
                empty_draw_size=0.01
            )
            layerParent.parent = Renderer.parent
            layerParents[layerIndex] = layerParent
        return layerParent
    
    @classmethod
    def getLayerName(self, layerIndex, op):
        return self.name + "_" + op.layerIds[layerIndex]
    
    def getMaterial(self, element):
        op = self.op
        # the material name is simply <layerId>
        name = op.layerIds[self.layerIndex]
        material = bpy.data.materials.get(name)
        
        if not material:
            # create Blender material
            material = createDiffuseMaterial(name, op.colors[name])
        return material
    
    def getElementMaterialIndex(self, element):
        op = self.op
        # the material name is simply <layerId>
        name = op.layerIds[self.layerIndex]
        materialIndex = self.getMaterialIndexByName(name)
        if materialIndex is None:
            # create Blender material
            materialIndex = self.getMaterialIndex( createDiffuseMaterial(name, op.colors[name]) )
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


class Renderer2d(Renderer):
    
    def __init__(self, op):
        super().__init__(op)
        # vertical position for polygons and multipolygons
        self.z = 0.
    
    def renderLineString(self, element, data):
        self._renderLineString(element, element.getData(data), element.isClosed())
    
    def _renderLineString(self, element, coords, closed):
        bm = self.bm
        # previous BMesh vertex
        _v = None
        for coord in coords:
            v = bm.verts.new((coord[0], coord[1], 0.))
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
        f = bm.faces.new(
            bm.verts.new((coord[0], coord[1], self.z)) for coord in element.getData(data)
        )
        f.normal_update()
        if f.normal.z < 0.:
            f.normal_flip()
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
        # the common list of all edges of all polygons
        edges = []
        for polygon in polygons:
            # previous BMesh vertex
            _v = None
            for coord in polygon:
                v = bm.verts.new((coord[0], coord[1], self.z))
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
                if f.normal.z < 0.:
                    f.normal_flip()
                f.material_index = materialIndex
        # Store <materialIndex> since it's returned
        # by the default implementation of <Renderer3d.getSideMaterialIndex(..)>
        self.materialIndex = materialIndex
        return edges

    def reset(self):
        self.z = 0.


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