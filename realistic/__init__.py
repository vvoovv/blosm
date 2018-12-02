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

import bpy, bmesh
import math
from app import app
from util.blender import getBmesh, setBmesh
from .renderer import AreaRenderer, ForestRenderer, WaterRenderer

import util.blender_extra.material


class OperatorMakeRealistic(bpy.types.Operator):
    bl_idname = "blosm.make_realistic"
    bl_label = "Make realistic"
    bl_description = "Make realistic representation on the terrain for the active object with areas"
    bl_options = {'REGISTER', 'UNDO'}
    
    layerId = "water"
    
    @classmethod
    def poll(cls, context):
        return context.scene.objects.get( context.scene.blender_osm.terrainObject )
    
    def invoke(self, context, event):
        obj = context.object
        addon = context.scene.blender_osm
        layerId = addon.makeRealisticLayer
        
        # remove all modifiers
        for i in range(len(obj.modifiers)):
            obj.modifiers.remove(obj.modifiers[i])
        
        # set z-coordinate for all vertices of the input mesh to zero
        bm = getBmesh(obj)
        for v in bm.verts:
            v.co[2] = 0.
        setBmesh(obj, bm)
        
        app.setAttributes(context)
        app.setTerrain(addon.terrainObject, context, False)
        layer = app.getLayer(self.layerId)
        if not layer:
            layer = app.createLayer(self.layerId, swOffset = app.swOffsetDp)
        layer.obj = obj
        
        # Place the input Blender object at the right location in order for
        # the BOOLEAN and SHRINKWRAP modifiers to work correctly
        obj.location = layer.location
        
        # create a renderer
        if layerId == "water":
            renderer = WaterRenderer()
        elif layerId == "forest":
            renderer = ForestRenderer()
        else:
            renderer = AreaRenderer()
            
        renderer.finalizeBlenderObject(layer, app)
        renderer.renderArea(layer, app)
        return {'FINISHED'}


class OperatorMakePolygon(bpy.types.Operator):
    bl_idname = "blosm.make_polygon"
    bl_label = "Make polygon"
    bl_description = "Make a polygon out of connected edges"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'
    
    def invoke(self, context, event):
        obj = context.object
        bm = getBmesh(obj)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0001)
        # a magic function that does everything
        bmesh.ops.triangle_fill(bm, use_beauty=True, use_dissolve=True, edges=bm.edges)
        setBmesh(obj, bm)
        return {'FINISHED'}
    

class OperatorFlattenSelected(bpy.types.Operator):
    bl_idname = "blosm.flatten_selected"
    bl_label = "Flatten selected"
    bl_description = "Flatten selected faces"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH'
    
    def invoke(self, context, event):
        obj = context.object
        bm = bmesh.from_edit_mesh(obj.data)
        
        islandIndex = 0
        islandHeights = []
        # a dictionary of visited faces
        visitedFaces = {}
        # a list of faces to visit during the traversal of adjacent faces
        facesToVisit = []
        
        def processFace(face):
            """
            An auxiliary function used in two places in the code below
            """
            visitedFaces[face.index] = (face, islandIndex)
            # calculate the minumun height of the vertices of <face>
            minZ = min(v.co[2] for v in face.verts)
            if minZ < islandHeights[islandIndex]:
                islandHeights[islandIndex] = minZ
        
        for face in (f for f in bm.faces if f.select):
            if face.index in visitedFaces:
                continue
            islandHeights.append(math.inf)
            processFace(face)
            while True:
                # find a connected face for the BMFace <face>
                for edge in face.edges:
                    linked_faces = edge.link_faces
                    if len(linked_faces) == 2:
                        _face = linked_faces[0] if linked_faces[1] == face else linked_faces[1]
                        if _face.select and not _face.index in visitedFaces:
                            processFace(_face)
                            facesToVisit.append(_face)
                if not facesToVisit:
                    break
                # pick a new face for the traversal
                face = facesToVisit.pop()
            # All adjacent faces are found! Increase <islandIndex>
            islandIndex += 1
        
        # Set new height for the selected faces depending on
        # which island a selected face belongs to
        for i in visitedFaces:
            face, islandIndex = visitedFaces[i]
            for v in face.verts:
                v.co[2] = islandHeights[islandIndex]
        
        bmesh.update_edit_mesh(obj.data)
        return {'FINISHED'}


def register():
    from util import blender_extra
    blender_extra.material.register()

def unregister():
    from util import blender_extra
    blender_extra.material.unregister()