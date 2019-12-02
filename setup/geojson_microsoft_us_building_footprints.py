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

from geojson import Manager, BuildingManager
from renderer import Renderer2d, Renderer
from building.renderer import BuildingRenderer

from manager.logging import Logger

import bpy, bmesh
from mathutils.bvhtree import BVHTree
from mathutils import Vector
from util.blender import getBmesh
from util import zeroVector, zAxis


_isBlender280 = bpy.app.version[1] >= 80


filterMeshObjectName = "filter_mesh"


class GeoJsonBuildingManager(BuildingManager):
    
    def render(self):
        self.renderer.prepareAuxMeshes()
        
        for building in self.buildings:
            self.renderer.render(building, self.osm)
        
        self.renderer.cleanupAuxMeshes()


class GeoJsonBuildingRenderer(BuildingRenderer):
    
    projectionDirection = -zAxis # downwards
    projectionLocationZ = 10000
    
    def prepareAuxMeshes(self):
        filterBmesh = getBmesh(bpy.data.objects[filterMeshObjectName])
        self.filterBvhTree = BVHTree.FromBMesh(filterBmesh)
        filterBmesh.free()
        # a Bmesh for the triangulation
        self.triangulationBmesh = bmesh.new()
    
    def render(self, building, data):
        outline = building.element
        outlineData = tuple(outline.getData(data) if outline.t is Renderer.polygon else outline.getOuterData(data))
        
        # check if we have a triangle
        numPoints = len(outlineData)
        if numPoints == 3:
            center = sum((Vector(coord) for coord in outlineData), zeroVector())/numPoints
            if self.hitFilterMesh(center):
                return
        else:
            # create a temporary polygon and triangulate it
            bm = self.triangulationBmesh
            bm.faces.new(
                bm.verts.new((coord[0], coord[1], 0.)) for coord in outlineData
            )
            bmesh.ops.triangulate(bm, faces=bm.faces)
            for face in bm.faces:
                # project the geometrical center of <triangle> on filter mesh
                center = sum((vert.co for vert in face.verts), zeroVector())/3.
                # if at least one center of the triangle hits the filter mesh, skip it
                if self.hitFilterMesh(center):
                    bm.clear()
                    return
            bm.clear()
        super().render(building, data)
    
    def hitFilterMesh(self, point):
        _faceIndex = self.filterBvhTree.ray_cast(
            (point[0], point[1], GeoJsonBuildingRenderer.projectionLocationZ), 
            GeoJsonBuildingRenderer.projectionDirection
        )[2]
        return not _faceIndex is None
    
    def cleanupAuxMeshes(self):
        self.filterBvhTree = None
        self.triangulationBmesh.free()
        self.triangulationBmesh = None


def setup(app, data):
    # comment the next line if logging isn't needed
    Logger(app, data)
    
    data.skipNoProperties = False
    
    manager = Manager(data)
    
    if app.buildings:
        if app.mode is app.twoD:
            data.addCondition(
                lambda tags, e: True,
                "buildings", 
                manager
            )
        else: # 3D
            # no building parts for the moment
            buildings = GeoJsonBuildingManager(data, None) if filterMeshObjectName in bpy.data.objects else BuildingManager(data, None)
            
            data.addCondition(
                lambda tags, e: True,
                "buildings",
                buildings
            )
            #osm.addCondition(
            #    lambda tags, e: "building:part" in tags,
            #    None,
            #    buildingParts
            #)
            buildings.setRenderer(
                GeoJsonBuildingRenderer(app) if filterMeshObjectName in bpy.data.objects else BuildingRenderer(app)
            )
            app.managers.append(buildings)
    
    numConditions = len(data.conditions)
    if not app.mode is app.twoD and app.buildings:
        # 3D buildings aren't processed by BaseManager
        numConditions -= 1
    if numConditions:
        manager.setRenderer(Renderer2d(app))
        app.managers.append(manager)