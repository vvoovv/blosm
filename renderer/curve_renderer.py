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
from mathutils.bvhtree import BVHTree
from . import Renderer
from terrain import direction
from util.osm import assignTags

_isBlender280 = bpy.app.version[1] >= 80


class CurveRenderer(Renderer):
    
    # <insetValue> the maximum way width in <assets/way_profiles.blend>
    insetValue = 5.
    
    def __init__(self, app):
        super().__init__(app)
        self.bvhTree = None
        # the current spline for the Blender curve
        self.spline = None
        # point index of the current point in <self.spline>
        self.pointIndex = 0
        # Node counter for the nodes of an OSM way actually added to <self.spline>;
        # used only
        # 1) in the presense of the terrain
        # 2) AND if the OSM way is closed;
        # If <self.nodeCounter> isn't equal to the number of nodes in the OSM way,
        # then the OSM way will have open ends instead of being closed
        self.nodeCounter = 0
    
    def prepare(self):
        terrain = self.app.terrain
        if terrain:
            # Do we need to add extra points for a long curve segment
            # to ensure that it lies above the terrain after the SHRINKWRAP modifier
            # is applied?
            self.subdivideSegment = self.app.subdivide
            self.subdivideSize = self.app.subdivisionSize
            
            if not terrain.envelope:
                terrain.createEnvelope()
            # BMesh <bm> is used to check if a way's node is located
            # within the terrain. It's smaller than <terrain.envelope>
            # since all points of the bevel object applied to the Blender curve
            # must be located within <terrain> to avoid weird results of
            # the SHRINKWRAP modifier
            bm = bmesh.new()
            bm.from_mesh(terrain.envelope.data)
            # inset faces to avoid weird results of the BOOLEAN modifier
            insetFaces = bmesh.ops.inset_region(bm, faces=bm.faces,
                use_boundary=True, use_even_offset=True, use_interpolate=True,
                use_relative_offset=False, use_edge_rail=False, use_outset=False,
                thickness=self.insetValue, depth=0.
            )['faces']
            bmesh.ops.delete(bm, geom=insetFaces, context='FACES' if _isBlender280 else 5)
            self.bvhTree = BVHTree.FromBMesh(bm)
            # <bm> isn't needed anymore
            bm.free()
    
    def preRender(self, element):
        layer = element.l
        self.layer = layer
        
        if layer.singleObject:
            if not layer.obj:
                layer.obj = self.createBlenderObject(
                    layer.name,
                    layer.location,
                    collection = self.collection,
                    parent = None if _isBlender280 else self.parent
                )
            self.obj = layer.obj
        else:
            self.obj = self.createBlenderObject(
                self.getName(element),
                self.offsetZ or self.offset or layer.location,
                collection = layer.getCollection(self.collection) if _isBlender280 else None,
                parent = layer.getParent(layer.getCollection(self.collection) if _isBlender280 else None)
            )

    def renderLineString(self, element, data):
        self._renderLineString(element, element.getData(data), element.isClosed())

    def renderMultiLineString(self, element, data):
        for i,l in enumerate( element.getDataMulti(data) ):
            self._renderLineString(element, l, element.isClosed(i))
    
    def _renderLineString(self, element, coords, closed):
        z = self.layer.meshZ
        if self.app.terrain:
            self.spline = None
            # the preceding point of the spline segment
            point0 = None
            onTerrain0 = None
            if self.subdivideSegment:
                for i, coord in enumerate(coords):
                    # Cast a ray from the point with horizontal coords equal to <coords> and
                    # z = <z> in the direction of <direction>
                    point = Vector((coord[0], coord[1], z))
                    onTerrain = self.isPointOnTerrain(point)
                    if closed and not i:
                        # remember the original point
                        _point = point
                        _onTerrain = onTerrain
                    #
                    # Perform calculations for the segment subdivision
                    #
                    if point0 and (onTerrain0 or onTerrain):
                        numPoints, vec = self.getSubdivisionParams(point0, point)
                    if onTerrain:
                        if onTerrain0:
                            self.processOnTerrainOnTerrain(point0, point, numPoints, vec, closed)
                        elif point0 and numPoints:
                            self.processNoTerrainOnTerrain(point0, point, numPoints, vec)
                    elif onTerrain0:
                        if point0 and numPoints:
                            self.processOnTerrainNoTerrain(point0, point, numPoints, vec)
                    elif self.spline:
                        self.spline = None
                    point0 = point
                    onTerrain0 = onTerrain
                if closed:
                    # <i+1> is equal to the number of nodes in the OSM way
                    if self.nodeCounter != i+1:
                        closed = False
                    numPoints, vec = self.getSubdivisionParams(point0, _point)
                    if onTerrain0 and _onTerrain:
                        self.processOnTerrainOnTerrain(point0, _point, numPoints, vec, closed)
                    elif numPoints:
                        if onTerrain0:
                            self.processOnTerrainNoTerrain(point0, _point, numPoints, vec)
                        elif _onTerrain:
                            self.processNoTerrainOnTerrain(point0, _point, numPoints, vec)
            else:
                for i, coord in enumerate(coords):
                    # Cast a ray from the point with horizontal coords equal to <coords> and
                    # z = <z> in the direction of <direction>
                    point = Vector((coord[0], coord[1], z))
                    onTerrain = self.isPointOnTerrain(point)
                    if onTerrain and onTerrain0:
                            if not self.spline:
                                self.createSpline()
                                self.setSplinePoint(point0)
                                if closed: self.nodeCounter = 1
                            self.spline.points.add(1)
                            self.setSplinePoint(point)
                            if closed: self.nodeCounter += 1
                    elif self.spline:
                        self.spline = None
                    point0 = point
                    onTerrain0 = onTerrain
                # <i+1> is equal to the number of nodes in the OSM way
                if closed and self.nodeCounter != i+1:
                    closed = False
        else:
            self.createSpline()
            for i, coord in enumerate(coords):
                if i:
                    self.spline.points.add(1)
                self.setSplinePoint((coord[0], coord[1], z))
        if closed and self.spline:
            self.spline.use_cyclic_u = True
    
    def getSubdivisionParams(self, point0, point):
        vec = point - point0
        numPoints = math.floor(vec.length/self.subdivideSize)
        # subdivision step (a vector) is equal to <vec/(numPoints+1)>
        return numPoints, vec/(numPoints+1) if numPoints else None
    
    def createSpline(self):
        self.spline = self.obj.data.splines.new('POLY')
        self.pointIndex = 0

    def setSplinePoint(self, point):
        self.spline.points[self.pointIndex].co = (point[0], point[1], point[2], 1.)
        self.pointIndex += 1
    
    def processOnTerrainOnTerrain(self, point0, point, numPoints, vec, closed):
        """
        A method to process the case:
        the terrain is available and <self.subdivideSegment == True>
        and both <point0> (the first one in the spline segment)
        and <point> (the second on in the spline segment) are located on the terrain
        """
        if not self.spline:
            self.createSpline()
            self.setSplinePoint(point0)
            if closed: self.nodeCounter = 1
        if numPoints:
            self.spline.points.add(numPoints+1)
            p = point0
            for _ in range(numPoints):
                p = p + vec
                self.setSplinePoint(p)
        else:
            self.spline.points.add(1)
        self.setSplinePoint(point)
        if closed: self.nodeCounter += 1
    
    def processNoTerrainOnTerrain(self, point0, point, numPoints, vec):
        """
        A method to process the case:
        the terrain is available and <self.subdivideSegment == True>
        and <point0> (the first one in the spline segment) is located outside the terrain
        and <point> (the second on in the spline segment) is located on the terrain
        """
        firstTerrainPointIndex = 0
        bound1 = 0
        bound2 = numPoints + 1
        # index of subdivision points starting from 1
        pointIndex = math.ceil((bound1 + bound2)/2)
        while True:
            # coordinates of the subdivision point with the index <pointIndex>
            if self.isPointOnTerrain(point0 + pointIndex * vec):
                firstTerrainPointIndex = pointIndex
                if pointIndex == bound1+1:
                    break
                bound2 = pointIndex
                pointIndex = math.floor((bound1 + bound2)/2)
            else:
                if pointIndex==bound2-1:
                    break
                bound1 = pointIndex
                pointIndex = math.ceil((bound1 + bound2)/2)
        if firstTerrainPointIndex:
            self.createSpline()
            self.spline.points.add(numPoints - firstTerrainPointIndex + 1)
            for pointIndex in range(firstTerrainPointIndex, numPoints+1):
                p = point0 + pointIndex * vec
                self.setSplinePoint(p)
            self.setSplinePoint(point)
    
    def processOnTerrainNoTerrain(self, point0, point, numPoints, vec):
        """
        A method to process the case:
        the terrain is available and <self.subdivideSegment == True>
        and <point0> (the first one in the spline segment) is located on the terrain
        and <point> (the second on in the spline segment) is located outside the terrain
        """
        lastTerrainPointIndex = 0
        bound1 = 0
        bound2 = numPoints + 1
        # index of subdivision points starting from 1
        pointIndex = math.floor((bound1 + bound2)/2)
        while True:
            # coordinates of the subdivision point with the index <pointIndex>
            if self.isPointOnTerrain(point0 + pointIndex * vec):
                lastTerrainPointIndex = pointIndex
                if pointIndex == bound2-1:
                    break
                bound1 = pointIndex
                pointIndex = math.ceil((bound1 + bound2)/2)
            else:
                if pointIndex==bound1+1:
                    break
                bound2 = pointIndex
                pointIndex = math.floor((bound1 + bound2)/2)
        if lastTerrainPointIndex:
            if not self.spline:
                self.createSpline()
                self.setSplinePoint(point0)
            self.spline.points.add(lastTerrainPointIndex)
            for pointIndex in range(1, lastTerrainPointIndex+1):
                p = point0 + pointIndex * vec
                self.setSplinePoint(p)
        self.spline = None
    
    def postRender(self, element):
        layer = element.l
        
        if not layer.singleObject:
            obj = self.obj
            # assign OSM tags to the blender object
            assignTags(obj, element.tags)
            layer.finalizeBlenderObject(obj)
    
    def cleanup(self):
        super().cleanup()
        self.bvhTree = None

    @classmethod
    def createBlenderObject(self, name, location, collection=None, parent=None):
        curve = bpy.data.curves.new(name, 'CURVE')
        curve.fill_mode = 'NONE'
        obj = bpy.data.objects.new(name, curve)
        if location:
            obj.location = location
        if _isBlender280:
            if not collection:
                collection = bpy.context.scene.collection
            # adding to the collection
            collection.objects.link(obj)
        else:
            bpy.context.scene.objects.link(obj)
        if parent:
            # perform parenting
            obj.parent = parent
        return obj
    
    def isPointOnTerrain(self, point):
        return self.bvhTree.ray_cast(point, direction)[0]