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

import math
from mathutils import Vector
from util.osm import parseNumber
from util.polygon import Polygon
from renderer import Renderer

"""
key: a cardinal direction
value: the related direction in degrees

directions = {
    'N': 0.,
    'NNE': 22.5,
    'NE': 45.,
    'ENE': 67.5,
    'E': 90.,
    'ESE': 112.5,
    'SE': 135.,
    'SSE': 157.5,
    'S': 180.,
    'SSW': 202.5,
    'SW': 225.,
    'WSW': 247.5,
    'W': 270.,
    'WNW': 292.5,
    'NW': 315.,
    'NNW': 337.5
}
"""


class Roof:
    
    assetPath = "roofs.blend"
    
    directions = {
        'N': Vector((0., 1., 0.)),
        'NNE': Vector((0.38268, 0.92388, 0.)),
        'NE': Vector((0.70711, 0.70711, 0.)),
        'ENE': Vector((0.92388, 0.38268, 0.)),
        'E': Vector((1., 0., 0.)),
        'ESE': Vector((0.92388, -0.38268, 0.)),
        'SE': Vector((0.70711, -0.70711, 0.)),
        'SSE': Vector((0.38268, -0.92388, 0.)),
        'S': Vector((0., -1., 0.)),
        'SSW': Vector((-0.38268, -0.92388, 0.)),
        'SW': Vector((-0.70711, -0.70711, 0.)),
        'WSW': Vector((-0.92388, -0.38268, 0.)),
        'W': Vector((-1., 0., 0.)),
        'WNW': Vector((-0.92388, 0.38268, 0.)),
        'NW': Vector((-0.70711, 0.70711, 0.)),
        'NNW': Vector((-0.38268, 0.92388, 0.))
    }
    
    def __init__(self):
        # Python list with vertices is shared accross all operations
        self.verts = []
        self.roofIndices = []
        self.wallIndices = []
    
    def init(self, element, data, minHeight, osm):
        self.verts.clear()
        self.roofIndices.clear()
        self.wallIndices.clear()
        
        self.element = element
        
        verts = self.verts
        self.verts.extend( Vector((coord[0], coord[1], minHeight)) for coord in data )
        # create a polygon located at <minHeight>
        self.polygon = Polygon(verts)
        self.valid = self.polygon.n > 2
        if self.valid:
            # check the direction of vertices, it must be counterclockwise
            self.polygon.checkDirection()
    
    def getHeight(self, app):
        tags = self.element.tags
        h = parseNumber(tags["roof:height"]) if "roof:height" in tags else None
        if h is None:
            # get the number of levels
            if "roof:levels" in tags:
                h = parseNumber(tags["roof:levels"])
            h = self.defaultHeight if h is None else h * app.levelHeight
        self.h = h
        return h
    
    def render(self):
        r = self.r
        wallIndices = self.wallIndices
        roofIndices = self.roofIndices
        if not (roofIndices or wallIndices):
            return
        
        bm = r.bm
        verts = self.verts
        # Create BMesh vertices directly in the Python list <self.verts>
        # First, deal with vertices defining <self.polygon>;
        # some vertices of <self.polygon> could be skipped because of the straight angle
        for i in self.polygon.indices:
            verts[i] = bm.verts.new(r.getVert(verts[i]))
        # Second, create BMesh vertices added after the creation of <self.polygon>;
        # <self.polygon.indexOffset> is used to distinguish between the two groups of vertices
        for i in range(self.polygon.indexOffset, len(verts)):
            verts[i] = bm.verts.new(r.getVert(verts[i]))
        
        if wallIndices:
            materialIndex = r.getWallMaterialIndex(self.element)
            # create BMesh faces for the building walls
            for f in (bm.faces.new(verts[i] for i in indices) for indices in wallIndices):
                f.material_index = materialIndex
        
        if roofIndices:
            materialIndex = r.getRoofMaterialIndex(self.element)
            # create BMesh faces for the building roof
            for f in (bm.faces.new(verts[i] for i in indices) for indices in roofIndices):
                f.material_index = materialIndex
    
    def processDirection(self):
        polygon = self.polygon
        tags = self.element.tags
        # <d> stands for direction
        d = tags.get("roof:direction")
        if d is None:
            d = tags.get("roof:slope:direction")
        # getting a direction vector with the unit length
        if d is None:
            if self.hasRidge and tags.get("roof:orientation") == "across":
                # The roof ridge is across the longest side of the building outline,
                # i.e. the profile direction is along the longest side
                d = max(self.polygon.edges).normalized()
            else:
                d = self.getDefaultDirection()
        elif d in Roof.directions:
            d = Roof.directions[d]
        else:
            # trying to get a direction angle in degrees
            d = parseNumber(d)
            if d is None:
                d = self.getDefaultDirection()
            else:
                d = math.radians(d)
                d = Vector((math.sin(d), math.cos(d), 0.))
        # the direction vector is used by <profile.RoofProfile>
        self.direction = d
        
        # For each vertex from <polygon.verts> calculate projection of the vertex
        # on the vector <d> that defines the roof direction
        projections = self.projections
        projections.extend( d[0]*v[0] + d[1]*v[1] for v in polygon.verts )
        minProjIndex = min(range(polygon.n), key = lambda i: projections[i])
        self.minProjIndex = minProjIndex
        maxProjIndex = max(range(polygon.n), key = lambda i: projections[i])
        self.maxProjIndex = maxProjIndex
        # <polygon> width along the vector <d>
        self.polygonWidth = projections[maxProjIndex] - projections[minProjIndex]
    
    def getDefaultDirection(self):
        polygon = self.polygon
        # a perpendicular to the longest edge of the polygon
        return max(self.polygon.edges).cross(polygon.normal).normalized()