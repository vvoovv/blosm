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

import bmesh
from mathutils import Vector
from util.polygon import PolygonOLD
from util.blender import pointNormalUpward
from . import Roof


class RoofFlat(Roof):
    """
    A class to deal with building or building parts with the flat roof
    for a simple polygon (not a multipolygon)
    """
    
    defaultHeight = 0.
    
    def make(self, osm):
        polygon = self.polygon
        n = len(self.verts)
        
        # Extrude <polygon> in the direction of <z> axis to bring
        # the extruded part to the height <bldgMaxHeight>
        polygon.extrude(self.z2, self.wallIndices)
        # fill the extruded part
        self.roofIndices.append( tuple(range(n, n+polygon.n)) )
        return True
    
    def getRoofItem(self):
        return 


class RoofFlatMulti(RoofFlat):
    """
    A class to deal with building or building parts with the flat roof
    for a multipolygon
    """
    
    def __init__(self):
        self.verts = []
        self.wallIndices = []
        self.polygons = []
        # The line below is needed for consistency;
        # it can be interpreted as the attribute that a roof is defined be the multipolygon
        # rather than a polygon
        self.polygon = None
    
    def init(self, element, data, osm, app):
        # <data> isn't used it this class
        self.verts.clear()
        self.wallIndices.clear()
        self.polygons.clear()
        
        self._levelHeight = None
        self._levels = None
        
        self.valid = True
        
        self.element = element
        
        self.calculateDimensions(self.getMinHeight())
        if not self.valid:
            return
        if self.noWalls:
            # a workaround for non-flat roofs with holes
            self.noWalls = False
            self.wallHeight = self.z2 - self.z1
    
    def make(self, osm):
        element = self.element
        
        # create vertices for all linestrings of the multipolygon
        verts = self.verts
        polygons = self.polygons
        indexOffset = 0
        for _l in element.ls:
            verts.extend(
                Vector((v[0], v[1], self.z2)) for v in element.getLinestringData(_l, osm)
            )
            n = len(verts)
            
            polygon = PolygonOLD(
                verts,
                tuple(range(indexOffset, n))             
            )
            if polygon.n > 2:
                polygons.append(polygon)
            indexOffset = n
        if not polygons:
            return False
        # vertices for the bottom  
        verts.extend(Vector((verts[i].x, verts[i].y, self.z1)) for p in polygons for i in p.indices)
        return True
    
    def render(self):
        r = self.r
        verts = self.verts
        polygons = self.polygons
        bm = r.bm
        # Create BMesh vertices directly in the Python list <self.verts>
        # First, deal with vertices defining each polygon in <polygons>;
        # some vertices of a polygon could be skipped because of the straight angle
        for polygon in polygons:
            for i in polygon.indices:
                verts[i] = bm.verts.new(r.getVert(verts[i]))
        # Second, create BMesh vertices added after the creation of <polygons>;
        # <polygons[-1].indexOffset> (i.e. <indexOffset> of the last polygon in <polygons>)
        # is used to distinguish between the two groups of vertices
        for i in range(polygons[-1].indexOffset, len(verts)):
            verts[i] = bm.verts.new(r.getVert(verts[i]))
        # create BMesh edges out of <verts>
        edges = tuple(
            bm.edges.new( (verts[polygon.indices[i-1]], verts[polygon.indices[i]]) )\
            for polygon in polygons\
            for i in range(polygon.n)
        )
        
        # a magic function that does everything
        self.renderRoofTexturedMulti(
            bmesh.ops.triangle_fill(bm, use_beauty=True, use_dissolve=True, edges=edges)
        )
        
        # create BMesh faces for the walls of the building
        indexOffset1 = 0
        indexOffset2 = polygons[-1].indexOffset
        wallIndices = self.wallIndices
        for polygon in polygons:
            n = polygon.n
            # the first edge of the polygon
            edge = edges[indexOffset1]
            if not edge.link_loops:
                # something wrong with the topology of the related OSM multipolygon
                # update index offsets to switch to the next polygon (i.e. a closed linestring)
                indexOffset1 += n
                indexOffset2 += n
                # skip that polygon
                continue
            # a BMLoop for <edge>
            l = edge.link_loops[0]
            # check if the direction of <polygon> needs to be reverted
            keepDirection = l.link_loop_next.vert == verts[polygon.indices[0]]
            
            wallIndices.extend(
                (
                    indexOffset2 - 1 + (i if i else n),
                    indexOffset2 + i,
                    polygon.indices[i],
                    polygon.indices[i-1]
                )\
                if keepDirection else\
                (
                    indexOffset2 + i,
                    indexOffset2 - 1 + (i if i else n),
                    polygon.indices[i-1],
                    polygon.indices[i]
                )\
                for i in range(n)
            )
            
            indexOffset1 += n
            indexOffset2 += n
        
        self.renderWalls()
        
    def renderWalls(self):
        """
        The method can be overriden by a child class
        """
        verts = self.verts
        bm = self.r.bm
        wallIndices = self.wallIndices
        
        materialIndex = self.r.getWallMaterialIndex(self.element)
        # actual code to create BMesh faces for the building walls out of <verts> and <wallIndices>
        for f in (bm.faces.new(verts[i] for i in indices) for indices in wallIndices):
            f.material_index = materialIndex
    
    def renderRoofTexturedMulti(self, geom):
        materialIndex = self.r.getRoofMaterialIndex(self.element)
        # check the normal direction of the created faces and assign material to all BMesh faces
        for f in geom["geom"]:
            if isinstance(f, bmesh.types.BMFace):
                pointNormalUpward(f)
                f.material_index = materialIndex