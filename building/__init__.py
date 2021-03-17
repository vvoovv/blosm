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

from numpy import zeros
import parse
from mathutils import Vector
from util.polygon import Polygon


class Building:
    """
    A wrapper for a OSM building
    """
    
    __slots__ = ("outline", "parts", "polygon", "visibility", "auxIndex")
    
    def __init__(self, element, buildingIndex, osm):
        self.outline = element
        self.parts = []
        # a polygon for the outline, used for facade classification only
        self.polygon = None
        # A numpy array to store facade visibility. It consists of two rows.
        # The first row contains the final visibility.
        # The second one contains the intermediary or auxiliary visibility.
        # After each calculation cycle the maximum of the above visibilities is stored in the elements of the first row.
        self.visibility = None
        # an auxiliary variable used to store the first index of the building vertices in an external list or array
        self.auxIndex = 0
        self.markUsedNodes(buildingIndex, osm)
    
    def addPart(self, part):
        self.parts.append(part)

    def markUsedNodes(self, buildingIndex, osm):
        """
        For each OSM node of <self.element> (OSM way or OSM relation) add the related
        <buildingIndex> (i.e. the index of <self> in Python list <buildings> of an instance
        of <BuildingManager>) to Python set <b> of the node 
        """
        for nodeId in self.outline.nodeIds(osm):
            osm.nodes[nodeId].b[buildingIndex] = 1
    
    def initPolygon(self, data):
        outline = self.outline
        polygon = Polygon()
        if outline.t is parse.multipolygon:
            polygon.init( Vector((coord[0], coord[1], 0.)) for coord in outline.getOuterData(data) )
        else:
            polygon.init( Vector((coord[0], coord[1], 0.)) for coord in outline.getData(data) )
        if polygon.n < 3:
            # skip it
            return
        # check the direction of vertices, it must be counterclockwise
        polygon.checkDirection()
        self.polygon = polygon
    
    def initVisibility(self):
        # <self.polygon> must be already initialized
        
        # The number of elements in each row of <self.visibility> is equal to the number of vertices in
        # <self.polygon.allVerts>, i.e. the vertices forming a straight angle are also included
        self.visibility = zeros((2, len(self.polygon.allVerts)))
    
    def updateAuxVisibilityAdd(self, polygonEdgeIndex, term):
        self.visibility[1][self.polygon.indices[polygonEdgeIndex]] += term
        
    def updateAuxVisibilityDivide(self, polygonEdgeIndex, denominator):
        self.visibility[1][self.polygon.indices[polygonEdgeIndex]] /= denominator
    
    def updateVisibilityMax(self, polygonEdgeIndex):
        index = self.polygon.indices[polygonEdgeIndex]
        self.visibility[0][index] = max(self.visibility[0][index], self.visibility[1][index])