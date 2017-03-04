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

from manager import Manager
from renderer import Renderer
from parse import Osm
from util import zAxis
from . import Building

from mathutils.bvhtree import BVHTree


class BuildingManager(Manager):
    
    def __init__(self, osm, buildingParts):
        """
        Args:
            osm (parse.Osm): Parsed OSM data
            buildingParts (BuildingParts): A manager for 3D parts of an OSM building
        """
        super().__init__(osm)
        self.buildings = []
        self.parts = buildingParts.parts

    def parseWay(self, element, elementId):
        if element.closed:
            element.t = Renderer.polygon
            # create a wrapper for the OSM way <element>
            building = Building(element)
            # store the related wrapper in the attribute <b>
            element.b = building
            self.buildings.append(building)
        else:
            element.valid = False
    
    def parseRelation(self, element, elementId):
        # create a wrapper for the OSM relation <element>
        building = Building(element)
        # store the related wrapper in the attribute <b>
        element.b = building
        self.buildings.append(building)
    
    def process(self):
        # create a Python set for each OSM node to store indices of the entries from <self.buildings>,
        # where instances of the wrapper class <building.manager.Building> are stored
        buildings = self.buildings
        osm = self.osm
        nodes = osm.nodes
        for n in nodes:
            nodes[n].b = set()
        
        # iterate through buildings (building=*) to mark nodes used by the building
        for i,building in enumerate(buildings):
            building.markUsedNodes(i, osm)
        
        # Iterate through building parts (building:part=*)
        # to find a building from <self.buildings> to which
        # the related <part> belongs
        
        # create a BHV tree on demand only
        bvhTree = None
        for part in self.parts:
            if part.o:
                # the outline for <part> is set in an OSM relation of the type 'building'
                osmId, osmType = part.o
                elements = osm.ways if osmType is Osm.way else osm.relations
                if osmId in elements:
                    building = elements[osmId].b
                    if building:
                        building.addPart(part)
                        # we are done
                        continue
            
            # Find a building from <self.buildings> which OSM nodes <part> uses
            # the maximum number of times
            
            # <shared> is Python dict for OSM nodes shared by <part> and some buildings from <self.buildings>:
            # key: the index of <self.buildings> of the related building
            # value: the number of times the nodes of the related building are used by <part>
            shared = {}
            # Indices of buildings from <self.buildings>
            # considered as candidates for building we are looking for
            candidates = []
            # how many nodes of <part> are used by buildings from <candidates> 
            maxNumber = 0
            # The first encountered free OSM node, i.e, the OSM node not used by any building from <self.buildings>
            freeNode = None
            for nodeId in part.nodeIds(osm):
                if nodes[nodeId].b:
                    for buildingIndex in nodes[nodeId].b:
                        if buildingIndex in shared:
                            shared[buildingIndex] += 1
                        else:
                            shared[buildingIndex] = 1
                        if shared[buildingIndex] > maxNumber:
                            candidates = [buildingIndex]
                            maxNumber += 1
                        elif shared[buildingIndex] == maxNumber:
                            candidates.append(buildingIndex)
                elif freeNode is None:
                    freeNode = nodeId
            
            numCandidates = len(candidates)
            if freeNode:
                if numCandidates == 1:
                    # To save time we won't check if all free OSM nodes of <part>
                    # are located inside the building
                    buildings[candidates[0]].addPart(part)
                else:
                    # Take the first encountered free node <freeNode> and
                    # calculated if it is located inside any building from <self.buildings>
                    if not bvhTree:
                        bvhTree = self.createBvhTree()
                    coords = nodes[freeNode].getData(osm)
                    # Cast a ray from the point with horizontal coords equal to <coords> and
                    # z = -1. in the direction of <zAxis>
                    buildingIndex = bvhTree.ray_cast((coords[0], coords[1], -1.), zAxis)[2]
                    if not buildingIndex is None:
                        # we condider that <part> is located inside <buildings[buildingIndex]>
                        buildings[buildingIndex].addPart(part)
            else:
                # all OSM nodes of <part> are used by one or more buildings from <self.buildings>
                # the case numCandidates > 1 probably means some weird configuration, so skip that <part>
                if numCandidates == 1:
                    buildings[candidates[0]].addPart(part)
    
    def render(self):
        for building in self.buildings:
            self.renderer.render(building, self.osm)
    
    def createBvhTree(self):
        osm = self.osm
        vertices = []
        polygons = []
        vertexIndex1 = 0
        vertexIndex2 = 0
        
        for building in self.buildings:
            # OSM element (a OSM way or an OSM relation of the type 'multipolygon')
            element = building.element
            # In the case of a multipolygon we consider the only outer linestring that defines the outline
            # of the polygon
            polygon = element.getData(osm) if element.t is Renderer.polygon else element.getOuterData(osm)
            if not polygon:
                # no outer linestring, so skip it
                continue
            vertices.extend(polygon)
            vertexIndex2 = len(vertices)
            polygons.append(tuple(range(vertexIndex1, vertexIndex2)))
            vertexIndex1 = vertexIndex2
        return BVHTree.FromPolygons(vertices, polygons)


class BuildingParts:
    
    def __init__(self):
        # don't accept broken multipolygons
        self.acceptBroken = False
        self.parts = []
        
    def parseWay(self, element, elementId):
        if element.closed:
            element.t = Renderer.polygon
            # empty outline
            element.o = None
            self.parts.append(element)
        else:
            element.valid = False
    
    def parseRelation(self, element, elementId):
        # empty outline
        element.o = None
        self.parts.append(element)


class BuildingRelations:
    
    def parseRelation(self, element, elementId):
        # no need to store the relation in <osm.relations>, so return True
        return True