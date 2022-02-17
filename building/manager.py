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

from manager import Manager
import parse
from parse.osm import Osm
from util import zAxis
from . import Building, BldgEdge, BldgPart
from defs.building import Visited


# Adapted from https://stackoverflow.com/questions/16542042/fastest-way-to-sort-vectors-by-angle-without-actually-computing-that-angle
# Output: a number from the range [0 .. 4] which is monotonic
#         in the angle this vector makes against the x axis.
def _pseudoangle(edge, _id):
    if _id == edge.id1:
        dx, dy = edge.v2 - edge.v1
    else:
        dx, dy = edge.v1 - edge.v2
    p = dx/(abs(dx)+abs(dy)) # -1 .. 1 increasing with x
    if dy < 0: 
        return 3 + p  #  2 .. 4 increasing with x
    else:
        return 1 - p  #  0 .. 2 decreasing with x


def _notInsideAnotherBldgPart(vector):
    vectorPrev = vector.prev
    vectorsOverlapPrev = vectorPrev.edge.partVectors12 if vectorPrev.direct else vectorPrev.edge.partVectors21
    if vectorsOverlapPrev:
        vector = vector.vector
        vectorPrev = vectorPrev.vector
        # project <vector> on <vectorPrev> and calculate the pseudoangle
        # dx = vectorPrev.dot(vector)
        # dy = vectorPrev.cross(vector)
        p = _pseudoangle(
            vectorPrev.dot(vector),
            vectorPrev.cross(vector)
        )
        for vectorOverlapPrev in vectorsOverlapPrev:
            vectorOverlapNext = vectorOverlapPrev.next.vector
            # project <vectorOverlapNext> on <vectorPrev> and calculate the pseudoangle
            _p = _pseudoangle( vectorPrev.dot(vectorOverlapNext), vectorPrev.cross(vectorOverlapNext) )
            if _pseudoangle( vectorPrev.dot(vectorOverlapNext), vectorPrev.cross(vectorOverlapNext) ) > p:
                return False
    return True


class BaseBuildingManager:
    
    def __init__(self, data, app, buildingParts, layerClass):
        """
        Args:
            data (parse.Osm): Parsed OSM data
            app (app.App): An application
            buildingParts (BuildingParts): A manager for 3D parts of an OSM building
            layerClass: A layer class
        """
        self.id = "buildings"
        self.data = data
        self.app = app
        self.layerClass = layerClass
        self.buildings = []
        self.buildingCounter = 0
        # don't accept broken multipolygons
        self.acceptBroken = False
        if buildingParts:
            self.parts = buildingParts.parts
            buildingParts.bldgManager = self
        self.actions = []
        self.edges = {}
        app.addManager(self)
    
    def setRenderer(self, renderer):
        self.renderer = renderer
        self.app.addRenderer(renderer)
    
    def createLayer(self, layerId, app, **kwargs):
        return app.createLayer(layerId, self.layerClass)
    
    def parseWay(self, element, elementId):
        if element.closed:
            element.t = parse.polygon
            self.createBuilding(element)
        else:
            element.valid = False
    
    def parseRelation(self, element, elementId):
        self.createBuilding(element)
    
    def createBuilding(self, element):
        # create a wrapper for the OSM way <element>
        building = Building(element, self)
        # store the related wrapper in the attribute <b>
        element.b = building
        self.buildings.append(building)
        self.buildingCounter += 1
    
    def process(self):
        for building in self.buildings:
            # create <building.polygon>
            building.init(self)
            
        for action in self.actions:
            action.do(self)
    
    def render(self):
        for building in self.buildings:
            self.renderer.render(building, self.data)
    
    def addAction(self, action):
        action.app = self.app
        self.actions.append(action)
    
    def getEdge(self, nodeId1, nodeId2):
        data = self.data
        
        if nodeId1 > nodeId2:
            nodeId1, nodeId2 = nodeId2, nodeId1
        key = "%s_%s" % (nodeId1, nodeId2)
        edge = self.edges.get(key)
        if not edge:
            edge = BldgEdge(
                nodeId1,
                data.nodes[nodeId1].getData(data),
                nodeId2,
                data.nodes[nodeId2].getData(data)
            )
            self.edges[key] = edge
        return edge
    
    def markBldgPartEdgeNodes(self, edge):
        """
        Add <edge> to <self.data.nodes[edge.id1].bldgPartEdges> and <self.data.nodes[edge.id2].bldgPartEdges>
        
        Args:
            edge (building.BldgEdge): an edge that belongs to a building part
        """
        node = self.data.nodes[edge.id1]
        if node.bldgPartEdges:
            node.bldgPartEdges[0].append(edge)
        else:
            # <False> means that the node was not visited during processing.
            # It also means that the list of edges was not sorted by the angle to the horizontal axis
            node.bldgPartEdges = ([edge], False)
            
        node = self.data.nodes[edge.id2]
        if node.bldgPartEdges:
            node.bldgPartEdges[0].append(edge)
        else:
            # <False> means that the node was not visited during processing.
            # It also means that the list of edges was not sorted by the angle to the horizontal axis
            node.bldgPartEdges = ([edge], False)


class BuildingManager(BaseBuildingManager, Manager):
    
    def __init__(self, osm, app, buildingParts, layerClass):
        self.osm = osm
        Manager.__init__(self, osm)
        BaseBuildingManager.__init__(self, osm, app, buildingParts, layerClass)
        self.partPolygonsSharingBldgEdge = []
    
    def process(self):
        buildings = self.buildings
        
        for building in buildings:
            # create <building.polygon>
            building.init(self)
        
        # Iterate through building parts (building:part=*)
        # to find a building from <self.buildings> to which
        # the related <part> belongs
        
        for part in self.parts:
            
            part.init(self)
            
            if part.element.o:
                # the outline for <part> is set in an OSM relation of the type 'building'
                osmId, osmType = part.element.o
                elements = self.osm.ways if osmType is Osm.way else self.osm.relations
                if osmId in elements:
                    building = elements[osmId].b
                    if building and not part.polygon.building:
                        building.addPart(part)

        for partPolygon in self.partPolygonsSharingBldgEdge:
            self.assignBuildingToPartPolygon(partPolygon.building, partPolygon, False)
            
        if sum(1 for part in self.parts if not part.polygon.building):
            bvhTree = self.createBvhTree()
            
            for part in self.parts:
                if not part.polygon.building:
                    # <part> doesn't have a vector sharing an edge with a building footprint
                    # Take <vector> and calculated if it is located inside any building from <self.buildings>
                    coords = next(part.polygon.verts)
                    # Cast a ray from the point with horizontal coords equal to <coords> and
                    # z = -1. in the direction of <zAxis>
                    buildingIndex = bvhTree.ray_cast((coords[0], coords[1], -1.), zAxis)[2]
                    if not buildingIndex is None:
                        # we condider that <part> is located inside <buildings[buildingIndex]>
                        # Assign <part> to <buildings[buildingIndex]> and check
                        # if this part has one more neighbors
                        self.assignBuildingToPartPolygon(buildings[buildingIndex], part.polygon, True)
                        
        
        # process the building parts for each building
        for building in buildings:
            if building.parts:
                building.processParts()
        
        for action in self.actions:
            action.do(self)
    
    def createBvhTree(self):
        from mathutils.bvhtree import BVHTree
        
        vertices = []
        polygons = []
        vertexIndex1 = 0
        vertexIndex2 = 0
        
        for building in self.buildings:
            # In the case of a multipolygon we consider the only outer linestring that defines the outline
            # of the polygon
            vertices.extend(building.polygon.getVerts3d())
            vertexIndex2 = len(vertices)
            polygons.append(tuple(range(vertexIndex1, vertexIndex2)))
            vertexIndex1 = vertexIndex2
        return BVHTree.FromPolygons(vertices, polygons)
    
    def assignBuildingToPartPolygon(self, building, partPolygon, isPartIsland):
        building.addPart(partPolygon.part)
        
        # find an initial edge that isn't part of the building footprint
        for edge in partPolygon.getEdges():
            if not edge.vectors:
                break
        else:
            return
        
        self.processEdge(
            edge,
            edge.id1,
            building
        )
        self.processEdge(
            edge,
            edge.id2,
            building
        )
    
    def processEdge(self, edge, _id, building):
        edges, _visited = self.data.nodes[_id].bldgPartEdges
        if _visited:
            return
        
        edge.visited = Visited.buildingAssigned
        
        if len(edges) > 2:
            edges.sort(key = lambda edge: _pseudoangle(edge, _id))
        self.data.nodes[_id].bldgPartEdges = (edges, True)
        
        for edge in edges:
            if edge.vectors or edge.visited == Visited.buildingAssigned:
                continue
            
            partVectors12, partVectors21 = edge.partVectors12, edge.partVectors21
            
            # exclude edges that belong to part in the neighbor building
            _building = (partVectors12 or partVectors21)[0].polygon.building
            if _building and not _building is building:
                continue
            
            if partVectors12:
                for vector in partVectors12:
                    if not vector.polygon.building:
                        building.addPart(vector.polygon.part)
            
            if partVectors21:
                for vector in partVectors21:
                    if not vector.polygon.building:
                        building.addPart(vector.polygon.part)
            
            self.processEdge(
                edge,
                edge.id2 if _id==edge.id1 else edge.id1,
                building
            )


class BuildingParts:
    
    def __init__(self):
        # don't accept broken multipolygons
        self.acceptBroken = False
        self.bldgManager = None
        self.parts = []
        
    def parseWay(self, element, elementId):
        if element.closed:
            element.t = parse.polygon
            # empty outline
            element.o = None
            self.createBldgPart(element)
        else:
            element.valid = False
    
    def parseRelation(self, element, elementId):
        # empty outline
        element.o = None
        self.createBldgPart(element)
    
    def createBldgPart(self, element):
        self.parts.append(BldgPart(element))


class BuildingRelations:
    
    def parseRelation(self, element, elementId):
        # no need to store the relation in <osm.relations>, so return True
        return True