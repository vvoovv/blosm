"""
This file is a part of Blosm addon for Blender.
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
            # <None> means that the node was not visited during processing.
            # It also means that the list of edges was not sorted by the angle to the horizontal axis
            node.bldgPartEdges = ([edge], None)
            
        node = self.data.nodes[edge.id2]
        if node.bldgPartEdges:
            node.bldgPartEdges[0].append(edge)
        else:
            # <None> means that the node was not visited during processing.
            # It also means that the list of edges was not sorted by the angle to the horizontal axis
            node.bldgPartEdges = ([edge], None)


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
            self.processBldgPartEdges(partPolygon.building, partPolygon, True)
            
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
                        self.processBldgPartEdges(buildings[buildingIndex], part.polygon, False)
        
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
    
    def processBldgPartEdges(self, building, partPolygon, createMissingParts):
        """
        Walk along edges of building parts, assign building to the part polygons
        """
        building.addPart(partPolygon.part)
        
        # find an initial edge that isn't part of the building footprint
        for edge in partPolygon.getEdges():
            if not edge.vectors:
                break
        else:
            return
        
        # start from <edge> and walk in the direction <edge.id1>
        self.processBldgPartEdge(
            edge,
            edge.id1,
            building,
            createMissingParts
        )
        # now walk in the direction <edge.id2>
        self.processBldgPartEdge(
            edge,
            edge.id2,
            building,
            createMissingParts
        )
    
    def processBldgPartEdge(self, edge, _id, building, createMissingParts):
        edges, visited = self.data.nodes[_id].bldgPartEdges
        if visited:
            return
        
        # mark <edge> as visited
        edge.visited = Visited.buildingAssigned
        
        # If <visited> is <None>, it means that <edges> were not sorted in <self.traceMissingPart(..)>
        if len(edges) > 2 and visited is None:
            # sort the edges by pseudo-angle
            edges.sort(key = lambda edge: _pseudoangle(edge, _id))
        # mark <bldgPartEdges> as visited
        self.data.nodes[_id].bldgPartEdges = (edges, True)
        
        isId1 = _id==edge.id1
        
        if createMissingParts:
            if (isId1 and edge.partVectors21 is None) or (not isId1 and edge.partVectors12 is None):
                # trace the candidate for a missing building part
                self.traceMissingPart(building, edge, _id)
            elif (isId1 and edge.partVectors12 is None) or (not isId1 and edge.partVectors21 is None):
                # trace the candidate for a missing building part
                self.traceMissingPart(building, edge, edge.id2 if isId1 else edge.id1)
        
        # process each <_edge> in <edges>
        for _edge in edges:
            # The condition <_edge is edge> is already presented below
            # through the condition <_edge.visited == Visited.buildingAssigned>
            # Edges shared with <building> footprint aren't considered
            # since building parts composed of those edges are already assigned to <building>
            if _edge.vectors or _edge.visited == Visited.buildingAssigned:
                continue
            
            partVectors12, partVectors21 = _edge.partVectors12, _edge.partVectors21
            
            _building = (partVectors12 or partVectors21)[0].polygon.building
            # exclude edges that belong to a part in the neighbor building
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
            
            self.processBldgPartEdge(
                _edge,
                _edge.id2 if isId1 else _edge.id1,
                building,
                createMissingParts
            )
    
    def traceMissingPart(self, building, edgeStart, _id):
        """
        Trace the candidate for a missing building part
        
        Args:
            building (building.Building): A building
            edgeStart (building.BldgEdge): The edge to start tracing from
            _id (str): The id of the edge's node gives the direction of tracing
        """
        _edge = edgeStart
        
        if _edge.visited == Visited.noMissingPart:
            return
        else:
            # mark the starting edge as visited to avoid tracing the same sequence of edges again and again
            _edge.visited = Visited.noMissingPart
        
        nodes = [_id]
        # The current vector along the footprint of the whole building if tracing is needed along
        # the building footprint.
        # <bldgVector> also serves as a switch between two modes in the <while> cycle:
        # 1) tracing along the buiding footprint
        # 2) tracing along building parts
        bldgVector = None
        while True:
            if bldgVector:
                bldgVector = bldgVector.next
                _edge = bldgVector.edge
                if (bldgVector.direct and _edge.partVectors12) or (not bldgVector.direct and _edge.partVectors21):
                    # reached a building part
                    _id = bldgVector.id1
                    # switch the mode to tracing along building parts
                    bldgVector = None
                else:
                    # continue tracing along the building footprint
                    nodes.append(bldgVector.id2)
            else:
                edges, visited = self.data.nodes[_id].bldgPartEdges
                numEdges = len(edges)
                if numEdges > 2:
                    if visited is None:
                        # sort the edges by angle around the node <_id>
                        edges.sort(key = lambda edge: _pseudoangle(edge, _id))
                        # <False> means that the edges were sorted but not yet visited
                        self.data.nodes[_id].bldgPartEdges = (edges, False)
                    
                    # use the neighbor edge to the left from <_edge>
                    _edge = edges[ edges.index(_edge) - 1 ]
                else:
                    # use the neighbor edge
                    _edge = edges[1] if edges[0] is _edge else edges[0]
                
                if _edge is edgeStart:
                    building.addMissingPart(nodes, self)
                    return
                
                if _edge.visited == Visited.noMissingPart:
                    return
                else:
                    # # mark <_edge> as visited to avoid tracing the same sequence of edges again and again
                    _edge.visited = Visited.noMissingPart
                
                # check if <_edge> takes part in the building footprint
                bldgVectors = _edge.vectors
                if bldgVectors:
                    # If there are two buildings using <_edge> choose the building vector
                    # belonging to <building>
                    # Then jump to the next building vector
                    # <bldgVector> is not <None> now. It means that we switched to mode
                    # of tracing along the building footprint
                    bldgVector = (
                        bldgVectors[0] if len(bldgVectors) == 1 else (
                            bldgVectors[0] if bldgVectors[0].polygon.building is building else bldgVectors[1]
                        )
                    ).next
                    # check validity of the edge
                    if (bldgVector.direct and bldgVector.edge.partVectors12) or\
                            (not bldgVector.direct and bldgVector.edge.partVectors21) or\
                            "building:part" in building.element.tags:
                        return
                    # continue tracing along the building footprint
                    nodes.append(bldgVector.id2)
                else:
                    # check validity of the edge
                    if (_edge.partVectors12 and _edge.partVectors21) or\
                            (_id == _edge.id1 and _edge.partVectors12) or\
                            (_id == _edge.id2 and _edge.partVectors21):
                        return
                    
                    # continue tracing along building parts
                    _id = _edge.id2 if _id == _edge.id1 else _edge.id1
                    nodes.append(_id)


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