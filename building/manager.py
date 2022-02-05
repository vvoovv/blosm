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


def _assignBuildingToPartPolygon(building, partPolygon):
    partPolygon.building = building
    building.parts.append(partPolygon.part)
    
    
    for vector in partPolygon.getVectors():
        edge = vector.edge
        if not edge.vectors:
            if edge.partVectors12:
                for _vector in edge.partVectors12:
                    if not _vector is vector and not _vector.polygon.building:
                        _assignBuildingToPartPolygon(building, _vector.polygon)
            if edge.partVectors21:
                for _vector in edge.partVectors21:
                    if not _vector is vector and not _vector.polygon.building:
                        _assignBuildingToPartPolygon(building, _vector.polygon)


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


class BuildingManager(BaseBuildingManager, Manager):
    
    def __init__(self, osm, app, buildingParts, layerClass):
        self.osm = osm
        Manager.__init__(self, osm)
        BaseBuildingManager.__init__(self, osm, app, buildingParts, layerClass)
        self.partPolygonsSharingBldgEdge = []
    
    def process(self):
        BaseBuildingManager.process(self)
        # create a Python set for each OSM node to store indices of the entries from <self.buildings>,
        # where instances of the wrapper class <building.manager.Building> are stored
        buildings = self.buildings
        
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
                    if building:
                        part.polygon.building = building
                        building.parts.append(part)

        for partPolygon in self.partPolygonsSharingBldgEdge:
            _assignBuildingToPartPolygon(partPolygon.building, partPolygon)
            
        if sum(1 for part in self.parts if not part.polygon.building):
            bvhTree = self.createBvhTree()
            
            for part in self.parts:
                if not part.polygon.building:
                    for vector in part.polygon.getVectors():
                        # <part> doesn't have a vector sharing an edge with a building footprint
                        # Take <vector> and calculated if it is located inside any building from <self.buildings>
                        coords = vector.v1
                        # Cast a ray from the point with horizontal coords equal to <coords> and
                        # z = -1. in the direction of <zAxis>
                        buildingIndex = bvhTree.ray_cast((coords[0], coords[1], -1.), zAxis)[2]
                        if not buildingIndex is None:
                            # we condider that <part> is located inside <buildings[buildingIndex]>
                            part.polygon.building = buildings[buildingIndex]
                            buildings[buildingIndex].parts.append(part)
        
        # process the building parts for each building
        for building in buildings:
            if building.parts:
                building.processParts()
    
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