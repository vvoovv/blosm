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

import xml.etree.cElementTree as etree
import inspect, importlib

from .node import Node
from .way import Way


class Osm:
    """
    Representation of data in an OSM file
    """
    
    # OSM data type
    node = 1
    way = 2
    relation = 3
    # <types> is set in the constructor
    types = None
    
    # <relationTypes> is set in the constructor
    relationTypes = None
    
    # roles of members of an OSM relation
    outer = 4
    inner = 5
    outline = 6
    part = 7
    # <roles> is set in the constructor
    roles = None
    
    def __init__(self, app):
        self.app = app
        
        self.projection = None
        
        if not Osm.types:
            # initialize dictionaries
            Osm.types = {'node':Osm.node, 'way':Osm.way, 'relation':Osm.relation}
            Osm.relationTypes = {'multipolygon': Multipolygon, 'building': Building}
            Osm.roles = {'outer':Osm.outer, 'inner':Osm.inner, 'outline':Osm.outline, 'part':Osm.part}
        
        self.nodes = {}
        self.ways = {}
        self.relations = {}
        
        # the following Python list contains OSM nodes to be rendered
        self.rNodes = []
        
        self.conditions = []
        # use separate conditions for nodes to get some performance gain
        self.nodeConditions = []
    
    def addCondition(self, condition, layerId=None, manager=None, renderer=None):
        self.conditions.append(
            (condition, manager, renderer, layerId)
        )
    
    def addNodeCondition(self, condition, layerId=None, manager=None, renderer=None):
        self.nodeConditions.append(
            (condition, manager, renderer, layerId)
        )
    
    def parse(self, filepath, **kwargs):
        forceExtentCalculation = kwargs.get("forceExtentCalculation")
        
        osm = etree.parse(filepath).getroot()
        
        # <self.projection> could be set during a previous call of <self.parse(..)>
        if not self.projection:
            self.projection = self.app.projection
        if not self.projection or forceExtentCalculation:
            self.minLat = 90.
            self.maxLat = -90.
            self.minLon = 180.
            self.maxLon = -180.
        
        relations = self.relations
        
        for e in osm: # e stands for element
            attrs = e.attrib
            if "action" in attrs and attrs["action"] == "delete": continue
            if e.tag == "node":
                _id = attrs["id"]
                tags = None
                for c in e:
                    if c.tag == "tag":
                        if not tags:
                            tags = {}
                        tags[c.get("k")] = c.get("v")
                node = Node(float(attrs["lat"]), float(attrs["lon"]), tags)
                if tags:
                    condition = self.checkNodeConditions(tags, node)
                    if condition:
                        self.processCondition(condition, node, _id, self.parseNode)
                        # set <node> for rendering by appending it to <self.rNodes>
                        self.rNodes.append(node)
                self.nodes[_id] = node
            elif e.tag == "way":
                _id = attrs["id"]
                nodes = []
                tags = None
                for c in e:
                    if c.tag == "nd":
                        nodes.append(c.get("ref"))
                    elif c.tag == "tag":
                        if not tags:
                            tags = {}
                        tags[c.get("k")] = c.get("v")
                way = Way(nodes, tags, self)
                if way.valid:
                    # do we need to skip the OSM <way> from storing in <self.ways>
                    skip = False
                    if tags:
                        tags["id"] = _id
                        condition = self.checkConditions(tags, way)
                        if condition:
                            skip = self.processCondition(condition, way, _id, self.parseWay)
                            if (not self.projection or forceExtentCalculation) and way.valid:
                                self.updateBounds(way)
                    if not skip:
                        self.ways[_id] = way
            elif e.tag == "relation":
                _id = attrs["id"]
                members = []
                tags = None
                relation = None
                for c in e:
                    if c.tag == "member":
                        mType = Osm.types.get( c.get("type") )
                        mId = c.get("ref")
                        mRole = Osm.roles.get( c.get("role") )
                        if not (mType and mId):
                            continue
                        members.append((mType, mId, mRole))
                    elif c.tag == "tag":
                        if not tags:
                            tags = {}
                        k = c.get("k")
                        v = c.get("v")
                        tags[k] = v
                        if k == "type":
                            relation = Osm.relationTypes.get(v)
                # skip the relation without tags
                if relation and tags:
                    tags["id"] = _id
                    createdBefore = _id in relations
                    if createdBefore:
                        # The empty OSM relation was created before,
                        # since it's referenced by another OSM relation
                        relation = relations[_id]
                    else:
                        relation = relation(self)
                    if relation.valid:
                        condition = self.checkConditions(tags, relation)
                        if condition:
                            complete = relation.process(members, tags, self)
                            if complete:
                                if relation.valid:
                                    skip = self.processCondition(condition, relation, _id, self.parseRelation)
                                    if not createdBefore and not skip:
                                        relations[_id] = relation
                            else:
                                self.app.incompleteRelations.append((relation, _id, members, tags, condition))
            elif e.tag == "bounds":
                # If <self.projection> isn't set,
                # it means we need to set <self.projection> here,
                # using <bounds> from the OSM file
                if not self.projection:
                    # also set the area extent
                    self.minLat = float(attrs["minlat"])
                    self.maxLat = float(attrs["maxlat"])
                    self.minLon = float(attrs["minlon"])
                    self.maxLon = float(attrs["maxlon"])
                    lat = ( self.minLat + self.maxLat )/2.
                    lon = ( self.minLon + self.maxLon )/2.
                    self.setProjection(lat, lon)
                    forceExtentCalculation = False
        
        if not self.projection:
            # set projection using the calculated bounds (self.minLat, self.maxLat, self.minLon, self.maxLon)
            lat = (self.minLat + self.maxLat)/2.
            lon = (self.minLon + self.maxLon)/2.
            self.setProjection(lat, lon)
    
    def checkConditions(self, tags, element):
        for c in self.conditions:
            if c[0](tags, element):
                # setting manager
                element.m = c[1]
                return c

    def checkNodeConditions(self, tags, element):
        for c in self.nodeConditions:
            if c[0](tags, element):
                # setting manager
                element.m = c[1]
                return c
    
    def processCondition(self, condition, element, elementId, parseElement):
        # do we need to skip the OSM <element> from storing in <self.ways> or <self.relations>
        skip = False
        # always set <layer>
        # layer = condition[3]
        element.l = condition[3]
        # check if we have a special manager for the element
        manager = condition[1]
        if manager:
            parseElement(manager, element, elementId)
        # check if wee need to set a special renderer
        if condition[2]:
            # renderer = condition[2]
            element.rr = condition[2]
        return skip
    
    @staticmethod
    def parseNode(manager, element, elementId):
        return manager.parseNode(element, elementId)
    
    @staticmethod
    def parseWay(manager, element, elementId):
        return manager.parseWay(element, elementId)
    
    @staticmethod
    def parseRelation(manager, element, elementId):
        return manager.parseRelation(element, elementId)
    
    def updateBounds(self, way):
        # <way> has been already used for bounds calculation
        if way.used:
            return
        for i in range(way.n):
            node = self.nodes[way.nodes[i]]
            lat = node.lat
            lon = node.lon
            if lat<self.minLat:
                self.minLat = lat
            elif lat>self.maxLat:
                self.maxLat = lat
            if lon<self.minLon:
                self.minLon = lon
            elif lon>self.maxLon:
                self.maxLon = lon
        # mark <way> as used for bounds calculation
        way.used = True

    def getRelation(self, _id, relationClass):
        """
        Returns a relation with <_id> if it has been already encountered in the OSM file or
        creates a relation of <relationClass> with no parameters and sets it to <selfs.relations[_id]>
        """
        relations = self.relations
        relation = relations.get(_id)
        if not relation:
            relation = relationClass()
            relations[_id] = relation
        return relation
    
    def setProjection(self, lat, lon):
        self.lat = lat
        self.lon = lon
        self.app.setProjection(lat, lon)
        self.projection = self.app.projection


from .relation.multipolygon import Multipolygon
from .relation.building import Building