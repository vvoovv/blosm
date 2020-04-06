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
import json
from .features import Polygon, Multipolygon, Node


class GeoJson:
    """
    Representation of data in an OSM file
    """
    
    def __init__(self, app):
        self.app = app
        
        self.projection = None
        
        self.nodes = []
        # a list of polygons: polygon is a closed polygonal chain without self-intersections and holes
        self.polygons = []
        # a list multipolygons: multipolygon is a polygon with one or more holes
        self.multipolygons = []
        
        # skip a feature without properties
        self.skipNoProperties = True
        
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
        self.forceExtentCalculation = kwargs.get("forceExtentCalculation")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)["features"]
        
        # <self.projection> could be set during a previous call of <self.parse(..)>
        if not self.projection:
            self.projection = self.app.projection
        if not self.projection or self.forceExtentCalculation:
            self.minLat = 90.
            self.maxLat = -90.
            self.minLon = 180.
            self.maxLon = -180.
        
        for f in data: # f stands for feature
            tags = f.get("properties")
            self.processTags(tags)
            if not tags and self.skipNoProperties:
                continue
            geometry = f.get("geometry")
            if not geometry:
                continue
            coords = geometry["coordinates"]
            if not coords:
                continue
            # t stands for type
            t = geometry.get("type")
            if t == "Polygon":
                if len(coords) == 1:
                    self.processPolygon(coords[0], tags)
                else: # len(coords) > 1
                    self.processMultipolygon(coords, tags)
            elif t == "MultiPolygon":
                for c in coords:
                    if len(c) == 1:
                        self.processPolygon(c[0], tags)
                    else: # len(c) > 1
                        self.processMultipolygon(c, tags)
            elif t == "Node":
                node = Node(coords, tags)
                condition = self.checkNodeConditions(tags, node)
                if condition:
                    self.processCondition(condition, node, None, self.parseNode)
                    self.nodes.append(node)
        
        if not self.projection:
            # set projection using the calculated bounds (self.minLat, self.maxLat, self.minLon, self.maxLon)
            lat = (self.minLat + self.maxLat)/2.
            lon = (self.minLon + self.maxLon)/2.
            self.setProjection(lat, lon)
    
    def processTags(self, tags):
        return
    
    def processPolygon(self, coords, tags):
        a = self.app
        if a.coordinatesAsFilter:
            for c in coords:
                if a.minLon <= c[0] <= a.maxLon and a.minLat <= c[1] <= a.maxLat:
                    # At least one point of the multipolygon is located
                    # within the extent <a.minLon, a.minLat, a.maxLon, a.maxLat>,
                    # so keep the multipolygon
                    break
            else:
                # No point of the polygon is within the extent <a.minLon, a.minLat, a.maxLon, a.maxLat>,
                # so skip the feature
                return
        polygon = Polygon(coords, tags)
        skip = self.processFeature(polygon, tags, self.parsePolygon)
        if not skip:
            self.polygons.append(polygon)
    
    def processMultipolygon(self, coords, tags):
        a = self.app
        if a.coordinatesAsFilter:
            # only the outer ring matters, that's way we use <coords[0]>
            for c in coords[0]:
                if a.minLon <= c[0] <= a.maxLon and a.minLat <= c[1] <= a.maxLat:
                    # At least one point of the multipolygon is located
                    # within the extent <a.minLon, a.minLat, a.maxLon, a.maxLat>,
                    # so keep the multipolygon
                    break
            else:
                # No point of the polygon is within the extent <a.minLon, a.minLat, a.maxLon, a.maxLat>,
                # so skip the feature
                return
        multipolygon = Multipolygon(coords, tags)
        skip = self.processFeature(multipolygon, tags, self.parseMultipolygon)
        if not skip:
            # the line below is needed in order for <ls> property to work
            multipolygon.geojson = self
            self.multipolygons.append(multipolygon)
    
    def processFeature(self, feature, tags, parseElement):
        """
        Common stuff for all features
        """
        # do we need to skip the OSM <way> from storing in <self.ways>
        skip = False
        condition = self.checkConditions(tags, feature)
        if condition:
            skip = self.processCondition(condition, feature, None, parseElement)
            if (not self.projection or self.forceExtentCalculation) and feature.valid:
                feature.updateBounds(self)
        return skip

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
    def parsePolygon(manager, element, elementId):
        return manager.parsePolygon(element, elementId)
    
    @staticmethod
    def parseMultipolygon(manager, element, elementId):
        return manager.parseMultipolygon(element, elementId)
    
    def setProjection(self, lat, lon):
        self.lat = lat
        self.lon = lon
        self.app.setProjection(lat, lon)
        self.projection = self.app.projection
    