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

from util.osm import parseNumber


class Building:
    """
    A wrapper for a OSM building
    """
    def __init__(self, element):
        self.element = element
        self.parts = []
    
    def addPart(self, part):
        self.parts.append(part)

    def markUsedNodes(self, buildingIndex, osm):
        """
        For each OSM node of <self.element> (OSM way or OSM relation) add the related
        <buildingIndex> (i.e. the index of <self> in Python list <buildings> of an instance
        of <BuildingManager>) to Python set <b> of the node 
        """
        for nodeId in self.element.nodeIds(osm):
            osm.nodes[nodeId].b.add(buildingIndex)
    
    def getHeight(self, element):
        return parseNumber(element.tags["height"]) if "height" in element.tags else None
    
    def getRoofMinHeight(self, element, app):
        # getting the number of levels
        h = element.tags.get("building:levels")
        if not h is None:
            h = parseNumber(h)
        if h is None:
            h = app.defaultNumLevels
        h *= app.levelHeight
        return h
    
    def getMinHeight(self, element, app):
        tags = element.tags
        if "min_height" in tags:
            z0 = parseNumber(tags["min_height"], 0.)
        elif "building:min_level" in tags:
            numLevels = parseNumber(tags["building:min_level"])
            z0 = 0. if numLevels is None else numLevels * app.levelHeight
        else:
            z0 = 0.
        return z0