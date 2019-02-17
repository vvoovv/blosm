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

class Node:
    """
    A class to represent an OSM node
    
    Some attributes:
        l (app.Layer): layer used to place the related geometry to a specific Blender object
        tags (dict): OSM tags
        m: A manager used during the rendering; if None, <manager.BaseManager> applies defaults
            during the rendering
        rr: A special renderer for the OSM node
        b (dict): Here we store building indices (i.e. the indices of instances of
            the wrapper class <building.manager.Building> in Python list <buildings> of an instance
            of <building.manager.BuildingManager>)
        f (dict): Here we store footprint indices of building parts (i.e. the indices of instances of
            the wrapper class <building.manager.Building> in Python list <buildings> of an instance
            of <building.manager.BuildingManager>)
        rr: A renderer for the OSM node
    """
    __slots__ = ("l", "tags", "lat", "lon", "coords", "b", "f", "m", "rr", "valid")
    
    def __init__(self, lat, lon, tags):
        self.tags = tags
        self.lat = lat
        self.lon = lon
        self.b = dict()
        self.f = dict()
        # projected coordinates
        self.coords = None
        self.rr = None
        self.valid = True
    
    def getData(self, osm):
        """
        Get projected coordinates
        """
        if not self.coords:
            # preserve coordinates in the local system of reference for a future use
            self.coords = osm.projection.fromGeographic(self.lat, self.lon)
        return self.coords