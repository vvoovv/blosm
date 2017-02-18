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

class Way:
    """
    A class to represent an OSM way
    
    Some attributes:
        l (app.Layer): layer used to place the related geometry to a specific Blender object
        t: type for rendering (Render.polygon, Render.linestring)
        tags (dict): OSM tags
        m: A manager used during the rendering, if None <manager.BaseManager> applies defaults
            during the rendering
        r (bool): Defines if we need to render (True) or not (False) the OSM way
            in the function <manager.BaseManager.render(..)>
        rr: A special renderer for the OSM way
        used (bool): Defines if the OSM way is already used (True) for bounds calculation or not (False)
        closed (bool): Defines if the OSM way forms a closed (True) or open (False) linestring
        n (bool): The number of OSM nodes in the OSM way; if the OSM way is closed,
            the number is decreased by 1
        o (tuple): Defines the related outline for a building part (building:part=*);
            has the form (osmId, osmElement); set only in the case of 3D buildings
        b (building.manager.Building): A related 3D Building; set only for 3D buildings
        o (tuple): Defines the related outline for a building part (building:part=*);
            has the form (osmId, osmElement); set only in the case of 3D buildings
    """
    
    __slots__ = ("l", "t", "tags", "nodes", "m", "r", "rr", "used", "n", "closed", "valid", "b", "o")
    
    def __init__(self, nodes, tags, osm):
        self.nodes = nodes
        self.tags = tags
        self.r = False
        self.rr = None
        self.used = False
        self.validate(osm)
    
    def isClosed(self):
        return self.closed
    
    def validate(self, osm):
        self.closed = False
        valid = True
        numNodes = len(self.nodes)
        if numNodes < 2:
            valid = False
        else:
            if self.nodes[0] == self.nodes[-1]:
                if numNodes == 3 or numNodes == 2:
                    valid = False
                else:
                    self.closed = True
                    numNodes -= 1
        if valid:
            # check that all related nodes are defined in the OSM file
            for i in range(numNodes):
                if not self.nodes[i] in osm.nodes:
                    valid = False
                    break
            self.n = numNodes
        self.valid = valid
    
    def getData(self, osm):
        """
        Get projected data for the way
        
        Returns a Python generator
        """
        return (osm.nodes[self.nodes[i]].getData(osm) for i in range(self.n))
    
    def nodeIds(self, osm):
        """
        A generator to get id of OSM nodes of the way
        
        Returns a Python generator
        """
        return (self.nodes[i] for i in range(self.n))