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

class Relation:
    """
    A class to represent an OSM relation
    
    Some attributes:
        l (app.Layer): layer index used to place the related geometry to a specific Blender object
        tags (dict): OSM tags
        m: A manager used during the rendering, if None <manager.BaseManager> applies defaults
            during the rendering
        r (bool): Defines if we need to render (True) or not (False) the OSM relation
            in the function <manager.BaseManager.render(..)>
        rr: A special renderer for the OSM relation
    """
    
    # use __slots__ for memory optimization
    __slots__ = ("l", "tags", "m", "r", "rr", "valid")
    
    def __init__(self):
        self.r = False
        self.rr = None
        self.valid = True