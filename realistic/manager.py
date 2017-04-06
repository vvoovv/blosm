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

from manager import BaseManager


class AreaManager(BaseManager):
    
    def __init__(self, osm, app, areaRenderer):
        """
        Args:
            osm (parse.Osm): Parsed OSM data
        """
        super().__init__(osm)
        self.app = app
        self.areaRenderer = areaRenderer
    
    def renderExtra(self):
        self.areaRenderer.render(self.app)