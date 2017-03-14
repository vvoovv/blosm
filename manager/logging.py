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

from datetime import datetime
from building.manager import BuildingManager


class Logger:
    
    def __init__(self, app, osm):
        self.parseStartTime = datetime.now()
        app.logger = self
        self.app = app
        self.osm = osm
        print("Parsing OSM file %s..." % app.osmFilepath)
    
    def processStart(self):
        print("Time for parsing OSM file: %s" % (datetime.now() - self.parseStartTime))
        self.processStartTime = datetime.now()
        print("Processing the parsed OSM data...")

    def processEnd(self):
        self.numBuildings()
        print("Time for processing of the parsed OSM data: %s" % (datetime.now() - self.processStartTime))
    
    def renderStart(self):
        self.renderStartTime = datetime.now()
        print("Creating meshes in Blender...")

    def renderEnd(self):
        t = datetime.now()
        print("Time for mesh creation in Blender: %s" % (t - self.renderStartTime))
        print("Total duration: %s" % (t - self.parseStartTime))
    
    def numBuildings(self):
        app = self.app
        if not (app.mode == '3D' and app.buildings):
            return
        for m in app.managers:
            if isinstance(m, BuildingManager):
                print("The number of buildings: %s" % len(m.buildings))