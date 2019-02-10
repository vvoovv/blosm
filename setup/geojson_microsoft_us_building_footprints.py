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

from geojson import Manager, BuildingManager
from renderer import Renderer2d
from building.renderer import BuildingRenderer

from manager.logging import Logger




def setup(app, data):
    # comment the next line if logging isn't needed
    Logger(app, data)
    
    data.skipNoProperties = False
    
    manager = Manager(data)
    
    if app.buildings:
        if app.mode is app.twoD:
            data.addCondition(
                lambda tags, e: True,
                "buildings", 
                manager
            )
        else: # 3D
            # no building parts for the moment
            buildings = BuildingManager(data, None)
            
            data.addCondition(
                lambda tags, e: True,
                "buildings",
                buildings
            )
            #osm.addCondition(
            #    lambda tags, e: "building:part" in tags,
            #    None,
            #    buildingParts
            #)
            buildings.setRenderer(
                BuildingRenderer(app)
            )
            app.managers.append(buildings)
    
    numConditions = len(data.conditions)
    if not app.mode is app.twoD and app.buildings:
        # 3D buildings aren't processed by BaseManager
        numConditions -= 1
    if numConditions:
        manager.setRenderer(Renderer2d(app))
        app.managers.append(manager)