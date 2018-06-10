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

from parse.relation.building import Building

from building.manager import BuildingManager, BuildingParts, BuildingRelations

from manager.logging import Logger

from realistic.building.renderer import RealisticBuildingRenderer


def building(tags, e):
    return "building" in tags

def buildingPart(tags, e):
    return "building:part" in tags

def buildingRelation(tags, e):
    return isinstance(e, Building)


def setup_base(app, osm, getMaterials, bldgPreRender):
    # comment the next line if logging isn't needed
    Logger(app, osm)
    
    # the code below was under the condition: if app.buildings:
    buildingParts = BuildingParts()
    buildingRelations = BuildingRelations()
    buildings = BuildingManager(osm, buildingParts)
    
    # Important: <buildingRelation> beform <building>,
    # since there may be a tag building=* in an OSM relation of the type 'building'
    osm.addCondition(
        buildingRelation, None, buildingRelations
    )
    osm.addCondition(
        building, None, buildings
    )
    osm.addCondition(
        buildingPart, None, buildingParts
    )
    # set building renderer
    br = RealisticBuildingRenderer(
        app,
        "buildings",
        bldgPreRender = bldgPreRender,
        materials = getMaterials()
    )
    # <br> stands for "building renderer"
    buildings.setRenderer(br)
    app.managers.append(buildings)