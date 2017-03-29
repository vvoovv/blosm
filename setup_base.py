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

from parse.relation.building import Building

from manager import BaseManager, Linestring, Polygon, PolygonAcceptBroken
from renderer import Renderer2d

from building.manager import BuildingManager, BuildingParts, BuildingRelations
from building.renderer import BuildingRenderer

from manager.logging import Logger


def building(tags, e):
    return "building" in tags

def buildingPart(tags, e):
    return "building:part" in tags

def buildingRelation(tags, e):
    return isinstance(e, Building)

def highway(tags, e):
    return "highway" in tags

def railway(tags, e):
    return "railway" in tags

def water(tags, e):
    return tags.get("natural") == "water" or\
        tags.get("waterway") == "riverbank" or\
        tags.get("landuse") == "reservoir"

def coastline(tags, e):
    return tags.get("natural") == "coastline" 

def forest(tags, e):
    return tags.get("natural") == "wood" or tags.get("landuse") == "forest"

def vegetation(tags, e):
    return ( "landuse" in tags and tags["landuse"] in ("grass", "meadow", "farmland") ) or\
        ( "natural" in tags and tags["natural"] in ("scrub", "grassland", "heath") )


def setup(app, osm):
    # comment the next line if logging isn't needed
    Logger(app, osm)
    
    # create managers
    linestring = Linestring(osm)
    polygon = Polygon(osm)
    polygonAcceptBroken = PolygonAcceptBroken(osm)
    
    if app.buildings:
        if app.mode == '2D':
            osm.addCondition(building, "buildings", polygon)
        else: # 3D
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
            buildings.setRenderer(
                BuildingRenderer(app, "buildings")
            )
            app.managers.append(buildings)
    
    if app.highways:
        osm.addCondition(highway, "highways", linestring)
    if app.railways:
        osm.addCondition(railway, "railways", linestring)
    if app.water:
        osm.addCondition(water, "water", polygonAcceptBroken)
        osm.addCondition(coastline, "water", linestring)
    if app.forests:
        osm.addCondition(forest, "forests", polygon)
    if app.vegetation:
        osm.addCondition(vegetation, "vegetation", polygon)
    
    numConditions = len(osm.conditions)
    if app.mode == '3D' and app.buildings:
        # 3D buildings aren't processed by BaseManager
        numConditions -= 1
    if numConditions:
        m = BaseManager(osm)
        m.setRenderer(Renderer2d(app))
        app.managers.append(m)