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

from manager import Linestring, Polygon, PolygonAcceptBroken
from renderer import Renderer2d
from realistic.manager import AreaManager
from realistic.renderer import AreaRenderer, ForestRenderer, WaterRenderer, BareRockRenderer

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

#
# "grass", "meadow", "grassland", "farmland"
#
def grass(tags, e):
    return tags.get("landuse") == "grass"
def meadow(tags, e):
    return tags.get("landuse") == "meadow"
def grassland(tags, e):
    return tags.get("natural") == "grassland"
def farmland(tags, e):
    return tags.get("landuse") == "farmland"

#
# "scrub", "heath"
#
def scrub(tags, e):
    return tags.get("natural") == "scrub"
def heath(tags, e):
    return tags.get("natural") == "heath"

#
# "marsh", "reedbed", "bog", "swamp"
#
def marsh(tags, e):
    return tags.get("natural") == "wetland" and tags.get("wetland") == "marsh"
def reedbed(tags, e):
    return tags.get("natural") == "wetland" and tags.get("wetland") == "reedbed"
def bog(tags, e):
    return tags.get("natural") == "wetland" and tags.get("wetland") == "bog"
def swamp(tags, e):
    return tags.get("natural") == "wetland" and tags.get("wetland") == "swamp"

#
# "glacier"
#
def glacier(tags, e):
    return tags.get("natural") == "glacier"

#
# "bare_rock:
#
def bare_rock(tags, e):
    return tags.get("natural") == "bare_rock"

#
# "scree", "shingle"
# "beach" with "gravel" or "pebbles"
#
def scree(tags, e):
    return tags.get("natural") == "scree"
def shingle(tags, e):
    natural = tags.get("natural")
    return natural == "shingle" or (natural == "beach" and tags.get("surface") in ("gravel", "pebbles"))

#
# "sand"
# "beach" with "sand"
#
def sand(tags, e):
    natural = tags.get("natural")
    # The condition is added after shingle(..),
    # so any value of <surface> for <natural=beach> or its absence is
    # considered as <surface=sand>
    return natural == "sand" or natural == "beach"


def setup(app, osm):
    # comment the next line if logging isn't needed
    Logger(app, osm)
    
    areaRenderers = {}
    
    # create managers
    linestring = Linestring(osm)
    polygon = Polygon(osm)
    polygonAcceptBroken = PolygonAcceptBroken(osm)
    
    if app.buildings:
        if app.mode is app.twoD:
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
        osm.addCondition(coastline, "coastlines", linestring)
        areaRenderers["water"] = WaterRenderer()
    if app.forests:
        osm.addCondition(forest, "forest", polygon)
        areaRenderers["forest"] = ForestRenderer()
    if app.vegetation:
        # "grass", "meadow", "grassland", "farmland"
        osm.addCondition(grass, "grass", polygon)
        osm.addCondition(meadow, "meadow", polygon)
        osm.addCondition(grassland, "grassland", polygon)
        osm.addCondition(farmland, "farmland", polygon)
        # "scrub", "heath"
        osm.addCondition(scrub, "scrub", polygon)
        osm.addCondition(heath, "heath", polygon)
        # "marsh", "reedbed", "bog", "swamp"
        osm.addCondition(marsh, "marsh", polygon)
        osm.addCondition(reedbed, "reedbed", polygon)
        osm.addCondition(bog, "bog", polygon)
        osm.addCondition(swamp, "swamp", polygon)
    #if app.otherAreas:
    osm.addCondition(glacier, "glacier", polygon)
    if False:
        osm.addCondition(bare_rock, "bare_rock", polygon)
        areaRenderers["bare_rock"] = BareRockRenderer()
    
    osm.addCondition(scree, "scree", polygon)
    osm.addCondition(shingle, "shingle", polygon)
    osm.addCondition(sand, "sand", polygon)
        
    
    numConditions = len(osm.conditions)
    if not app.mode is app.twoD and app.buildings:
        # 3D buildings aren't processed by BaseManager
        numConditions -= 1
    if numConditions:
        m = AreaManager(osm, app, AreaRenderer(), **areaRenderers)
        m.setRenderer(Renderer2d(app, applyMaterial=False))
        app.managers.append(m)