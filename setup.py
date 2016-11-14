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


def setup(op, osm):
    # comment the next line if logging isn't needed
    Logger(op, osm)
    
    # create managers
    linestring = Linestring(osm)
    polygon = Polygon(osm)
    polygonAcceptBroken = PolygonAcceptBroken(osm)
    
    if op.buildings:
        if op.mode == '2D':
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
                BuildingRenderer(op, "buildings")
            )
            op.managers.append(buildings)
    
    if op.highways:
        osm.addCondition(highway, "highways", linestring)
    if op.railways:
        osm.addCondition(railway, "railways", linestring)
    if op.water:
        osm.addCondition(water, "water", polygonAcceptBroken)
        osm.addCondition(coastline, "water", linestring)
    if op.forests:
        osm.addCondition(forest, "forests", polygon)
    if op.vegetation:
        osm.addCondition(vegetation, "vegetation", polygon)
    
    numConditions = len(osm.conditions)
    if op.mode == '3D' and op.buildings:
        # 3D buildings aren't processed by BaseManager
        numConditions -= 1
    if numConditions:
        m = BaseManager(osm)
        m.setRenderer(Renderer2d(op))
        op.managers.append(m)