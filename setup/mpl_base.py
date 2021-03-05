#from manager import BaseManager, Linestring, Polygon, PolygonAcceptBroken
from building.manager import BaseBuildingManager
from way.manager import RealWayManager
from mpl.renderer import BuildingRenderer, WayRenderer
from action.facade_visibility import FacadeVisibilityOther

#from manager.logging import Logger


def tunnel(tags, e):
    if tags.get("tunnel") == "yes":
        e.valid = False
        return True
    return False


def setup(app, osm):
    # comment the next line if logging isn't needed
    #Logger(app, osm)
    
    # create managers
    
    wayManager = RealWayManager(osm, app)
    wayManager.setRenderer(WayRenderer(), app)
    
    #linestring = Linestring(osm)
    #polygon = Polygon(osm)
    #polygonAcceptBroken = PolygonAcceptBroken(osm)
    
    # conditions for point objects in OSM
    #osm.addNodeCondition(
    #    lambda tags, e: tags.get("natural") == "tree",
    #    "trees",
    #    None,
    #    BaseNodeRenderer(app, path, filename, collection)
    #)
    
    if app.buildings:
        buildings = BaseBuildingManager(osm, app, None, None)
        buildings.setRenderer(BuildingRenderer())
        buildings.addAction(FacadeVisibilityOther())
        osm.addCondition(
            lambda tags, e: "building" in tags,
            "buildings", 
            buildings
        )
    
    if app.highways or app.railways:
        osm.addCondition(tunnel)
    
    if app.highways:
        osm.addCondition(
            lambda tags, e: tags.get("highway") in ("motorway", "motorway_link"),
            "roads_motorway",
            wayManager
        )
        osm.addCondition(
            lambda tags, e: tags.get("highway") in ("trunk", "trunk_link"),
            "roads_trunk",
            wayManager
        )
        osm.addCondition(
            lambda tags, e: tags.get("highway") in ("primary", "primary_link"),
            "roads_primary",
            wayManager
        )
        osm.addCondition(
            lambda tags, e: tags.get("highway") in ("secondary", "secondary_link"),
            "roads_secondary",
            wayManager
        )
        osm.addCondition(
            lambda tags, e: tags.get("highway") in ("tertiary", "tertiary_link"),
            "roads_tertiary",
            wayManager
        )
        osm.addCondition(
            lambda tags, e: tags.get("highway") == "unclassified",
            "roads_unclassified",
            wayManager
        )
        osm.addCondition(
            lambda tags, e: tags.get("highway") in ("residential", "living_street"),
            "roads_residential",
            wayManager
        )
        # footway to optimize the walk through conditions
        osm.addCondition(
            lambda tags, e: tags.get("highway") in ("footway", "path"),
            "paths_footway",
            wayManager
        )
        osm.addCondition(
            lambda tags, e: tags.get("highway") == "service",
            "roads_service",
            wayManager
        )
        osm.addCondition(
            lambda tags, e: tags.get("highway") == "pedestrian",
            "roads_pedestrian",
            wayManager
        )
        osm.addCondition(
            lambda tags, e: tags.get("highway") == "track",
            "roads_track",
            wayManager
        )
        osm.addCondition(
            lambda tags, e: tags.get("highway") == "steps",
            "paths_steps",
            wayManager
        )
        osm.addCondition(
            lambda tags, e: tags.get("highway") == "cycleway",
            "paths_cycleway",
            wayManager
        )
        osm.addCondition(
            lambda tags, e: tags.get("highway") == "bridleway",
            "paths_bridleway",
            wayManager
        )
        osm.addCondition(
            lambda tags, e: tags.get("highway") in ("road", "escape", "raceway"),
            "roads_other",
            wayManager
        )
    if app.railways:
        osm.addCondition(
            lambda tags, e: "railway" in tags,
            "railways",
            wayManager
        )
    if app.water:
        osm.addCondition(
            lambda tags, e: tags.get("natural") == "water" or tags.get("waterway") == "riverbank" or tags.get("landuse") == "reservoir",
            "water",
            polygonAcceptBroken
        )
        osm.addCondition(
            lambda tags, e: tags.get("natural") == "coastline",
            "coastlines",
            linestring
        )
    if app.forests:
        osm.addCondition(
            lambda tags, e: tags.get("natural") == "wood" or tags.get("landuse") == "forest",
            "forest",
            polygon
        )
    if app.vegetation:
        osm.addCondition(
            lambda tags, e: ("landuse" in tags and tags["landuse"] in ("grass", "meadow", "farmland")) or ("natural" in tags and tags["natural"] in ("scrub", "grassland", "heath")),
            "vegetation",
            polygon
        )