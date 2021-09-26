#from manager import BaseManager, Linestring, Polygon, PolygonAcceptBroken
from building.manager import BaseBuildingManager
from way.manager import WayManager
from mpl.renderer.facade_classification import \
    BuildingVisibilityRender, WayVisibilityRenderer, BuildingClassificationRender, BuildingFeatureRender
from action.facade_visibility import FacadeVisibilityOther
from action.facade_classification import FacadeClassification
from action.feature_detection import FeatureDetection
from action.curved_features import CurvedFeatures
from action.straight_angles import StraightAngles
from action.road_clustering import RoadClustering

#from manager.logging import Logger


def tunnel(tags, e):
    if tags.get("tunnel") == "yes":
        e.valid = False
        return True
    return False


def setup(app, osm):
    # comment the next line if logging isn't needed
    #Logger(app, osm)
    
    # add the definition of the custom command line arguments
    app.argParserExtra.add_argument("--classification", action='store_true', help="Display facade classification", default=False)
    app.argParserExtra.add_argument("--sideFacadeColor", help="The color for a side facade", default="yellow")
    app.argParserExtra.add_argument("--showAssoc", action='store_true', help="Show the associations between way-segment and facade", default=False)
    app.argParserExtra.add_argument("--showIDs", action='store_true', help="Show the IDs of facades", default=False)
    app.argParserExtra.add_argument("--detectFeatures", action='store_true', help="Detect features", default=False)
    app.argParserExtra.add_argument("--showFeatures", action='store_true', help="Show detected features", default=False)
    app.argParserExtra.add_argument("--showFeatureSymbols", action='store_true', help="Show a symbol for each unskipped polygon vector. The symbol is used for pattern matching", default=False)
    app.argParserExtra.add_argument("--simplifyPolygons", action='store_true', help="Simplify polygons with the detected features", default=False)
    app.argParserExtra.add_argument("--restoreFeatures", action='store_true', help="Restore simplified features", default=False)
    app.argParserExtra.add_argument("--roadClustering", action='store_true', help="Create road clusters", default=False)
    
    # parse the newly added command line arguments
    app.parseArgs()
    classifyFacades = getattr(app, "classification", False)
    showAssoc = getattr(app, "showAssoc", False)
    showIDs = getattr(app, "showIDs", False)
    
    showFeatures = getattr(app, "showFeatures", False)
    detectFeatures = True if showFeatures else getattr(app, "detectFeatures", False)
    showFeatureSymbols = getattr(app, "showFeatureSymbols", False)
    
    simplifyPolygons = getattr(app, "simplifyPolygons", False)
    restoreFeatures = getattr(app, "restoreFeatures", False)

    roadClustering = getattr(app, "roadClustering", False)
    # create managers
    
    wayManager = WayManager(osm, app)
    wayManager.setRenderer(WayVisibilityRenderer(showIDs=showIDs), app)
    
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
        buildings.setRenderer(
            BuildingClassificationRender(sideFacadeColor=app.sideFacadeColor, showAssoc=showAssoc,showIDs=showIDs)\
                if classifyFacades else (\
                    BuildingFeatureRender(
                        restoreFeatures=restoreFeatures,
                        showFeatureSymbols=showFeatureSymbols,
                        showIDs=showIDs
                    ) if detectFeatures and showFeatures else\
                        BuildingVisibilityRender(showAssoc=showAssoc, showIDs=showIDs)
                )
        )
        
        if detectFeatures:
            buildings.addAction(CurvedFeatures())
            buildings.addAction(StraightAngles())
            #buildings.addAction(FeatureDetection(simplifyPolygons))
        #buildings.addAction(FacadeVisibilityOther())
        if classifyFacades:
            buildings.addAction(FacadeClassification())

        if roadClustering:
            buildings.addAction(RoadClustering())
        
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
        # filter out pedestrian areas for now
        osm.addCondition(
            lambda tags, e: tags.get("highway") == "pedestrian" and not tags.get("area") and not tags.get("area:highway"),
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