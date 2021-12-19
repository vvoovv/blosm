#from manager import BaseManager, Linestring, Polygon, PolygonAcceptBroken
from building.manager import BaseBuildingManager
from way.manager import WayManager
from mpl.renderer.facade_classification import \
    BuildingVisibilityRender, WayVisibilityRenderer, BuildingClassificationRender, BuildingFeatureRender
from mpl.renderer import BuildingBaseRenderer
from mpl.renderer.way_cluster import WayClusterRenderer
from action.facade_visibility import FacadeVisibilityOther
from action.facade_classification import FacadeClassification
from action.feature_detection import FeatureDetection
from action.curved_features import CurvedFeatures
from action.straight_angles import StraightAngles
from action.way_clustering import WayClustering
from action.skip_features import SkipFeatures
from action.unskip_features import UnskipFeatures
from action.skip_features_again import SkipFeaturesAgain

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
    app.argParserExtra.add_argument("--classifyFacades", action='store_true', help="Display facade classification", default=False)
    app.argParserExtra.add_argument("--facadeVisibility", action='store_true', help="Display facade visibility", default=False)
    app.argParserExtra.add_argument("--sideFacadeColor", help="The color for a side facade", default="yellow")
    app.argParserExtra.add_argument("--showAssoc", action='store_true', help="Show the associations between way-segment and facade", default=False)
    app.argParserExtra.add_argument("--showIDs", action='store_true', help="Show the IDs of facades", default=False)
    app.argParserExtra.add_argument("--detectFeatures", action='store_true', help="Detect features", default=False)
    app.argParserExtra.add_argument("--showFeatures", action='store_true', help="Show detected features", default=False)
    app.argParserExtra.add_argument("--showFeatureSymbols", action='store_true', help="Show a symbol for each unskipped polygon vector. The symbol is used for pattern matching", default=False)
    app.argParserExtra.add_argument("--simplifyPolygons", action='store_true', help="Simplify polygons with the detected features", default=False)
    app.argParserExtra.add_argument("--restoreFeatures", action='store_true', help="Restore simplified features", default=False)
    app.argParserExtra.add_argument("--wayClustering", action='store_true', help="Create way clusters", default=False)
    app.argParserExtra.add_argument("--simplifyPolygonsAgain", action='store_true', help="Restore the features and simplify the polygons again", default=False)
    
    # parse the newly added command line arguments
    app.parseArgs()
    classifyFacades = getattr(app, "classifyFacades", False)
    facadeVisibility = getattr(app, "facadeVisibility", False)
    showAssoc = getattr(app, "showAssoc", False)
    showIDs = getattr(app, "showIDs", False)
    
    showFeatures = getattr(app, "showFeatures", False)
    detectFeatures = True if showFeatures else getattr(app, "detectFeatures", False)
    showFeatureSymbols = getattr(app, "showFeatureSymbols", False)
    
    simplifyPolygons = getattr(app, "simplifyPolygons", False)
    restoreFeatures = getattr(app, "restoreFeatures", False)

    wayClustering = getattr(app, "wayClustering", False)
    
    simplifyPolygonsAgain = getattr(app, "simplifyPolygonsAgain", False)
    
    # create managers
    
    wayManager = WayManager(osm, app)
    
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
                    ) if detectFeatures and showFeatures else (\
                        BuildingVisibilityRender(showAssoc=showAssoc, showIDs=showIDs) \
                            if facadeVisibility else \
                            BuildingBaseRenderer()
                    )
                )
        )
        
        # create some actions that can be reused
        skipFeaturesAction = SkipFeatures()
        unskipFeaturesAction = UnskipFeatures()
        featureDetectionAction = FeatureDetection(skipFeaturesAction if simplifyPolygons else None)
        
        
        if detectFeatures:
            buildings.addAction(CurvedFeatures())
            buildings.addAction(StraightAngles())
            buildings.addAction(featureDetectionAction)
        
        if facadeVisibility or classifyFacades:
            buildings.addAction(FacadeVisibilityOther())
        
        if classifyFacades:
            buildings.addAction(
                FacadeClassification(unskipFeaturesAction)
            )
        
        # the code below is for a test
        if simplifyPolygonsAgain:
            buildings.addAction(SkipFeaturesAgain(skipFeaturesAction, unskipFeaturesAction, featureDetectionAction))
        
        osm.addCondition(
            lambda tags, e: "building" in tags,
            "buildings",
            buildings
        )
    
    if app.highways or app.railways:
        osm.addCondition(tunnel)
        
        if wayClustering:
            wayManager.addRenderer(WayClusterRenderer())
            wayManager.addAction(WayClustering())
        else:
            wayManager.addRenderer(WayVisibilityRenderer(showIDs=showIDs))
    
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