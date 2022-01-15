#from manager import BaseManager, Linestring, Polygon, PolygonAcceptBroken
from setup import Setup
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
    
    
    setup = Setup(osm)
    
    # create managers
    
    wayManager = WayManager(osm, app)
    setup.wayManager = wayManager
    
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
        setup.skipWays()
        
        if wayClustering:
            wayManager.addRenderer(WayClusterRenderer())
            wayManager.addAction(WayClustering())
        else:
            wayManager.addRenderer(WayVisibilityRenderer(showIDs=showIDs))
    
    if app.highways:
        setup.roadsAndPaths()
    
    if app.railways:
        setup.railways()