#from manager import BaseManager, Linestring, Polygon, PolygonAcceptBroken
from setup import Setup
from building.manager import BaseBuildingManager
from mpl.renderer.facade_classification import \
    BuildingVisibilityRender, WayVisibilityRenderer, BuildingClassificationRender, BuildingFeatureRender
from mpl.renderer import BuildingBaseRenderer
from action.facade_visibility import FacadeVisibilityOther
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
    app.argParserExtra.add_argument("--generateStreets", action='store_true', help="Generate and show streets and intersections", default=False)
    app.argParserExtra.add_argument("--debugStreetRendering", action='store_true', help="debug in StreetRenderer", default=False)
    app.argParserExtra.add_argument("--assetsDir", help="Path to a folder with assets and PML styles", default='')
    
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
    
    simplifyPolygonsAgain = getattr(app, "simplifyPolygonsAgain", False)
    
    generateStreets = getattr(app, "generateStreets", False)
    debugStreetRendering = getattr(app,"debugStreetRendering",False)
    
    
    app.setAssetPackagePaths()
    
    setup = Setup(app, osm)
    
    # create managers
    
    wayManager = setup.getWayManager()
    
    if app.buildings:
        buildings = setup.buildingManager = BaseBuildingManager(osm, app, None, None)
        
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
                
        if detectFeatures:
            setup.detectFeatures(simplifyPolygons)
        
        if facadeVisibility:
            setup.facadeVisibility(FacadeVisibilityOther())
        
        if classifyFacades:
            setup.classifyFacades(FacadeVisibilityOther())
        
        # the code below is for a test
        if simplifyPolygonsAgain:
            buildings.addAction(
                SkipFeaturesAgain(
                    setup.getSkipFeaturesAction(),
                    setup.getUnskipFeaturesAction(),
                    setup.getFeatureDetectionAction(simplifyPolygons)
                )
            )
        
        setup.buildings()
    
    if app.highways or app.railways:
        setup.skipWays()
        if generateStreets:
            from action.generate_streets import StreetGenerator
            from mpl.renderer.streets import StreetRenderer
            from style import StyleStore
            from setup.realistic_streets import getStyleStreet
            
            styleStore = StyleStore(app.pmlFilepathStreet, app.assetsDir, styles=None)
            
            wayManager.addAction(StreetGenerator(styleStore, getStyle=getStyleStreet))
            wayManager.addRenderer(StreetRenderer(debug=debugStreetRendering))
        else:
            wayManager.addRenderer(WayVisibilityRenderer(showIDs=showIDs))
    
    if app.highways:
        setup.roadsAndPaths()
    
    if app.railways:
        setup.railways()