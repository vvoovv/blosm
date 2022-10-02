from setup import SetupBlender as Setup

from manager.logging import Logger

from setup.premium import setup_forests


def setup(app, osm):
    setup = Setup(app, osm)
    
    classifyFacades = True
    
    # comment the next line if logging isn't needed
    Logger(app, osm)
    
    # Important the stuff for roadways and railways goes before the one for buildings.
    # Reason: instances of way.Way must be already initialized when actions for the building manager
    # are processed
    
    if app.highways or app.railways:
        setup.skipWays()
        
        from action.generate_streets import StreetGenerator
        from way.renderer_streets import StreetRenderer
        wayManager = setup.getWayManager()
        wayManager.addAction(StreetGenerator())
        wayManager.addRenderer(StreetRenderer(app))
    
        if app.highways:
            setup.roadsAndPaths()
        
        if app.railways:
            setup.railways()
        
        if app.forests:
            setup_forests(app, osm)
    
    if app.buildings:
        setup.buildingsRealistic(getStyle=getStyle)
        
        setup.detectFeatures(simplifyPolygons=True)
        
        if classifyFacades:
            setup.classifyFacades()


def getStyle(building, app):
    #return "mid rise apartments zaandam"
    #return "high rise mirrored glass"
    buildingTag = building["building"]
    
    if buildingTag in ("commercial", "office"):
        return "high rise"
    
    if buildingTag in ("house", "detached"):
        return "single family house"
    
    if buildingTag in ("residential", "apartments", "house", "detached"):
        return "residential"
    
    if building["amenity"] == "place_of_worship":
        return "place of worship"
    
    if building["man_made"] or building["barrier"] or buildingTag=="wall":
        return "man made"
    
    buildingArea = building.polygon.area()
    
    if buildingArea < 20.:
        return "small structure"
    elif buildingArea < 200.:
        return "single family house"
    
    return "high rise"