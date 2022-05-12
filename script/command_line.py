from parse.osm import Osm
from app.command_line import CommandLineApp


def importData():
    
    a = CommandLineApp()
    
    #try:
    a.initOsm()
    #except Exception as e:
    #    print("Error")
    #    return
    
    forceExtentCalculation = bool(a.osmFilepath)
    
    setupScript = a.setupScript
    if setupScript:
        setup_function = a.loadSetupScript(setupScript)
        if not setup_function:
            return
    else:
        from setup.base import setup as setup_function
    
    osm = Osm(a)
    
    setup_function(a, osm)
    
    a.createLayers(osm)
    
    if not a.osmFilepath and a.coords:
        osm.setProjection( (a.minLat+a.maxLat)/2., (a.minLon+a.maxLon)/2. )
    
    osm.parse(a.osmFilepath, forceExtentCalculation=forceExtentCalculation)
    if a.loadMissingMembers and a.incompleteRelations:
        try:
            a.loadMissingWays(osm)
        except Exception as e:
            print(str(e))
            a.loadMissingMembers = False
        a.processIncompleteRelations(osm)
        if not osm.projection:
            # <osm.projection> wasn't set so far if there were only incomplete relations that
            # satisfy <osm.conditions>.
            # See also the comments in <parse.osm.__init__.py>
            # at the end of the method <osm.parse(..)>
            osm.setProjection( (osm.minLat+osm.maxLat)/2., (osm.minLon+osm.maxLon)/2. )
    
    if forceExtentCalculation:
        a.minLat = osm.minLat
        a.maxLat = osm.maxLat
        a.minLon = osm.minLon
        a.maxLon = osm.maxLon
    
    a.initLayers()
    
    a.process()
    
    if a.overlayType:
        importOverlay(a)
    
    a.render()
    
    a.clean()


def importOverlay(a):
    import matplotlib.pyplot as plt
    
    overlay = a.overlay
    
    overlay.prepareImport(a.minLon, a.minLat, a.maxLon, a.maxLat)
    
    hasTiles = True
    while hasTiles:
        hasTiles = overlay.importNextTile()
    if overlay.finalizeImport():
        print("Overlay import is finished!")
    else:
        print("Probably something is wrong with the tile server!")
        return
    
    plt.imshow(overlay.imageData, extent=overlay.fromTileCoordsToAppCoords())


importData()