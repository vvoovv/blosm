import os, argparse

from . import BaseApp
from util.transverse_mercator import TransverseMercator


class Layer:
    
    def __init__(self, layerId, app):
        self.app = app
        self.id = layerId
    
    def init(self):
        pass


class CommandLineApp(BaseApp):
    
    def __init__(self):
        super().__init__()
        
        self.projection = None
        
        self.osmServer = "http://overpass-api.de"
        self.loadMissingMembers = True
        self.straightAngleThreshold = 175.5
        
        # default layer class used in <self.createLayer(..)>
        self.layerClass = Layer
        # default node layer class used in <self.createLayer(..)>
        self.nodeLayerClass = Layer
        
        self.initArgParser()
        
        self.mode = BaseApp.twoD
    
    def initArgParser(self):
        self.argParser = argParser = argparse.ArgumentParser()
        # a parser for <self.unparsedArgs>
        self.argParserExtra = argparse.ArgumentParser()
        
        argParser.add_argument("--coords", help="Coordinates of the area of interest, for example -78.3105468750,39.1982386,-77.5854492188,39.6057213001")
        argParser.add_argument("--osmFilepath", help="Path to an OSM file")
        argParser.add_argument("--osmDir", help="A directory for the downloaded OSM files")
        argParser.add_argument("--setupScript", help="The path to a custom setup script")
        
        argParser.add_argument("--buildings", action='store_true', help="Import buildings", default=False)
        argParser.add_argument("--highways", action='store_true', help="Import roads and paths", default=False)
        argParser.add_argument("--railways", action='store_true', help="Import railways", default=False)
        argParser.add_argument("--water", action='store_true', help="Import water objects", default=False)
        argParser.add_argument("--forests", action='store_true', help="Import forests", default=False)
        argParser.add_argument("--vegetation", action='store_true', help="Import vegetation", default=False)
        
        # arguments for an overlay
        argParser.add_argument("--overlayDir", help="A directory for overlay images")
        argParser.add_argument("--overlayType", help="Type of an overlay. Possible values: arcgis-satellite, mapbox-satellite, osm-mapnik, mapbox-streets, custom")
        argParser.add_argument("--overlayUrl", help="URL template for the custom overlay type")
        argParser.add_argument("--overlayAccessToken", help="An access token for some overlay providers (ArcGIS, Mapbox)")
        argParser.add_argument("--maxNumTiles", type=int, default=256, help="Maximum number of overlay tiles")
        argParser.add_argument("--showOverlayOnly", action='store_true', help="Show overlay only", default=False)
        
        
        args, self.unparsedArgs = argParser.parse_known_args()
        args = vars(args)
        for arg in args:
            setattr(self, arg, args[arg])

        if self.coords:
            self.minLon, self.minLat, self.maxLon, self.maxLat =\
                map(lambda coord: float(coord), self.coords.split(',') )
        
        if self.overlayType:
            self.initOverlay()
    
    def parseArgs(self):
        """
        Parse <self.unparsedArgs>
        """
        if self.unparsedArgs:
            args = vars( self.argParserExtra.parse_args(self.unparsedArgs) )
            for arg in args:
                setattr(self, arg, args[arg])
    
    def setProjection(self, lat, lon):
        self.projection = TransverseMercator(lat=lat, lon=lon)
    
    def initOsm(self):
        super().initOsm()
        if self.osmFilepath:
            self.osmFilepath = os.path.realpath(self.osmFilepath)
        else:
            self.downloadOsmFile(self.osmDir, self.minLon, self.minLat, self.maxLon, self.maxLat)

    def render(self):
        logger = self.logger
        if logger: logger.renderStart()
        
        for r in self.renderers:
            r.prepare()
        
        for m in self.managers:
            m.render()
        
        for r in self.renderers:
            r.finalize()
            r.cleanup()
        
        if logger: logger.renderEnd()
    
    def clean(self):
        self.managers = None
        self.renderers = None
    
    def initOverlay(self):
        from overlay.command_line import OverlayMixin
        
        overlay = self.setOverlay(OverlayMixin)
        overlay.originAtTop = True
        if overlay.imageExtension in ('jpg', 'jpeg'):
            # jpg-images don't have the opacity component
            overlay.numComponents = 3
        overlay.overlayDir = os.path.join( self.overlayDir, overlay.getOverlaySubDir() )
    
    def getArcgisAccessToken(self):
        return self.overlayAccessToken
    
    def getMapboxAccessToken(self):
        return self.overlayAccessToken
    
    def print(self, value):
        print(value)