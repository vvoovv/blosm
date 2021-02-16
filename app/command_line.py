import os, argparse

from . import BaseApp
from util.transverse_mercator import TransverseMercator


class CommandLineApp(BaseApp):
    
    def __init__(self):
        super().__init__()
        
        self.osmServer = "http://overpass-api.de"
        self.loadMissingMembers = True
        self.straightAngleThreshold = 175.5
        
        self.initArgParser()
    
    def initArgParser(self):
        self.argParser = argParser = argparse.ArgumentParser()
        argParser.add_argument("--coords", help="Coordinates of the area of interest, for example -78.3105468750,39.1982386,-77.5854492188,39.6057213001")
        argParser.add_argument("--osmFilepath", help="Path to an OSM file")
        argParser.add_argument("--osmDir", help="A directory for the downloaded OSM files")
        argParser.add_argument("--setupScript", help="The path to a custom setup script")
        
        argParser.add_argument("--buildings", action='store_true', help="Import buildings")
        argParser.add_argument("--highways", action='store_true', help="Import roads and paths")
        argParser.add_argument("--railways", action='store_true', help="Import railways")
        argParser.add_argument("--water", action='store_true', help="Import water objects")
        argParser.add_argument("--forests", action='store_true', help="Import forests")
        argParser.add_argument("--vegetation", action='store_true', help="Import vegetation")
        
        args, self.unparsedArgs = argParser.parse_known_args()
        
        self.setupScript = args.setupScript
        self.osmFilepath = args.osmFilepath
        
        self.args = args
    
    def parseArgs(self):
        """
        Parse <self.unparsedArgs>
        """
        if self.unparsedArgs:
            self.argParser.parse_args(self.unparsedArgs, self.args)
        
        args = vars(self.args)
        for arg in args:
            setattr(self, arg, args[arg])
        
        if self.coords:
            self.minLon, self.minLat, self.maxLon, self.maxLat =\
                map(lambda coord: float(coord), self.coords.split(',') )
    
    def setProjection(self, lat, lon):
        self.projection = TransverseMercator(lat=lat, lon=lon)
    
    def initOsm(self):
        super().initOsm()
        if self.osmFilepath:
            self.osmFilepath = os.path.realpath(self.osmFilepath)
        else:
            self.downloadOsmFile(self.osmDir, self.minLon, self.minLat, self.maxLon, self.maxLat)
    
    def clean(self):
        self.managers = None
        self.renderers = None