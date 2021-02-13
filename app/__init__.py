import os, math
from urllib import request

from util.polygon import Polygon


class BaseApp:
    
    osmFileName = "map%s.osm"
    
    osmUrlPath = "/api/map?bbox=%s,%s,%s,%s"
    
    def initOsm(self):
        # a data attribute to mark a building entrance
        self.buildingEntranceAttr = "entrance"
        
        # <self.logger> may be set in <setup(..)>
        self.logger = None
        
        if self.loadMissingMembers:
            self.incompleteRelations = []
            self.missingWays = set()
        
        # managers (derived from manager.Manager) performing some processing
        self.managers = []
        
        # renderers (derived from renderer.Renderer) actually making 3D objects
        self.renderers = []
        
        # tangent to check if an angle of the polygon is straight
        Polygon.straightAngleTan = math.tan(math.radians( abs(180.-self.straightAngleThreshold) ))
    
    def download(self, url, filepath, data=None):
        print("Downloading the file from %s..." % url)
        if data:
            data = data.encode('ascii')
        request.urlretrieve(url, filepath, None, data)
        print("Saving the file to %s..." % filepath)
    
    def downloadOsmFile(self, osmDir, minLon, minLat, maxLon, maxLat):
        # find a file name for the OSM file
        osmFileName = self.osmFileName % ""
        counter = 1
        while True:
            osmFilepath = os.path.realpath( os.path.join(osmDir, osmFileName) )
            if os.path.isfile(osmFilepath):
                counter += 1
                osmFileName = self.osmFileName % "_%s" % counter
            else:
                break
        self.osmFilepath = osmFilepath
        self.download(
            self.osmServer + self.osmUrlPath % (minLon, minLat, maxLon, maxLat),
            osmFilepath
        )
    
    def setAssetPackagePaths(self):
        """
        Set the following variables:
            <self.assetPackageDir>
            <self.pmlFilepath>
            <self.assetInfoFilepath>
        """
        assetPackageDir = os.path.join(self.assetsDir, self.assetPackage)
        if not os.path.isdir(assetPackageDir):
            raise Exception("The directory for the asset package %s doesn't exist" % assetPackageDir)
        self.assetPackageDir = assetPackageDir
        
        pmlFilepath = os.path.join(assetPackageDir, "style/building/main.pml")
        if not os.path.isfile(pmlFilepath):
            raise Exception("%s isn't a valid path for the PML file" % pmlFilepath)
        self.pmlFilepath = pmlFilepath
        
        assetInfoFilepath = os.path.join(assetPackageDir, "asset_info/asset_info.json")
        if self.enableExperimentalFeatures and self.importForExport:
            _assetInfoFilepath = "%s_export.json" % assetInfoFilepath[:-5]
            if os.path.isfile(_assetInfoFilepath):
                assetInfoFilepath = _assetInfoFilepath
        if not os.path.isfile(assetInfoFilepath):
            raise Exception("%s isn't a valid path for the asset info file" % assetInfoFilepath)
        self.assetInfoFilepath = assetInfoFilepath
    
    def loadSetupScript(self, setupScript):
        setupScript = os.path.realpath(setupScript)
        if not os.path.isfile(setupScript):
            raise Exception("The script file doesn't exist")
        import imp
        # remove extension from the path
        setupScript = os.path.splitext(setupScript)[0]
        moduleName = os.path.basename(setupScript)
        try:
            _file, _pathname, _description = imp.find_module(moduleName, [os.path.dirname(setupScript)])
            module = imp.load_module(moduleName, _file, _pathname, _description)
            _file.close()
            return module.setup
        except Exception as e:
            print("File \"%s\", line %s" % (e.filename, e.lineno))
            print(e.text)
            print("Error: %s", e.msg)
            raise Exception(
                "Unable to execute the setup script! See the error message in the Blender console!"
            )
    
    def process(self):
        logger = self.logger
        if logger: logger.processStart()
        
        for m in self.managers:
            m.process()
        
        if logger: logger.processEnd()