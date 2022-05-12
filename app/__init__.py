import os, math
from urllib import request

from building import BldgPolygon


class BaseApp:
    
    # app mode
    twoD = 1
    simple = 2
    realistic = 3
    
    osmFileName = "map%s.osm"
    
    osmFileExtraName = "%s_extra.osm"
    
    # request to the Overpass server to get both ways and their nodes for the given way ids
    overpassWays = "((way(%s););node(w););out;"
    
    osmUrlPath = "/api/map?bbox=%s,%s,%s,%s"
    
    osmUrlPath2 = "/api/interpreter"
    
    def initOsm(self):
        self.baseInit()
        
        # a data attribute to mark a building entrance
        self.buildingEntranceAttr = "entrance"
        
        # <self.logger> may be set in <setup(..)>
        self.logger = None
        
        if self.loadMissingMembers:
            self.incompleteRelations = []
            self.missingWays = set()
    
    def baseInit(self):
        # managers (derived from manager.Manager) performing some processing
        self.managers = []
        
        self.managersById = {}
        
        # renderers (derived from renderer.Renderer) actually making 3D objects
        self.renderers = []
        
        self.layerIndices = {}
        self.layers = []
        self.layerKwargs = {}
        
        # tangent to check if an angle of the polygon is straight
        BldgPolygon.straightAngleSin = math.sin(math.radians( abs(180.-self.straightAngleThreshold) ))
    
    def download(self, url, filepath, data=None):
        print("Downloading the file from %s..." % url)
        if data:
            data = data.encode('ascii')
        request.urlretrieve(url, filepath, None, data)
        print("Saving the file to %s..." % filepath)
    
    def downloadOsmFile(self, osmDir, minLon, minLat, maxLon, maxLat):
        # find a file name for the OSM file
        osmFileName = BaseApp.osmFileName % ""
        counter = 1
        while True:
            osmFilepath = os.path.realpath( os.path.join(osmDir, osmFileName) )
            if os.path.isfile(osmFilepath):
                counter += 1
                osmFileName = BaseApp.osmFileName % "_%s" % counter
            else:
                break
        self.osmFilepath = osmFilepath
        self.download(
            self.osmServer + BaseApp.osmUrlPath % (minLon, minLat, maxLon, maxLat),
            osmFilepath
        )
    
    def downloadOsmWays(self, ways, filepath):
        self.download(
            self.osmServer + BaseApp.osmUrlPath2,
            filepath,
            BaseApp.overpassWays % ");way(".join(ways)
        )
    
    def loadMissingWays(self, osm):
        filepath = BaseApp.osmFileExtraName % self.osmFilepath[:-4]
        if not os.path.isfile(filepath):
            print("Downloading data for incomplete OSM relations")
            self.downloadOsmWays(self.missingWays, filepath)
        self.loadMissingMembers = False
        print("Parsing and processing data from the file %s for incomplete OSM relations" % filepath)
        osm.parse(filepath)
    
    def processIncompleteRelations(self, osm):
        """
        Download missing OSM ways with ids stored in <self.missingWays>,
        add them to <osm.ways>. Process incomplete relations stored in <self.incompleteRelations>
        """
        for relation, _id, members, tags, condition in self.incompleteRelations:
            # below there is the same code for a relation as in osm.parse(..)
            relation.process(members, tags, osm)
            if relation.valid:
                skip = osm.processCondition(condition, relation, _id, osm.parseRelation)
                if not _id in osm.relations and not skip:
                    osm.relations[_id] = relation
        # cleanup
        self.incompleteRelations = None
        self.missingWays = None
    
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
        except Exception:
            raise Exception(
                "Unable to execute the setup script! See the error message in the Blender console!"
            )
    
    def process(self):
        logger = self.logger
        if logger: logger.processStart()
        
        for m in self.managers:
            m.process()
        
        if logger: logger.processEnd()
        
    def createLayers(self, osm):
        layerIndices = self.layerIndices
        
        if osm.conditions:
            # go through <osm.conditions> to fill <layerIndices> and <self.layers> with values
            for c in osm.conditions:
                manager = c[1]
                layerId = c[3]
                if layerId and not layerId in layerIndices:
                    if manager and manager.layerClass:
                        manager.createLayer(
                            layerId,
                            self,
                            **self.layerKwargs
                        )
                    else:  
                        self.createLayer(
                            layerId,
                            self.layerClass,
                            **self.layerKwargs
                        )
            # Replace <osm.conditions> with new entries
            # where <layerId> is replaced by the related instance of <Layer>
            osm.conditions = tuple(
                (c[0], c[1], c[2], None if c[3] is None else self.getLayer(c[3])) \
                for c in osm.conditions
            )
        
        # the same for <osm.nodeConditions> 
        if osm.nodeConditions:
            # go through <osm.conditions> to fill <layerIndices> and <self.layers> with values
            for c in osm.nodeConditions:
                manager = c[1]
                layerId = c[3]
                if layerId and not layerId in layerIndices:
                    if manager:
                        manager.createNodeLayer(
                            layerId,
                            self,
                            **self.layerKwargs
                        )
                    else:  
                        self.createLayer(
                            layerId,
                            self.nodeLayerClass,
                            **self.layerKwargs
                        )
            # Replace <osm.nodeConditions> with new entries
            # where <layerId> is replaced by the related instance of <Layer>
            osm.nodeConditions = tuple(
                (c[0], c[1], c[2], None if c[3] is None else self.getLayer(c[3])) \
                for c in osm.nodeConditions
            )

    def initLayers(self):
        for layer in self.layers:
            layer.init()
    
    def getLayer(self, layerId):
        layerIndex = self.layerIndices.get(layerId)
        return None if layerIndex is None else self.layers[layerIndex] 
    
    def createLayer(self, layerId, layerClass, **kwargs):
        layer = layerClass(layerId, self)
        for k in kwargs:
            setattr(layer, k, kwargs[k])
        self.layerIndices[layerId] = len(self.layers)
        self.layers.append(layer)
        return layer

    def addRenderer(self, renderer):
        if not renderer in self.renderers:
            self.renderers.append(renderer)
            
    def addManager(self, manager):
        if manager.id:
            self.managersById[manager.id] = manager
        self.managers.append(manager)
    
    def setOverlay(self, OverlayMixin):
        from types import MethodType
        from overlay import overlayTypeData
        
        data = overlayTypeData[self.overlayType]
        
        overlay = data[0](
            self.overlayUrl if self.overlayType == "custom" else data[1],
            data[2],
            self
        )
        
        for methodName in OverlayMixin.__dict__:
            if not methodName.startswith('_'):
                setattr(
                    overlay,
                    methodName,
                    MethodType( OverlayMixin.__dict__[methodName], overlay)
                )
        self.overlay = overlay
        
        return overlay