"""
This file is part of blender-osm (OpenStreetMap importer for Blender).
Copyright (C) 2014-2020 Vladimir Elistratov, Alain (al1brn)
prokitektura+support@gmail.com

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import os, json, webbrowser, math, gzip, struct
from urllib import request
import bpy
from mathutils import Vector

import defs
from renderer.layer import Layer
from renderer.node_layer import NodeLayer
from renderer import Renderer
from terrain import Terrain
from util.blender import makeActive
from util.polygon import Polygon

_isBlender280 = bpy.app.version[1] >= 80


class App:
    
    #layerIds = ["buildings", "highways", "railways", "water", "forests", "vegetation"]
    
    layerIds = [
        "building", "highway", "railway",
        "water", "forest",
        "grass", "meadow", "grassland", "farmland"
        "scrub", "heath",
        "marsh", "reedbed", "bog", "swamp",
        "glacier",
        "bare_rock",
        "scree", "shingle",
        "sand",
        "beach" # sand, gravel, pebbles
    ]
    
    osmServers = {
        "overpass-api.de": "http://overpass-api.de",
        "openstreetmap.ru": "http://overpass.openstreetmap.ru",
        "openstreetmap.fr": "http://overpass.openstreetmap.fr",
        "kumi.systems": "http://overpass.kumi.systems"
    }
    
    devOsmServer = "overpass-api.de"
    
    osmUrlPath = "/api/map?bbox=%s,%s,%s,%s"
    
    osmUrlPath2 = "/api/interpreter"
    
    terrainUrl = "http://s3.amazonaws.com/elevation-tiles-prod/skadi/%s/%s"
    
    osmDir = "osm"
    
    terrainSubDir = "terrain"
    
    overlaySubDir = "overlay"
    
    osmFileName = "map%s.osm"
    
    osmFileExtraName = "%s_extra.osm"
    
    # request to the Overpass server to get both ways and their nodes for the given way ids
    overpassWays = "((way(%s););node(w););out;"
    
    bldgMaterialsFileName = "building_materials.blend"
    
    vegetationFileName = "vegetation.blend"
    
    # app mode
    twoD = 1
    simple = 2
    realistic = 3
    
    layerOffsets = {
        "buildings": 0.2,
        "water": 0.2,
        "forest": 0.1,
        "vegetation": 0.
    }
    
    # default color (asphalt)
    defaultColor = (0.0865, 0.090842, 0.088656)
    
    # diffuse colors for some layers
    colors = {
        "buildings": (0.309, 0.013, 0.012),
        "water": (0.009, 0.002, 0.8),
        "forest": (0.02, 0.208, 0.007),
        "vegetation": (0.007, 0.558, 0.005),
        
        "roads_track": (0.564712, 0.332452, 0.066626),
        
        "railways": (0.2, 0.2, 0.2)
    }
    
    # Default value for <offset> parameter of the SHRINKWRAP modifier;
    # it's used to project flat meshes onto a terrain
    swOffset = 0.05
    
    # default z-coordinate of Blender objects (curves) representing OSM ways
    wayZ = 0.3
    
    # Default value for <offset> parameter of the SHRINKWRAP modifier;
    # it's used to project OSM ways represented as Blender curves onto a terrain
    swWayOffset = 0.3
    
    # Value for <offset> parameter of the SHRINKWRAP modifier in the realistic mode
    # to ensure correct results for dynamic paint. <dp> stands for dynamic paint.
    swOffsetDp = 50.
    
    voidValue = -32768
    voidSubstitution = 0
    
    def __init__(self):
        # path to the top directory of the addon
        self.basePath = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            os.pardir
        )
        # a cache for the license keys
        self._keys = {}
        # fill in the cache <self._keys> for the license keys
        for licenseKey in (getattr(defs.Keys, attr) for attr in dir(defs.Keys) if not attr.startswith("__")):
            self.has(licenseKey)
        
        self.version = None
        self.isPremium = False
        self.layerIndices = {}
        self.layers = []
    
    def initOsm(self, op, context):
        addonName = self.addonName
        addon = context.scene.blender_osm
        prefs = context.preferences.addons if _isBlender280 else context.user_preferences.addons
        
        if app.has(defs.Keys.mode3d) and self.mode != "2D":
            self.mode = App.realistic\
                if app.has(defs.Keys.mode3dRealistic) and self.mode == "3Drealistic"\
                else App.simple
        else:
            self.mode = App.twoD
            
        if self.mode is App.realistic:
            _setAssetsDirStr = "Please set a directory with assets (building_materials.blend, vegetation.blend) in the addon preferences or addon GUI!"
            # first try the <assetsDir> from the addon GUI
            assetsDir = self.assetsDir
            if not assetsDir:
                # second try the <assetsDir> from the addon preferences
                assetsDir = prefs[addonName].preferences.assetsDir if addonName in prefs else None
            if assetsDir:
                assetsDir = os.path.realpath(bpy.path.abspath(assetsDir))
                if not os.path.isdir(assetsDir):
                    raise Exception(
                        "The directory with assets %s doesn't exist. " % assetsDir +
                        _setAssetsDirStr
                    )
                self.assetsDir = assetsDir
                bldgMaterialsFilepath = os.path.join(assetsDir, self.bldgMaterialsFileName)
                if not os.path.isfile(bldgMaterialsFilepath):
                    raise Exception(
                        "The directory with assets %s doesn't contain the file %s. " % (assetsDir, self.bldgMaterialsFileName) +
                        _setAssetsDirStr
                    )
                self.bldgMaterialsFilepath = bldgMaterialsFilepath
                if self.forests:
                    vegetationFilepath = os.path.join(assetsDir, self.vegetationFileName)
                    if not os.path.isfile(vegetationFilepath):
                        raise Exception(
                            "The directory with assets %s doesn't contain the file %s. " % (assetsDir, self.vegetationFileName) +
                            _setAssetsDirStr
                        )
                    self.vegetationFilepath = vegetationFilepath
            else:
                raise Exception(_setAssetsDirStr)
        
        basePath = self.basePath
        self.op = op
        self.assetPath = os.path.join(basePath, "assets")
        self.setDataDir(context, basePath, addonName)
        # create a sub-directory under <self.dataDir> for OSM files
        osmDir = os.path.join(self.dataDir, self.osmDir)
        if not os.path.exists(osmDir):
            os.makedirs(osmDir)
        
        # <self.logger> may be set in <setup(..)>
        self.logger = None
        # a Python dict to cache Blender meshes loaded from Blender files serving as an asset library
        self.meshes = {}
        
        self.layerIndices = {}
        self.layers = []
        
        self.osmServer = self.osmServers[
            prefs[addonName].preferences.osmServer if addonName in prefs else self.devOsmServer
        ]
        
        if addon.osmSource == "server":
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
                self.osmServer + self.osmUrlPath % (app.minLon, app.minLat, app.maxLon, app.maxLat),
                osmFilepath
            )
        else:
            self.osmFilepath = os.path.realpath(bpy.path.abspath(self.osmFilepath))
        
        if self.loadMissingMembers:
            self.incompleteRelations = []
            self.missingWays = set()
        
        # managers (derived from manager.Manager) performing some processing
        self.managers = []
        
        # renderers (derived from renderer.Renderer) actually making 3D objects
        self.renderers = []
        
        # tangent to check if an angle of the polygon is straight
        Polygon.straightAngleTan = math.tan(math.radians( abs(180.-self.straightAngleThreshold) ))

        # dealing with export to the popular 3D formats
        if self.mode is App.realistic:
            self.enableExperimentalFeatures = prefs[addonName].preferences.enableExperimentalFeatures if addonName in prefs else True
        else:
            self.enableExperimentalFeatures = False
        
        # we ignore building entrances if a file path to the PML style file isn't given
        self.buildingEntranceAttr = None
        if self.enableExperimentalFeatures and self.mode is App.realistic:
            pmlFilepath = self.pmlFilepath
            if pmlFilepath:
                pmlFilepath = os.path.realpath(bpy.path.abspath(pmlFilepath))
                if not os.path.isfile(pmlFilepath):
                    raise Exception("%s isn't a valid path for the PML file" % pmlFilepath)
                self.pmlFilepath = pmlFilepath
                # We set <self.buildingEntranceAttr> only if a custom PML file is given. That
                # must be changed later.
                self.buildingEntranceAttr = "entrance"
            
            assetInfoFilepath = self.assetInfoFilepath
            if assetInfoFilepath:
                assetInfoFilepath = os.path.realpath(bpy.path.abspath(assetInfoFilepath))
                if not os.path.isfile(assetInfoFilepath):
                    raise Exception("%s isn't a valid path for the asset info file" % assetInfoFilepath)
                self.assetInfoFilepath = assetInfoFilepath
    
    def setTerrain(self, context, createFlatTerrain=True, createBvhTree=False):
        addon = context.scene.blender_osm
        
        terrainObjectName =\
            addon.terrainObject\
            if context.scene.objects.get(addon.terrainObject.strip()) else\
            (
                Terrain.createFlatTerrain(
                    self.minLon, self.minLat, self.maxLon, self.maxLat,
                    self.projection,
                    context
                )
                if createFlatTerrain else None
            )
        
        # check if have a terrain Blender object set
        terrain = Terrain(terrainObjectName, context)
        self.terrain = terrain if terrain.terrain else None
        if self.terrain:
            terrain.init(createBvhTree)
    
    def initTerrain(self, context):
        addonName = self.addonName
        self.setDataDir(context, self.basePath, addonName)
        # create a sub-directory under <self.dataDir> for OSM files
        terrainDir = os.path.join(self.dataDir, self.terrainSubDir)
        self.terrainDir = terrainDir
        if not os.path.exists(terrainDir):
            os.makedirs(terrainDir)
        
        self.terrainSize = 3600//int(self.terrainResolution)
        
        # we are going from top to down, that's why we call reversed()
        self.latIntervals = tuple(reversed(Terrain.getHgtIntervals(self.minLat, self.maxLat)))
        self.lonIntervals = Terrain.getHgtIntervals(self.minLon, self.maxLon)

        missingHgtFiles = self.getMissingHgtFiles()
        # download missing .hgt files
        for missingPath in missingHgtFiles:
            missingFile = os.path.basename(missingPath)
            self.download(
                self.terrainUrl % (missingFile[:3], missingFile),
                missingPath
            )
    
    def initOverlay(self, context):
        addonName = self.addonName
        from overlay import Overlay, overlayTypeData
        addon = context.scene.blender_osm
        data = overlayTypeData[addon.overlayType]
        
        # <addonName> can be used by some classes derived from <Overlay>
        # to access addon settings
        overlay = data[0](
            addon.overlayUrl if addon.overlayType == "custom" else data[1],
            data[2],
            addonName
        )
        self.overlay = overlay
        
        self.setDataDir(context, self.basePath, addonName)
        # create a sub-directory under <self.dataDir> for overlay tiles
        j = os.path.join
        overlayDir = j( j(self.dataDir, self.overlaySubDir), overlay.getOverlaySubDir() )
        if not os.path.exists(overlayDir):
            os.makedirs(overlayDir)
        overlay.overlayDir = overlayDir
        
        self.setTerrain(context, createFlatTerrain=True, createBvhTree=False)
    
    def initGpx(self, context, addonName):
        gpxFilepath = os.path.realpath(bpy.path.abspath(self.gpxFilepath))
        if not os.path.isfile(gpxFilepath):
            raise Exception("A valid GPX file isn't set")
        self.gpxFilepath = gpxFilepath
    
    def initGeoJson(self, op, context):
        prefs = context.preferences.addons if _isBlender280 else context.user_preferences.addons
        
        if app.has(defs.Keys.mode3d) and self.mode != "2D":
            self.mode = App.realistic\
                if app.has(defs.Keys.mode3dRealistic) and self.mode == "3Drealistic"\
                else App.simple
        else:
            self.mode = App.twoD
            
        if self.mode is App.realistic:
            _setAssetsDirStr = "Please set a directory with assets (building_materials.blend, vegetation.blend) in the addon preferences or addon GUI!"
            # first try the <assetsDir> from the addon GUI
            assetsDir = self.assetsDir
            if not assetsDir:
                # second try the <assetsDir> from the addon preferences
                assetsDir = prefs[addonName].preferences.assetsDir if addonName in prefs else None
            if assetsDir:
                assetsDir = os.path.realpath(bpy.path.abspath(assetsDir))
                if not os.path.isdir(assetsDir):
                    raise Exception(
                        "The directory with assets %s doesn't exist. " % assetsDir +
                        _setAssetsDirStr
                    )
                self.assetsDir = assetsDir
                bldgMaterialsFilepath = os.path.join(assetsDir, self.bldgMaterialsFileName)
                if not os.path.isfile(bldgMaterialsFilepath):
                    raise Exception(
                        "The directory with assets %s doesn't contain the file %s. " % (assetsDir, self.bldgMaterialsFileName) +
                        _setAssetsDirStr
                    )
                self.bldgMaterialsFilepath = bldgMaterialsFilepath
                if self.forests:
                    vegetationFilepath = os.path.join(assetsDir, self.vegetationFileName)
                    if not os.path.isfile(vegetationFilepath):
                        raise Exception(
                            "The directory with assets %s doesn't contain the file %s. " % (assetsDir, self.vegetationFileName) +
                            _setAssetsDirStr
                        )
                    self.vegetationFilepath = vegetationFilepath
            else:
                raise Exception(_setAssetsDirStr)
        
        basePath = self.basePath
        self.op = op
        self.assetPath = os.path.join(basePath, "assets")
        
        # <self.logger> may be set in <setup(..)>
        self.logger = None
        # a Python dict to cache Blender meshes loaded from Blender files serving as an asset library
        self.meshes = {}
        
        self.layerIndices = {}
        self.layers = []
        
        self.osmFilepath = os.path.realpath(bpy.path.abspath(self.osmFilepath))
        
        # managers (derived from manager.Manager) performing some processing
        self.managers = []
        
        # renderers (derived from renderer.Renderer) actually making 3D objects
        self.renderers = []
        
        # tangent to check if an angle of the polygon is straight
        Polygon.straightAngleTan = math.tan(math.radians( abs(180.-self.straightAngleThreshold) ))
    
    def setAttributes(self, context):
        """
        Copies properties from <context.scene.blender_osm>
        """
        addon = context.scene.blender_osm
        for p in dir(addon):
            # don't know why <int> started to appear in <dir(addon)>
            if not (p.startswith("__") or p in ("bl_rna", "rna_type", "int", "string")):
                setattr(self, p, getattr(addon, p))
    
    def setDataDir(self, context, basePath, addonName):
        """
        Sets <self.dataDir>, i.e. path to data
        """
        prefs = context.preferences.addons if _isBlender280 else context.user_preferences.addons
        j = os.path.join
        if addonName in prefs:
            dataDir = prefs[addonName].preferences.dataDir
            if not dataDir:
                raise Exception("A valid directory for data in the addon preferences isn't set")
            dataDir = os.path.realpath(bpy.path.abspath(dataDir))
            if not os.path.isdir(dataDir):
                raise Exception("The directory for data in the addon preferences doesn't exist")
            self.dataDir = dataDir
        else:
            # set <self.dataDir> to basePath/../../../data (development version)
            self.dataDir = os.path.realpath( j( j( j( j(basePath, os.pardir), os.pardir), os.pardir), "data") )
    
    def createLayers(self, osm):
        layerIndices = self.layerIndices
        kwargs = dict(swOffset=self.swOffsetDp) if self.mode is App.realistic else {}
        
        if osm.conditions:
            # go through <osm.conditions> to fill <layerIndices> and <self.layers> with values
            for c in osm.conditions:
                manager = c[1]
                layerId = c[3]
                if layerId and not layerId in layerIndices:
                    if manager:
                        manager.createLayer(
                            layerId,
                            self,
                            **kwargs
                        )
                    else:  
                        self.createLayer(
                            layerId,
                            Layer,
                            **kwargs
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
                            **kwargs
                        )
                    else:  
                        self.createLayer(
                            layerId,
                            NodeLayer,
                            **kwargs
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
    
    def process(self):
        logger = self.logger
        if logger: logger.processStart()
        
        for m in self.managers:
            m.process()
        
        if logger: logger.processEnd()
    
    def render(self):
        logger = self.logger
        if logger: logger.renderStart()
        
        for r in self.renderers:
            r.prepare()
        
        Renderer.begin(self)
        for m in self.managers:
            m.render()
        Renderer.end(self)
        
        for m in self.managers:
            m.renderExtra()
        
        for r in self.renderers:
            r.cleanup()
        
        if logger: logger.renderEnd()
    
    def addRenderer(self, renderer):
        self.renderers.append(renderer)
    
    def clean(self):
        self.meshes = None
        self.managers = None
        self.renderers = None
    
    def has(self, key):
        has = self._keys.get(key)
        if has is None:
            if key == "mode3d":
                has = True
            elif key == "mode3dRealistic":
                has = os.path.isdir(os.path.join(self.basePath, "realistic"))
            elif key == "overlay":
                has = os.path.isdir(os.path.join(self.basePath, "overlay"))
            elif key == "geojson":
                has = os.path.isdir(os.path.join(self.basePath, "geojson"))
            self._keys[key] = has
            
        return has
    
    def download(self, url, filepath, data=None):
        print("Downloading the file from %s..." % url)
        if data:
            data = data.encode('ascii')
        request.urlretrieve(url, filepath, None, data)
        print("Saving the file to %s..." % filepath)
    
    def downloadOsmWays(self, ways, filepath):
        self.download(
            self.osmServer + self.osmUrlPath2,
            filepath,
            self.overpassWays % ");way(".join(ways)
        )
    
    def loadMissingWays(self, osm):
        filepath = self.osmFileExtraName % self.osmFilepath[:-4]
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
    
    def getLayer(self, layerId):
        layerIndex = self.layerIndices.get(layerId)
        return None if layerIndex is None else self.layers[layerIndex] 
    
    def createLayer(self, layerId, layerConstructor, **kwargs):
        layer = layerConstructor(layerId, self)
        for k in kwargs:
            setattr(layer, k, kwargs[k])
        self.layerIndices[layerId] = len(self.layers)
        self.layers.append(layer)
        return layer

    def getMissingHgtFiles(self):
        """
        Returns the list of missing .hgt files
        """
        latIntervals = self.latIntervals
        lonIntervals = self.lonIntervals
        missingFiles = []
        for latInterval in latIntervals:
            # latitude of the lower-left corner of the .hgt tile
            _lat = math.floor(latInterval[0])
            for lonInterval in lonIntervals:
                # longitude of the lower-left corner of the .hgt tile
                _lon = math.floor(lonInterval[0])
                hgtFileName = os.path.join(self.terrainDir, Terrain.getHgtFileName(_lat, _lon))
                # check if the .hgt file exists
                if not os.path.isfile(hgtFileName):
                    missingFiles.append(hgtFileName)
        return missingFiles
    
    def importTerrain(self, context):
        verts = []
        indices = []
        
        try:
            heightOffset = Terrain(context).terrain["height_offset"]
        except Exception:
            heightOffset = None
        
        minHeight = self.buildTerrain(verts, indices, heightOffset)
        
        # apply the offset along z-axis
        for v in verts:
            v[2] -= minHeight
        
        # create a mesh object in Blender
        mesh = bpy.data.meshes.new("Terrain")
        mesh.from_pydata(verts, [], indices)
        mesh.update()
        obj = bpy.data.objects.new("Terrain", mesh)
        obj["height_offset"] = minHeight
        if _isBlender280:
            context.scene.collection.objects.link(obj)
        else:
            context.scene.objects.link(obj)
        context.scene.blender_osm.terrainObject = obj.name
        # force smooth shading
        makeActive(obj, context)
        bpy.ops.object.shade_smooth()

    def buildTerrain(self, verts, indices, heightOffset):
        """
        The method fills verts and indices lists with values
        verts is a list of vertices
        indices is a list of tuples; each tuple is composed of 3 indices of verts that define a triangle
        
        Returns the minimal height
        """
        latIntervals = self.latIntervals
        lonIntervals = self.lonIntervals
        size = self.terrainSize
        makeQuads = (self.terrainPrimitiveType == "quad")
        
        # Number of vertex reduction
        # The reduction algorithm uses a reduction ratio which is a divider of terrainSize ie of 1200 and 3600
        decimate  = int(self.terrainReductionRatio)
        hard_size = size   # size of the file records
        size //= decimate  # size used in the loop computations
        
        minHeight = 32767
        
        vertsCounter = 0
        
        # we have an extra row for the first latitude interval
        firstLatInterval = 1
        
        # initialize the array of vertCounter values
        lonIntervalVertsCounterValues = []
        for lonInterval in lonIntervals:
            lonIntervalVertsCounterValues.append(None)
        
        for latInterval in latIntervals:
            # latitude of the lower-left corner of the .hgt tile
            _lat = math.floor(latInterval[0])
            # vertical indices that limit the active .hgt tile area
            y1 = math.floor( size * (latInterval[0] - _lat) )
            y2 = math.ceil( size * (latInterval[1] - _lat) ) + firstLatInterval - 1
            
            # we have an extra column for the first longitude interval
            firstLonInterval = 1
            
            for lonIntervalIndex,lonInterval in enumerate(lonIntervals):
                # longitude of the lower-left corner of the .hgt tile
                _lon = math.floor(lonInterval[0])
                # horizontal indices that limit the active .hgt tile area
                x1 = math.floor( size * (lonInterval[0] - _lon) ) + 1 - firstLonInterval 
                x2 = math.ceil( size * (lonInterval[1] - _lon) )
                xSize = x2-x1
                
                filepath = os.path.join(self.terrainDir, Terrain.getHgtFileName(_lat, _lon))
                
                with gzip.open(filepath, "rb") as f:
                    for y in range(y2, y1-1, -1):
                        # set the file object position at y, x1
                        # f.seek( 2*((size-y)*(size+1) + x1) )
                        # Vertex reduction: use hard_size and decimate divider
                        f.seek( 2*((hard_size-y*decimate)*(hard_size+1) + x1*decimate) )
                        
                        for x in range(x1, x2+1):
                            lat = _lat + y/size
                            lon = _lon + x/size
                            xy = self.projection.fromGeographic(lat, lon)
                            # read two bytes and convert them
                            buf = f.read(2)
                            # Vertex reduction : read more bytes for next loop
                            if decimate > 1:
                                _nothing = f.read(2*decimate-2)
                            # ">h" is a signed two byte integer
                            z = struct.unpack('>h', buf)[0]
                            if z==self.voidValue:
                                z = self.voidSubstitution
                            if heightOffset is None and z<minHeight:
                                minHeight = z
                            # add a new vertex to the verts array
                            verts.append([xy[0], xy[1], z])
                            if not firstLatInterval and y==y2:
                                topNeighborIndex = lonIntervalVertsCounterValues[lonIntervalIndex] + x - x1
                                if x!=x1:
                                    if makeQuads:
                                        indices.append((vertsCounter, topNeighborIndex, topNeighborIndex-1, vertsCounter-1))
                                    else: # self.primitiveType == "triangle"
                                        indices.append((vertsCounter-1, topNeighborIndex, topNeighborIndex-1))
                                        indices.append((vertsCounter, topNeighborIndex, vertsCounter-1))
                                elif not firstLonInterval:
                                    leftNeighborIndex = prevLonIntervalVertsCounter - (y2-y1)*(prevXsize+1)
                                    leftTopNeighborIndex = topNeighborIndex-prevYsize*(x2-x1+1)-1
                                    if makeQuads:
                                        indices.append((vertsCounter, topNeighborIndex, leftTopNeighborIndex, leftNeighborIndex))
                                    else: # self.primitiveType == "triangle"
                                        indices.append((leftNeighborIndex, topNeighborIndex, leftTopNeighborIndex))
                                        indices.append((vertsCounter, topNeighborIndex, leftNeighborIndex))
                            elif not firstLonInterval and x==x1:
                                if y!=y2:
                                    leftNeighborIndex = prevLonIntervalVertsCounter - (y-y1)*(prevXsize+1)
                                    topNeighborIndex = vertsCounter-xSize-1
                                    leftTopNeighborIndex = leftNeighborIndex-prevXsize-1
                                    if makeQuads:
                                        indices.append((vertsCounter, topNeighborIndex, leftTopNeighborIndex, leftNeighborIndex))
                                    else: # self.primitiveType == "triangle"
                                        indices.append((leftNeighborIndex, topNeighborIndex, leftTopNeighborIndex))
                                        indices.append((vertsCounter, topNeighborIndex, leftNeighborIndex))
                            elif x>x1 and y<y2:
                                topNeighborIndex = vertsCounter-xSize-1
                                leftTopNeighborIndex = vertsCounter-xSize-2
                                if makeQuads:
                                    indices.append((vertsCounter, topNeighborIndex, leftTopNeighborIndex, vertsCounter-1))
                                else: # self.primitiveType == "triangle"
                                    indices.append((vertsCounter-1, topNeighborIndex, leftTopNeighborIndex))
                                    indices.append((vertsCounter, topNeighborIndex, vertsCounter-1))
                            vertsCounter += 1
                
                if firstLonInterval:
                    # we don't have an extra column anymore
                    firstLonInterval = 0
                # remembering vertsCounter value
                prevLonIntervalVertsCounter = vertsCounter - 1
                lonIntervalVertsCounterValues[lonIntervalIndex] = prevLonIntervalVertsCounter - xSize
                # remembering xSize
                prevXsize = xSize
            if firstLatInterval:
                firstLatInterval = 0
            # remembering ySize
            prevYsize = y2-y
        
        return minHeight if heightOffset is None else heightOffset
    
    def print(self, value):
        self.stateMessage = value
        if self.area:
            self.area.tag_redraw()
    
    def getExtentFromObject(self, obj, context):
        """
        Returns a tuple minLon, minLat, maxLon, maxLat
        """
        # transform <obj.bound_box> to the world system of coordinates
        bound_box = tuple(obj.matrix_world @ Vector(v) for v in obj.bound_box)\
            if _isBlender280 else\
            tuple(obj.matrix_world*Vector(v) for v in obj.bound_box)
        bbox = []
        for i in (0,1):
            for f in (min, max):
                bbox.append(
                    f( bound_box, key=lambda v: v[i] )[i]
                )
        bbox = (
            self.projection.toGeographic(bbox[0], bbox[2]),
            self.projection.toGeographic(bbox[1], bbox[3])
        )
        # minLon, minLat, maxLon, maxLat
        return bbox[0][1], bbox[0][0], bbox[1][1], bbox[1][0]
    
    def loadExtensions(self, context=bpy.context):
        """
        Currently not used. Might be revisited later when a true extension appears
        """
        numExtensions = 0
        import sys
        # check if <bpyproj> is activated and is available in sys.modules
        self.bpyproj = "bpyproj" in (context.preferences.addons if _isBlender280 else context.user_preferences.addons) and sys.modules.get("bpyproj")
        if self.bpyproj:
            numExtensions += 1
        return numExtensions
    
    def setProjection(self, lat, lon):
        import sys
        
        projection = None
        # check if <bpyproj> is activated and is available in sys.modules
        bpyproj = "bpyproj" in (bpy.context.preferences.addons if _isBlender280 else bpy.context.user_preferences.addons) and sys.modules.get("bpyproj")
        if bpyproj:
            projection = bpyproj.getProjection(lat, lon)
        if not projection:
            from util.transverse_mercator import TransverseMercator
            # fall back to the Transverse Mercator
            projection = TransverseMercator(lat=lat, lon=lon)
        self.projection = projection


app = App()