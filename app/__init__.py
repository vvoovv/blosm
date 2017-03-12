"""
This file is part of blender-osm (OpenStreetMap importer for Blender).
Copyright (C) 2014-2017 Vladimir Elistratov
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

import bpy
import os, json, webbrowser, base64, math, gzip, struct
from urllib import request

import defs
from .layer import Layer
from renderer import Renderer
from terrain import Terrain
from util.polygon import Polygon


class App:
    
    layerIds = ["buildings", "highways", "railways", "water", "forests", "vegetation"]
    
    osmUrl = "http://overpass-api.de/api/map?bbox=%s,%s,%s,%s"
    
    osmUrl2 = "http://overpass-api.de/api/interpreter"
    
    terrainUrl = "https://s3.amazonaws.com/elevation-tiles-prod/skadi/%s/%s"
    
    osmDir = "osm"
    
    terrainSubDir = "terrain"
    
    osmFileName = "map%s.osm"
    
    osmFileExtraName = "%s_extra.osm"
    
    # request to the Overpass server to get both ways and their nodes for the given way ids
    overpassWays = "((way(%s););node(w););out;"
    
    layerOffsets = {
        "buildings": 0.2,
        "water": 0.2,
        "forests": 0.1,
        "vegetation": 0.,
        "highways": 0.2,
        "railways": 0.2
    }
    
    # diffuse colors for some layers
    colors = {
        "buildings": (0.309, 0.013, 0.012),
        "water": (0.009, 0.002, 0.8),
        "forests": (0.02, 0.208, 0.007),
        "vegetation": (0.007, 0.558, 0.005),
        "highways": (0.1, 0.1, 0.1),
        "railways": (0.2, 0.2, 0.2)
    }
    
    # Default value for <offset> parameter for the SHRINKWRAP modifier;
    # it's used to project flat meshes onto a terrain
    swOffset = 0.05
    
    voidValue = -32768
    voidSubstitution = 0
    
    def __init__(self):
        self.version = None
        self.load()
    
    def initOsm(self, op, context, basePath, addonName):
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
        
        self.setAttributes(context)
        addon = context.scene.blender_osm
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
                self.osmUrl % (app.minLon, app.minLat, app.maxLon, app.maxLat),
                osmFilepath
            )
        else:
            self.osmFilepath = os.path.realpath(bpy.path.abspath(self.osmFilepath))
        
        if self.loadMissingMembers:
            self.incompleteRelations = []
            self.missingWays = set()
        
        if not app.has(defs.Keys.mode3d):
            self.mode = '2D'
        
        # check if have a terrain Blender object set
        terrain = Terrain(context)
        self.terrain = terrain if terrain.terrain else None
        if self.terrain:
            terrain.init()
        
        # manager (derived from manager.Manager) performing some processing
        self.managers = []
        
        self.prepareLayers()
        if self.terrain and self.singleObject and not self.layered:
            print("Imported OpenStreetMap objects will be arranged into layers")
            self.layered = True
        
        # tangent to check if an angle of the polygon is straight
        Polygon.straightAngleTan = math.tan(math.radians( abs(180.-self.straightAngleThreshold) ))
    
    def initTerrain(self, op, context, basePath, addonName):
        self.setDataDir(context, basePath, addonName)
        # create a sub-directory under <self.dataDir> for OSM files
        terrainDir = os.path.join(self.dataDir, self.terrainSubDir)
        self.terrainDir = terrainDir
        if not os.path.exists(terrainDir):
            os.makedirs(terrainDir)
        
        self.setAttributes(context)
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
    
    def setAttributes(self, context):
        """
        Copies properties from <context.scene.blender_osm>
        """
        addon = context.scene.blender_osm
        for p in dir(addon):
            if not (p.startswith("__") or p in ("bl_rna", "rna_type")):
                setattr(self, p, getattr(addon, p))
    
    def setDataDir(self, context, basePath, addonName):
        """
        Sets <self.dataDir>, i.e. path to data
        """
        prefs = context.user_preferences.addons
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
    
    def prepareLayers(self):
        layerIndices = {}
        layers = []
        i = 0
        for layerId in self.layerIds:
            if getattr(self, layerId):
                layerIndices[layerId] = i
                layers.append(Layer(layerId, self))
                i += 1
        self.layerIndices = layerIndices
        self.layers = layers
    
    def process(self):
        logger = self.logger
        if logger: logger.processStart()
        
        for m in self.managers:
            m.process()
        
        if logger: logger.processEnd()
    
    def render(self):
        logger = self.logger
        if logger: logger.renderStart()
        
        Renderer.begin(self)
        for m in self.managers:
            m.render()
        Renderer.end(self)
        
        if logger: logger.renderEnd()
    
    def clean(self):
        self.meshes = None
        self.managers = None
    
    def has(self, key):
        return self.license and (self.all or key in self.keys)
    
    def load(self):
        # this directory
        directory = os.path.dirname(os.path.realpath(__file__))
        # app/..
        directory = os.path.realpath( os.path.join(directory, os.pardir) )
        path = os.path.join(directory, defs.App.file)
        self.license = os.path.isfile(path)
        if not self.license:
            return
        
        with open(path, "r", encoding="ascii") as data:
            data = json.loads( base64.b64decode( bytes.fromhex(data.read()) ).decode('ascii') )
        
        self.all = data.get("all", False)
        self.keys = set(data.get("keys", ()))
    
    def show(self):
        bpy.ops.prk.check_version_osm('INVOKE_DEFAULT')
    
    def download(self, url, filepath, data=None):
        print("Downloading the file from %s..." % url)
        if data:
            data = data.encode('ascii')
        request.urlretrieve(url, filepath, None, data)
        print("Saving the file to %s..." % filepath)
    
    def downloadOsmWays(self, ways, filepath):
        self.download(
            self.osmUrl2,
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
        for relation, _id, members, tags, ci in self.incompleteRelations:
            # below there is the same code for a relation as in osm.parse(..)
            relation.process(members, tags, osm)
            if relation.valid:
                skip = osm.processCondition(ci, relation, _id, osm.parseRelation)
                if not _id in osm.relations and not skip:
                    osm.relations[_id] = relation
        # cleanup
        self.incompleteRelations = None
        self.missingWays = None
    
    def getLayer(self, layerId):
        return self.layers[ self.layerIndices.get(layerId) ]

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
        context.scene.objects.link(obj)
        context.scene.blender_osm.terrainObject = obj.name
        # force smooth shading
        obj.select = True
        context.scene.objects.active = obj
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
                        f.seek( 2*((size-y)*(size+1) + x1) )
                        for x in range(x1, x2+1):
                            lat = _lat + y/size
                            lon = _lon + x/size
                            xy = self.projection.fromGeographic(lat, lon)
                            # read two bytes and convert them
                            buf = f.read(2)
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


class OperatorPopup(bpy.types.Operator):
    bl_idname = "prk.check_version_osm"
    bl_label = ""
    bl_description = defs.App.description
    bl_options = {'INTERNAL'}
    
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)
    
    def execute(self, context):
        webbrowser.open_new_tab(defs.App.url)
        return {'FINISHED'}
    
    def cancel(self, context):
        webbrowser.open_new_tab(defs.App.url)
    
    def draw(self, context):
        layout = self.layout
        
        iconPlaced = False
        for label in defs.App.popupStrings:
            if iconPlaced:
                self.label(label)
            else:
                self.label(label, icon='INFO')
                iconPlaced = True
        
        layout.separator()
        layout.separator()
        
        self.label("Click to buy")
    
    def label(self, text, **kwargs):
        row = self.layout.row()
        row.alignment = "CENTER"
        row.label(text, **kwargs)


app = App()


def register():
    bpy.utils.register_module(__name__)

def unregister():
    bpy.utils.unregister_module(__name__)