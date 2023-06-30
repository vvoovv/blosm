from operator import itemgetter
from math import radians, cos, sin, sqrt
from urllib import request
from urllib.parse import urlparse
from os.path import splitext, basename, join as joinStrings, exists as pathExists
import json, ssl
from mathutils import Vector


_geometricErrorRanges = {
    'lod1': (60., 200.),
    'lod2': (30., 60.),
    'lod3': (15., 30),
    'lod4': (7., 15.),
    'lod5': (3., 7),
    'lod6': (0., 3)
}

# heights above the sea level
_minHeight = 0.
_maxHeight = 9000.


_radii = _wgs84Radii = (
    6378137.0,
    6378137.0,
    6356752.3142451793
)
_radiiSquared = _wgs84RadiiSquared = (
    _radii[0] * _radii[0],
    _radii[1] * _radii[1],
    _radii[2] * _radii[2]
)


class BaseManager:
    
    def __init__(self, rootUri, renderer):
        self.rootUri = rootUri
        rootUriComponents = urlparse(rootUri)
        self.uriServer = rootUriComponents.scheme + "://" + rootUriComponents.netloc
        
        self.renderer = renderer
        # <self.constantUriQuery> is defined by the application (e.g. an API key), it is
        # always appended to all URIs
        self.constantUriQuery = ''
        # <self.uriQuery> is returned by a 3D Tiles server (e.g. a session ID), it is appended
        # to all subsequent URI requests
        self.uriQuery = ''
        
        self.tilesDir = ''
        
        self.cacheJsonFiles = True
        self.cache3dFiles = False
    
    def setGeometricErrorRange(self, rangeId):
        self.geometricErrorMin, self.geometricErrorMax = _geometricErrorRanges[rangeId]
    
    def render(self, minLon, minLat, maxLon, maxLat):
        self.errors = []
        self.areaBbox = BaseManager.getAreaBbox(minLon, minLat, maxLon, maxLat)
        
        self.renderer.prepare(self)
        
        try:
            tileset = self.getJsonFile(self.rootUri, None, False)
            self.renderTileset(tileset)
        except Exception as e:
            # return only a critical error
            return ("Unable to process the root URI of the 3D Tiles: %s" % str(e),)
        
        numRenderedTiles = self.renderer.finalize(self)
        
        # return the number of rendered tiles and uncritical errors
        return numRenderedTiles, self.errors
    
    def renderTileset(self, tileset):
        tileset = tileset["root"]
        
        bbox = BaseManager.getTileBbox( tileset["boundingVolume"]["box"] )
                
        if BaseManager.bboxesOverlap(self.areaBbox, bbox):
            self.renderChildren(tileset["children"])
            
    def renderChildren(self, children):
        # render children
        for tile in children:
            tileBbox = BaseManager.getTileBbox( tile["boundingVolume"]["box"] )
            if BaseManager.bboxesOverlap(self.areaBbox, tileBbox):
                geometricError = tile["geometricError"]
                if self.geometricErrorMin < geometricError <= self.geometricErrorMax:
                    if "content" in tile:
                        self.renderTile(tile, jsonOnly=False)
                    else:
                        _children = tile.get("children")
                        if _children:
                            self.renderChildren(_children)
                else:
                    _children = tile.get("children")
                    if _children:
                        self.renderChildren(_children)
                    elif "content" in tile:
                        self.renderTile(tile, jsonOnly=True)
        
    def renderTile(self, tile, jsonOnly):
        uriComponents = urlparse(tile["content"]["uri"])
        
        contentExtension = splitext(uriComponents.path)[1].lower()
        
        if jsonOnly and contentExtension != ".json":
            return
        
        if uriComponents.query:
            self.uriQuery = uriComponents.query
        
        uri = self.getUri(uriComponents)
        
        try:
            if contentExtension == ".json":
                tileset = self.getJsonFile(uri, uriComponents.path, self.cacheJsonFiles)
                self.renderTileset(tileset)
            elif contentExtension == ".glb":
                self.renderer.renderGlb(self, uri, uriComponents.path, self.cache3dFiles)
        except Exception as e:
            self.processError(e, uri)
    
    def getUri(self, uriComponents):
        uri = (self.uriServer if self.uriServer else uriComponents.scheme + "://" + uriComponents.netloc) +\
            uriComponents.path
        if self.uriQuery and self.constantUriQuery:
            uri += '?' + self.uriQuery + '&' + self.constantUriQuery
        elif self.constantUriQuery:
            uri += '?' + self.constantUriQuery
        elif self.uriQuery:
            uri += '?' + self.uriQuery
        
        return uri                 
        
    @staticmethod
    def getAreaBbox(minLon, minLat, maxLon, maxLat):
        bbox2d = (
            (minLat, minLon), (minLat, maxLon), (maxLat, maxLon), (maxLat, minLon)
        )
        # The bounding box with the axes that are not parallel to the axes of
        # the system of reference
        return BaseManager.getParallelAxesBbox([
            BaseManager.fromGeographic(lat,lon,height)\
                for height in (_minHeight, _maxHeight) for lat, lon in bbox2d
        ])
    
    @staticmethod
    def getParallelAxesBbox(bbox):
        # get the bounding box with axes parallel to the axes of the system of reference
        return\
            (
                min(bbox, key=itemgetter(0))[0],
                min(bbox, key=itemgetter(1))[1],
                min(bbox, key=itemgetter(2))[2]
            ),\
            (
                max(bbox, key=itemgetter(0))[0],
                max(bbox, key=itemgetter(1))[1],
                max(bbox, key=itemgetter(2))[2]
            )
    
    def getJsonFile(self, uri, path, cacheContent):
        if cacheContent:
            filePath = joinStrings(self.tilesDir, basename(path))
            if pathExists(filePath):
                with open(filePath, 'r') as jsonFile:
                    result = json.load(jsonFile)
            else:
                result = self.download(uri)
                with open(filePath, 'wb') as jsonFile:
                    jsonFile.write(result)
                result = json.loads(result)
        else:
            result = self.download(uri)
            result = json.loads(result)
        
        return result
    
    def download(self, url):
        req = request.Request(
            url,
            data = None,
            headers = {
                "User-Agent": "Blosm"
            }
        )
        # a hack to avoid CERTIFICATE_VERIFY_FAILED error
        ctx = ssl._create_unverified_context()
        response = request.urlopen(req, context=ctx).read()
        return response
    
    @staticmethod
    def fromGeographic(lat, lon, height):
        lat = radians(lat)
        lon = radians(lon)
        
        cosLatitude = cos(lat)
        
        scratchN = Vector((
            cosLatitude * cos(lon),
            cosLatitude * sin(lon),
            sin(lat)
        ))
        scratchN.normalize()
        
        scratchK = Vector((
            _radiiSquared[0]*scratchN[0],
            _radiiSquared[1]*scratchN[1],
            _radiiSquared[2]*scratchN[2]
        ))
        
        gamma = sqrt( scratchN.dot(scratchK) )
        scratchK /= gamma
        scratchN *= height
        
        return scratchK + scratchN
    
    @staticmethod
    def getTileBbox(bbox):
        bboxCenter = Vector((bbox[0], bbox[1], bbox[2]))
        # direction and half-length for the local X-axis of <bbox>
        bboxX = Vector((bbox[3], bbox[4], bbox[5]))
        # direction and half-length for the local Y-axis of <bbox>
        bboxY = Vector((bbox[6], bbox[7], bbox[8]))
        # direction and half-length for the local Z-axis of <bbox>
        bboxZ = Vector((bbox[9], bbox[10], bbox[11]))
        
        # get the bounding box with axes parallel to the axes of the system of reference
        return BaseManager.getParallelAxesBbox([
            bboxCenter - bboxX - bboxY - bboxZ,
            bboxCenter + bboxX - bboxY - bboxZ,
            bboxCenter + bboxX + bboxY - bboxZ,
            bboxCenter - bboxX + bboxY - bboxZ,
            bboxCenter - bboxX - bboxY + bboxZ,
            bboxCenter + bboxX - bboxY + bboxZ,
            bboxCenter + bboxX + bboxY + bboxZ,
            bboxCenter - bboxX + bboxY + bboxZ
        ])
    
    @staticmethod
    def bboxesOverlap(bbox1, bbox2):
        return \
            bbox1[1][0] >= bbox2[0][0] and bbox2[1][0] >= bbox1[0][0]\
            and\
            bbox1[1][1] >= bbox2[0][1] and bbox2[1][1] >= bbox1[0][1]\
            and\
            bbox1[1][2] >= bbox2[0][2] and bbox2[1][2] >= bbox1[0][2]
    
    def processError(self, e, uri):
        self.errors.append(
            "There was an error when processing the URI %s: %s" % (uri, str(e))
        )