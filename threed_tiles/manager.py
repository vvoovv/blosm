from math import radians, cos, sin, sqrt
from urllib import request
from urllib.parse import urlparse
from os.path import splitext, basename, join as joinStrings, exists as pathExists
import json, ssl
from mathutils import Vector, Matrix


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
        self.initAreaData(minLon, minLat, maxLon, maxLat)
        
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
                
        if self.areaOverlapsWith( tileset["boundingVolume"]["box"] ):
            self.renderChildren(tileset["children"])
            
    def renderChildren(self, children):
        # render children
        for tile in children:
            if self.areaOverlapsWith( tile["boundingVolume"]["box"] ):
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
    
    def initAreaData(self, minLon, minLat, maxLon, maxLat):
        p0 = self.areaOrigin = BaseManager.fromGeographic(minLat, minLon, _minHeight)
        pZ = BaseManager.fromGeographic(minLat, minLon, _maxHeight)
        pX = BaseManager.fromGeographic(minLat, maxLon, _minHeight)
        pY = BaseManager.fromGeographic(maxLat, minLon, _minHeight)
        
        # unit vectors
        dX = pX - p0
        self.areaSizeX = dX.length
        eX = dX.normalized()
        
        dZ = pZ - p0
        self.areaSizeZ = dZ.length
        eZ = dZ.normalized()
        
        eY = eZ.cross(eX)
        self.areaSizeY = (pY - p0).length
        dY = self.areaSizeY * eY
        
        # rotation matrix
        self.areaRotationMatrix = Matrix((
            (eX[0], eX[1], eX[2]),
            (eY[0], eY[1], eY[2]),
            (eZ[0], eZ[1], eZ[2])
        ))
        
        self.areaVerts = (
            p0,
            p0 + dX,
            p0 + dX + dY,
            p0 + dY,
            pZ,
            pZ + dX,
            pZ + dX + dY,
            pZ + dY
        )
        
        #self.debugInit()
    
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
    
    def areaOverlapsWith(self, bbox):
        bboxCenter = Vector((bbox[0], bbox[1], bbox[2]))
        # direction and half-length for the local X-axis of <bbox>
        bboxX = Vector((bbox[3], bbox[4], bbox[5]))
        # direction and half-length for the local Y-axis of <bbox>
        bboxY = Vector((bbox[6], bbox[7], bbox[8]))
        # direction and half-length for the local Z-axis of <bbox>
        bboxZ = Vector((bbox[9], bbox[10], bbox[11]))
        
        bboxOrigin = bboxCenter - bboxX - bboxY - bboxZ
        
        bboxVerts = (
            self.areaRotationMatrix @ ( bboxOrigin - self.areaOrigin ),
            self.areaRotationMatrix @ ( bboxCenter + bboxX - bboxY - bboxZ - self.areaOrigin ),
            self.areaRotationMatrix @ ( bboxCenter + bboxX + bboxY - bboxZ - self.areaOrigin ),
            self.areaRotationMatrix @ ( bboxCenter - bboxX + bboxY - bboxZ - self.areaOrigin ),
            self.areaRotationMatrix @ ( bboxCenter - bboxX - bboxY + bboxZ - self.areaOrigin ),
            self.areaRotationMatrix @ ( bboxCenter + bboxX - bboxY + bboxZ - self.areaOrigin ),
            self.areaRotationMatrix @ ( bboxCenter + bboxX + bboxY + bboxZ - self.areaOrigin ),
            self.areaRotationMatrix @ ( bboxCenter - bboxX + bboxY + bboxZ - self.areaOrigin )
        )
        
        #self.debugBboxes(bboxVerts)
        
        if BaseManager.edgesIntersectsBbox(bboxVerts, self.areaSizeX, self.areaSizeY, self.areaSizeZ):
            return True

        bboxSizeX = bboxX.length
        bboxSizeY = bboxY.length
        bboxSizeZ = bboxZ.length
                
        eX = bboxX/bboxSizeX
        eY = bboxY/bboxSizeY
        eZ = bboxZ/bboxSizeZ
        
        bboxSizeX *= 2.
        bboxSizeY *= 2.
        bboxSizeZ *= 2.
        
        # bbox rotation matrix
        bboxRotationMatrix = Matrix((
            (eX[0], eX[1], eX[2]),
            (eY[0], eY[1], eY[2]),
            (eZ[0], eZ[1], eZ[2])
        ))
        
        areaVerts = [bboxRotationMatrix @ (areaVert - bboxOrigin) for areaVert in self.areaVerts]
        
        if BaseManager.edgesIntersectsBbox(areaVerts, bboxSizeX, bboxSizeY, bboxSizeZ):
            return True
        
        return False
    
    @staticmethod
    def edgesIntersectsBbox(verts, bboxSizeX, bboxSizeY, bboxSizeZ):
        for vert in verts:
            if 0. < vert[0] < bboxSizeX and\
                0. < vert[1] < bboxSizeY and\
                0. < vert[2] < bboxSizeZ:
                    return True
        
        edges = (
            (verts[0], verts[1]), (verts[1], verts[2]), (verts[2], verts[3]), (verts[3], verts[0]),
            (verts[4], verts[5]), (verts[5], verts[6]), (verts[6], verts[7]), (verts[7], verts[4]),
            (verts[0], verts[4]), (verts[1], verts[5]), (verts[2], verts[6]), (verts[3], verts[7])
        )
        
        for edge in edges:
            v1, v2 = edge
            
            if v1[0] != v2[0]:
                # check if <edge> intersects the face of bbox at <x == 0>
                factor = v1[0]/(v1[0] - v2[0])
                if 0. < factor < 1. and 0. < v1[1] + factor * (v2[1] - v1[1]) < bboxSizeY and 0. < v1[2] + factor * (v2[2] - v1[2]) < bboxSizeZ:
                    return True
                # check if <edge> intersects the face of bbox at <x == bboxSizeX>
                factor = (v1[0] - bboxSizeX)/(v1[0] - v2[0])
                if 0. < factor < 1. and 0. < v1[1] + factor * (v2[1] - v1[1]) < bboxSizeY and 0. < v1[2] + factor * (v2[2] - v1[2]) < bboxSizeZ:
                    return True
            
            if v1[1] != v2[1]:
                # check if <edge> intersects the face of bbox at <y == 0>
                factor = v1[1]/(v1[1] - v2[1])
                if 0. < factor < 1. and 0. < v1[0] + factor * (v2[0] - v1[0]) < bboxSizeX and 0. < v1[2] + factor * (v2[2] - v1[2]) < bboxSizeZ:
                    return True
                # check if <edge> intersects the face of bbox at <y == bboxSizeY>
                factor = (v1[1] - bboxSizeY)/(v1[1] - v2[1])
                if 0. < factor < 1. and 0. < v1[0] + factor * (v2[0] - v1[0]) < bboxSizeX and 0. < v1[2] + factor * (v2[2] - v1[2]) < bboxSizeZ:
                    return True
    
            if v1[2] != v2[2]:
                # check if <edge> intersects the face of bbox at <z == 0>
                factor = v1[2]/(v1[2] - v2[2])
                if 0. < factor < 1. and 0. < v1[0] + factor * (v2[0] - v1[0]) < bboxSizeX and 0. < v1[1] + factor * (v2[1] - v1[1]) < bboxSizeY:
                    return True
                # check if <edge> intersects the face of bbox at <z == bboxSizeZ>
                factor = (v1[2] - bboxSizeZ)/(v1[2] - v2[2])
                if 0. < factor < 1. and 0. < v1[0] + factor * (v2[0] - v1[0]) < bboxSizeX and 0. < v1[1] + factor * (v2[1] - v1[1]) < bboxSizeY:
                    return True
        
        return False
    
    def processError(self, e, uri):
        self.errors.append(
            "There was an error when processing the URI %s: %s" % (uri, str(e))
        )

"""
    def debugInit(self):
        self.debugBboxCounter = 0
        
        self.debugGenerateBbox(self.areaVerts)
        
    
    def debugBboxes(self, bboxVerts):
        self.debugBboxCounter += 1
        if self.debugBboxCounter <= 4:
            return
        
        self.debugGenerateBbox(bboxVerts)
    
    def debugGenerateBbox(self, bboxVerts):
        scale = 1000000.
        
        import bpy
        from util.blender import createMeshObject, getBmesh, setBmesh
        
        obj = createMeshObject("Bbox " +str(self.debugBboxCounter))
        bm = getBmesh(obj)
        
        verts = [ bm.verts.new(bboxVert/scale) for bboxVert in bboxVerts ]
        
        bm.faces.new((verts[3], verts[2], verts[1], verts[0]))
        bm.faces.new((verts[0], verts[1], verts[5], verts[4]))
        bm.faces.new((verts[1], verts[2], verts[6], verts[5]))
        bm.faces.new((verts[2], verts[3], verts[7], verts[6]))
        bm.faces.new((verts[3], verts[0], verts[4], verts[7]))
        bm.faces.new((verts[4], verts[5], verts[6], verts[7]))
        
        setBmesh(obj, bm)
"""