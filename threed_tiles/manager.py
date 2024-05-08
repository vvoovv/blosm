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
        
        # A rotation matrix the rotates the unit vectors along the axes of the global system of references to <eX>, <eY>, <eZ>
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
        
        self.areaCenter = 0.5 * (p0 + self.areaVerts[6])
        self.areaRadius = 0.5 * (self.areaVerts[6] - p0).length
        self.areaRadiusSquared = self.areaRadius * self.areaRadius
        self.areaRadiusDoubled = 2 * self.areaRadius
    
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
        
        # The coordinates of the current bounding box are transformed to the system of reference where
        # (1) the edges of the bounding box of the area of interest are parallel to the axes of the system of reference and
        # (2) the origin of the bounding box of the area of interest is located at the origin of that system of reference 
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
        
        bboxRadius = (bboxCenter - bboxOrigin).length
        
        # (1)
        # A quick check if the spheres with the centers at <self.areaCenter> and <bboxCenter> and radii <self.areaRadius> and <bboxRadius>
        # respectively do not intersect. That means the bounding box of the area of interest and
        # the current bounding box do not intersect either.
        # The condition for that is the following: the distance between <self.areaCenter> and <bboxCenter> is larger than the sum
        # of the radii <self.areaRadius> and <bboxRadius>.
        if (self.areaCenter - bboxCenter).length_squared > self.areaRadiusSquared + self.areaRadiusDoubled * bboxRadius + bboxRadius*bboxRadius:
            return False
        
        # (2)
        # Perform a thorough interesection test (read the description of the method <BaseManager.bboxIntersectsBbox> to understand
        # what exactly is tested)
        if BaseManager.bboxIntersectsBbox(bboxVerts, self.areaSizeX, self.areaSizeY, self.areaSizeZ):
            return True

        bboxSizeX = bboxX.length
        bboxSizeY = bboxY.length
        bboxSizeZ = bboxZ.length
        
        # Unit vectors along the sides of the current bounding box
        eX = bboxX/bboxSizeX
        eY = bboxY/bboxSizeY
        eZ = bboxZ/bboxSizeZ
        
        bboxSizeX *= 2.
        bboxSizeY *= 2.
        bboxSizeZ *= 2.
        
        # A rotation matrix the rotates the unit vectors along the axes of the global system of references to <eX>, <eY>, <eZ>
        bboxRotationMatrix = Matrix((
            (eX[0], eX[1], eX[2]),
            (eY[0], eY[1], eY[2]),
            (eZ[0], eZ[1], eZ[2])
        ))
        
        # The coordinates of the bounding box of the area of interest are transformed to the system of reference where
        # (1) the edges of the curent bounding box are parallel to the axes of the system of reference and
        # (2) the origin of the current bounding box is located at the origin of that system of reference 
        areaVerts = [bboxRotationMatrix @ (areaVert - bboxOrigin) for areaVert in self.areaVerts]
        
        # (3)
        # Perform a thorough interesection test (read the description of the method <BaseManager.bboxIntersectsBbox> to understand
        # what exactly is tested)
        if BaseManager.bboxIntersectsBbox(areaVerts, bboxSizeX, bboxSizeY, bboxSizeZ):
            return True
        
        return False
    
    @staticmethod
    def bboxIntersectsBbox(verts, bboxSizeX, bboxSizeY, bboxSizeZ):
        """
        Checks the partial conditions if the bounding box defined by its <verts> intersect the bounding box
        with the dimensions <bboxSizeX>, <bboxSizeY>, <bboxSizeZ> and the edges parallel to the axes of the system of reference
        and the origin located at the origin of the system of reference.
        """
        
        # (1)
        # Check if at least one vertex of <verts> is located inside the bounding box
        # with the dimensions <bboxSizeX>, <bboxSizeY>, <bboxSizeZ> and the edges parallel to the axes of the system of reference
        # and he origin located at the origin of the system of reference.
        for vert in verts:
            if 0. < vert[0] < bboxSizeX and\
                0. < vert[1] < bboxSizeY and\
                0. < vert[2] < bboxSizeZ:
                    return True
        
        # Create edges out of <verts> as a pair of vertices the form the related edge
        edges = (
            (verts[0], verts[1]), (verts[1], verts[2]), (verts[2], verts[3]), (verts[3], verts[0]),
            (verts[4], verts[5]), (verts[5], verts[6]), (verts[6], verts[7]), (verts[7], verts[4]),
            (verts[0], verts[4]), (verts[1], verts[5]), (verts[2], verts[6]), (verts[3], verts[7])
        )
        
        # (2)
        # Check if at least one edge intersects the bounding box with the dimensions <bboxSizeX>, <bboxSizeY>, <bboxSizeZ> and
        # the edges pareallel to the axes of the system of reference and the origin located at the origin of the system of reference.
        
        for edge in edges:
            v1, v2 = edge
            
            if v1[0] != v2[0]:
                # check if <edge> intersects the face of the bounding box at <x == 0>
                factor = v1[0]/(v1[0] - v2[0])
                if 0. < factor < 1. and 0. < v1[1] + factor * (v2[1] - v1[1]) < bboxSizeY and 0. < v1[2] + factor * (v2[2] - v1[2]) < bboxSizeZ:
                    return True
                # check if <edge> intersects the face of the bounding box at <x == bboxSizeX>
                factor = (v1[0] - bboxSizeX)/(v1[0] - v2[0])
                if 0. < factor < 1. and 0. < v1[1] + factor * (v2[1] - v1[1]) < bboxSizeY and 0. < v1[2] + factor * (v2[2] - v1[2]) < bboxSizeZ:
                    return True
            
            if v1[1] != v2[1]:
                # check if <edge> intersects the face of the bounding box at <y == 0>
                factor = v1[1]/(v1[1] - v2[1])
                if 0. < factor < 1. and 0. < v1[0] + factor * (v2[0] - v1[0]) < bboxSizeX and 0. < v1[2] + factor * (v2[2] - v1[2]) < bboxSizeZ:
                    return True
                # check if <edge> intersects the face of the bounding box at <y == bboxSizeY>
                factor = (v1[1] - bboxSizeY)/(v1[1] - v2[1])
                if 0. < factor < 1. and 0. < v1[0] + factor * (v2[0] - v1[0]) < bboxSizeX and 0. < v1[2] + factor * (v2[2] - v1[2]) < bboxSizeZ:
                    return True
    
            if v1[2] != v2[2]:
                # check if <edge> intersects the face of the bounding box at <z == 0>
                factor = v1[2]/(v1[2] - v2[2])
                if 0. < factor < 1. and 0. < v1[0] + factor * (v2[0] - v1[0]) < bboxSizeX and 0. < v1[1] + factor * (v2[1] - v1[1]) < bboxSizeY:
                    return True
                # check if <edge> intersects the face of the bounding box at <z == bboxSizeZ>
                factor = (v1[2] - bboxSizeZ)/(v1[2] - v2[2])
                if 0. < factor < 1. and 0. < v1[0] + factor * (v2[0] - v1[0]) < bboxSizeX and 0. < v1[1] + factor * (v2[1] - v1[1]) < bboxSizeY:
                    return True
        
        return False
    
    def processError(self, e, uri):
        self.errors.append(
            "There was an error when processing the URI %s: %s" % (uri, str(e))
        )