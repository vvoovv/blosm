from math import radians, cos, sin, sqrt
from urllib import request, error
from urllib.parse import urlparse
from os.path import splitext, dirname
import json, ssl, gzip
from functools import partial
from hashlib import sha256
from time import sleep
from mathutils import Vector, Matrix
from ..downloads import Request
from ..downloads.diagnostics import redact_text, redact_url
from ..downloads.transport import DownloadTransport
from ..progressive.core import Job, Download, Drain


# heights above the sea level
_minHeight = -5000.
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
    useDownloadTransport = False
    
    geometricErrors = {
        'lod1': 200.,
        'lod2': 60.,
        'lod3': 30.,
        'lod4': 15.,
        'lod5': 7.,
        'lod6': 3.
    }
    
    def __init__(self, rootUri, renderer):
        self.rootUri = rootUri
        rootUriComponents = urlparse(rootUri)
        self.uriServer = rootUriComponents.scheme + "://" + rootUriComponents.netloc
        self.baseUri = self.uriServer + dirname(rootUriComponents.path) + '/'
        
        self.renderer = renderer
        # <self.constantUriQuery> is defined by the application (e.g. an API key), it is
        # always appended to all URIs
        self.constantUriQuery = ''
        # <self.uriQuery> is returned by a 3D Tiles server (e.g. a session ID), it is appended
        # to all subsequent URI requests
        self.uriQuery = ''
        self.concurrentDownloads = False
        self.downloadWorkers = 4
        
    def setGeometricError(self, geometricError):
        self.geometricError = geometricError
    
    def render(self, minLon, minLat, maxLon, maxLat):
        steps = self.iterRender(minLon, minLat, maxLon, maxLat)
        if self.concurrentDownloads or self.useDownloadTransport:
            try:
                transport = DownloadTransport(max_requests=self.downloadWorkers if self.concurrentDownloads else 1)
            except Exception as exc:
                return ('Unable to start the download helper: %s' % redact_text(exc),)
            job = Job(steps, transport)
            try:
                while not job.done:
                    job.step()
                    if job.waiting:
                        sleep(.001)  # Only the explicitly blocking runner waits.
                return (str(job.error),) if job.error else job.result
            finally:
                if not job.done:
                    job.cancel()
        value, failure = None, None
        try:
            while True:
                uri = steps.throw(failure) if failure else steps.send(value)
                value, failure = None, None
                if uri is not None:
                    try:
                        value = self.download(uri.url)
                    except Exception as exc:
                        failure = exc
        except StopIteration as finished:
            return finished.value
        finally:
            steps.close()

    def iterRender(self, minLon, minLat, maxLon, maxLat):
        """Yield download URLs or cooperative checkpoints, accepting bytes back.

        Both blocking and progressive modes use the same traversal and errors.
        The renderer determines whether final positioning happens per tile.
        """
        self.errors = []
        self.initAreaData(minLon, minLat, maxLon, maxLat)
        
        self.renderer.prepare(self)

        try:
            critical = yield from self.iterContents()
            if self.concurrentDownloads:
                yield Drain()
        finally:
            # Drain even after a traversal error, but never yield on close().
            self.renderer.finalize(self)
        return critical or (self.renderer.num_imported_tiles, self.errors)

    def iterContents(self):
        try:
            tileset = json.loads((yield self.downloadRequest(self.rootUri)))
            yield from self.renderTileset(tileset, self.baseUri, None)
        except error.HTTPError as e:
            error_message = "Unable to process the root URI of the 3D Tiles: %s" % str(e)
            if self.uriServer == "https://tile.googleapis.com":
                try:
                    response = json.loads(e.read(65536))
                    if isinstance(response, dict) and isinstance(response.get('error'), dict):
                        message = response['error'].get('message')
                        if isinstance(message, str):
                            error_message += "\nMessage from Google Cloud:\n%s" % message
                except Exception:
                    # Error details are optional, including on the old blocking
                    # path. A broken body must not hide the original HTTP error.
                    pass
                finally:
                    e.close()
            return (self.redactError(error_message, self.rootUri),)
        except Exception as e:
            # return only a critical error
            return ("Unable to process the root URI of the 3D Tiles: %s" % self.redactError(e, self.rootUri),)
    
    def renderTileset(self, tileset, baseUri, transformMatrix):
        yield from self.renderTile(tileset["root"], baseUri, transformMatrix)
    
    def renderTile(self, element, baseUri, transformMatrix):
        """
        Renders a tile or <tileset["root"]> and its children if they overlap with the area of interest
        and if their geometric error is smaller than <self.geometricError>.
        """
        
        yield None
        transformMatrix = ( transformMatrix @ BaseManager.initMatrixFromTransform(element["transform"]) if transformMatrix else BaseManager.initMatrixFromTransform(element["transform"]) )\
            if "transform" in element else transformMatrix

        if self.areaOverlapsWith( element["boundingVolume"], transformMatrix ):
            geometricError = element["geometricError"]
            #if self.geometricErrorMin < geometricError <= self.geometricErrorMax:
            if geometricError <= self.geometricError:
                if "content" in element:
                    yield from self.renderTileContent(element, baseUri, transformMatrix, jsonOnly=False)
                else:
                    children = element.get("children")
                    if children:
                        yield from self.renderChildren(children, baseUri, transformMatrix)
            else:
                children = element.get("children")
                if children:
                    yield from self.renderChildren(children, baseUri, transformMatrix)
                elif "content" in element:
                    yield from self.renderTileContent(element, baseUri, transformMatrix, jsonOnly=True)
    
    def renderChildren(self, children, baseUri, transformMatrix):
        # render children
        for tile in children:
            yield from self.renderTile(tile, baseUri, transformMatrix)
        
    def renderTileContent(self, tile, baseUri, transformMatrix, jsonOnly):
        content = tile["content"]
        boundingVolume = content.get("boundingVolume")
        if not boundingVolume or self.areaOverlapsWith(boundingVolume, transformMatrix):
            # we also support the legacy property <url>
            uriComponents = urlparse(content.get("uri") or content.get("url"))
            
            contentExtension = splitext(uriComponents.path)[1].lower()
            
            if jsonOnly and contentExtension != ".json":
                return
            
            if uriComponents.query:
                self.uriQuery = uriComponents.query
            
            uri, baseUri = self.getUri(uriComponents, baseUri)
            
            try:
                if contentExtension == ".json":
                    tileset = json.loads((yield self.downloadRequest(uri)))
                    yield from self.renderTileset(tileset, baseUri, transformMatrix)
                elif contentExtension in (".glb", ".b3dm"):
                    consume = partial(self.consumeContent, uri, uriComponents.path, transformMatrix, contentExtension)
                    if self.concurrentDownloads:
                        yield Download(self.downloadRequest(uri), consume)
                    else:
                        consume((yield self.downloadRequest(uri)), None)
            except Exception as e:
                self.processError(e, uri)

    def downloadRequest(self, uri):
        # Preserve this provider adapter's existing gzip/TLS behavior; the
        # reusable transport itself verifies certificates and returns raw bytes.
        scope, reasons = '', ()
        if urlparse(uri).hostname == 'tile.googleapis.com':
            # The helper has no Google-specific policy. Only this adapter opts
            # into stopping a confirmed invalid credential/session scope.
            scope = sha256((self.constantUriQuery + '\n' + urlparse(uri).query).encode()).hexdigest()
            reasons = ('API_KEY_INVALID', 'API_KEY_EXPIRED', 'SESSION_EXPIRED', 'UNAUTHENTICATED', 'expired')
        return Request(uri, decode_gzip=True, verify_tls=False,
                       auth_scope=scope, auth_error_reasons=reasons)

    def consumeContent(self, uri, path, transformMatrix, extension, content, failure):
        try:
            if failure is not None:
                raise failure
            render = self.renderer.renderGlb if extension == '.glb' else self.renderer.renderB3dm
            render(self, uri, path, transformMatrix, content=content)
        except Exception as exc:
            self.processError(exc, uri)
    
    def getUri(self, uriComponents, baseUri):
        if uriComponents.netloc:
            baseUri = uriComponents.scheme + "://" + uriComponents.netloc
        else:
            baseUri = (self.uriServer if uriComponents.path[0] == '/' else baseUri)
        
        uri = baseUri + uriComponents.path
        _dirname = dirname(uriComponents.path)
        if _dirname:
            baseUri = baseUri + _dirname + '/' 
        
        if self.uriQuery and self.constantUriQuery:
            uri += '?' + self.uriQuery + '&' + self.constantUriQuery
        elif self.constantUriQuery:
            uri += '?' + self.constantUriQuery
        elif self.uriQuery:
            uri += '?' + self.uriQuery
        
        return uri, baseUri
    
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
        
        self.areaSizes = (self.areaSizeX, self.areaSizeY, self.areaSizeZ)
        
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
        
        self.areaMinLon = radians(minLon)
        self.areaMinLat = radians(minLat)
        self.areaMaxLon = radians(maxLon)
        self.areaMaxLat = radians(maxLat)
    
    def download(self, url):
        from ..downloads.access import require_online_access
        require_online_access()
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
        
        # check for gzip magic number
        if isinstance(response, bytes) and response[:2] == b"\x1f\x8b":
            # decompress the gzip file
            response = gzip.decompress(response)
        
        return response

    def redactError(self, error, uri):
        return redact_text(error, uri)
    
    @staticmethod
    def initMatrixFromTransform(transform):
        # Rember that the transform is in column-major order
        return Matrix((
            (transform[0], transform[4], transform[8], transform[12]),
            (transform[1], transform[5], transform[9], transform[13]),
            (transform[2], transform[6], transform[10], transform[14]),
            (transform[3], transform[7], transform[11], transform[15])
        ))
    
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
    
    def areaOverlapsWith(self, boundingVolume, transformMatrix):
        result = False
        
        if "box" in boundingVolume:
            result = self.areaOverlapsWithBbox(boundingVolume["box"], transformMatrix)
        elif "sphere" in boundingVolume:
            result = self.areaOverlapsWithSphere(boundingVolume["sphere"], transformMatrix)
        elif "region" in boundingVolume:
            result = self.areaOverlapsWithRegion(boundingVolume["region"])
        
        return result
        
    def areaOverlapsWithBbox(self, bbox, transformMatrix):
        
        bboxCenter = Vector((bbox[0], bbox[1], bbox[2]))
        # direction and half-length for the local X-axis of <bbox>
        bboxX = Vector((bbox[3], bbox[4], bbox[5]))
        # direction and half-length for the local Y-axis of <bbox>
        bboxY = Vector((bbox[6], bbox[7], bbox[8]))
        # direction and half-length for the local Z-axis of <bbox>
        bboxZ = Vector((bbox[9], bbox[10], bbox[11]))

        if transformMatrix:
            bboxCenter = transformMatrix @ bboxCenter
            bboxX = transformMatrix.to_3x3() @ bboxX
            bboxY = transformMatrix.to_3x3() @ bboxY
            bboxZ = transformMatrix.to_3x3() @ bboxZ
        
        bboxOrigin = bboxCenter - bboxX - bboxY - bboxZ
        
        # The coordinates are transformed to the system of reference where
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
    
    def areaOverlapsWithSphere(self, sphere, transformMatrix):
        sphereCenter, sphereRadius = Vector(sphere[:3]), sphere[3]
        if transformMatrix:
            sphereCenter = transformMatrix @ sphereCenter
        sphereRadiusSquared = sphereRadius * sphereRadius
        
        # A quick check if the spheres with the centers at <self.areaCenter> and <sphereCenter> and radii <self.areaRadius> and <sphereRadius>
        # respectively do not intersect. That means the bounding box of the area of interest and
        # the current sphere do not intersect either.
        # The condition for that is the following: the distance between <self.areaCenter> and <sphereCenter> is larger than the sum
        # of the radii <self.areaRadius> and <sphereRadius>.
        if (self.areaCenter - sphereCenter).length_squared > self.areaRadiusSquared + self.areaRadiusDoubled * sphereRadius + sphereRadius*sphereRadius:
            return False
        
        # The coordinates are transformed to the system of reference where
        # (1) the edges of the bounding box of the area of interest are parallel to the axes of the system of reference and
        # (2) the origin of the bounding box of the area of interest is located at the origin of that system of reference 
        sphereCenter = self.areaRotationMatrix @ ( sphereCenter - self.areaOrigin )
        
        # Another quick check.
        # Check if <sphereCenter> is within the bounding box of the area of interest. That means that either the surfaces of the sphere and
        # the bounding box of the area of interest intersect OR the sphere is entirely within the bounding box of the area of interest.
        
        if  0. <= sphereCenter[0] <= self.areaSizes[0] and\
            0. <= sphereCenter[1] <= self.areaSizes[1] and\
            0. <= sphereCenter[2] <= self.areaSizes[2]:
            return True
        #
        # The algorithm is from the paper "A Simple Method For Box-Sphere Intersection Testing" by James Arvo
        #
        
        # The minimum squared distance between <sphereCenter> and the bounding box of the area of interest
        dmin = 0.
        # The maximum squared distance between <sphereCenter> and the bounding box of the area of interest
        dmax = 0.
        for i in range(3):
            a = sphereCenter[i] * sphereCenter[i]
            b = (sphereCenter[i] - self.areaSizes[i]) * (sphereCenter[i] - self.areaSizes[i])
            dmax += max(a, b)
            if sphereCenter[i] < 0.:
                dmin += a
            elif sphereCenter[i] > self.areaSizes[i]:
                dmin += b
        
        # The first condition is for the intersection of surfaces of the sphere and the bounding box of the area of interest.
        # The second condition is for the case when the bounding box of the area of interest is entirely contained within the sphere.
        return dmin <= sphereRadiusSquared or dmax <= sphereRadiusSquared
    
    def areaOverlapsWithRegion(self, region):
        regionMinLon, regionMinLat, regionMaxLon, regionMaxLat, _, _ = region
        
        # Two regions do NOT overlap if one of the following conditions is true:
        # 1) One region is completely to the east of the other region
        # 2) One region is completely to the north of the other region
        
        return not (
            regionMinLon > self.areaMaxLon or self.areaMinLon > regionMaxLon or
            regionMinLat > self.areaMaxLat or self.areaMinLat > regionMaxLat
        )
    
    def processError(self, e, uri):
        self.errors.append(
            "There was an error when processing the URI %s: %s" % (redact_url(uri), self.redactError(e, uri))
        )
