import math, os
from threading import Thread
import numpy
from urllib import request
import bpy

from util.blender import getBmesh, setBmesh, loadMaterialsFromFile
from app import app


earthRadius = 6378137.
halfEquator = math.pi * earthRadius
equator = 2. * halfEquator

# a Python dictionary to replace prohibited characters in a file name
prohibitedCharacters = {
    ord('/'):'!',
    ord('\\'):'!',
    ord(':'):'!',
    ord('*'):'!',
    ord('?'):'!',
    ord('"'):'!',
    ord('<'):'!',
    ord('>'):'!',
    ord('|'):'!'
}


class Overlay:
    
    tileWidth = 256
    tileHeight = 256
        
    maxNumTiles = 256 # i.e. 4096x4096 pixels
    
    tileCoordsTemplate = "{z}/{x}/{y}"
    
    blenderImageName = "overlay"
    
    # the name for the base UV map
    uvName = "UVMap"
    
    # relative path to default materials
    materialPath = "realistic/assets/base.blend"
    
    # name of the default material from <Overlay.materialPath>
    defaultMaterial = "overlay"
    
    def __init__(self, url, maxZoom, addonName):
        self.maxZoom = maxZoom
        self.subdomains = None
        self.numSubdomains = 0
        self.tileCounter = 0
        self.numTiles = 0
        self.imageExtension = "png"
        
        # where to stop searching for sundomains {suddomain1,subdomain2}
        subdomainsEnd = len(url)
        # check if have {z}/{x}/{y} in <url> (i.e. tile coords)
        coordsPosition = url.find(self.tileCoordsTemplate)
        if coordsPosition > 0:
            subdomainsEnd = coordsPosition
            urlEnd = url[coordsPosition+len(self.tileCoordsTemplate):]
        else:
            if url[-1] != '/':
                url = url + '/'
            urlEnd = ".png"
        leftBracketPosition = url.find("{", 0, subdomainsEnd)
        rightBracketPosition = url.find("}", leftBracketPosition+2, subdomainsEnd)
        if leftBracketPosition > -1 and rightBracketPosition > -1:
            self.subdomains = tuple(
                s.strip() for s in url[leftBracketPosition+1:rightBracketPosition].split(',')
            )
            self.numSubdomains = len(self.subdomains)
            urlStart = url[:leftBracketPosition]
            urlMid = url[rightBracketPosition+1:coordsPosition]\
                if coordsPosition > 0 else\
                url[rightBracketPosition+1:]
        else:
            urlStart = url[rightBracketPosition+1:coordsPosition] if coordsPosition > 0 else url
            urlMid = None
        self.urlStart = urlStart
        self.urlMid = urlMid
        self.urlEnd = urlEnd
    
    def prepareImport(self, left, bottom, right, top):
        app.print("Preparing overlay for import...")
        
        # Convert the coordinates from degrees to spherical Mercator coordinate system
        # and move zero to the top left corner (that's why the 3d argument in the function below)
        b, l = Overlay.toSphericalMercator(bottom, left, True)
        t, r = Overlay.toSphericalMercator(top, right, True)
        # find the maximum zoom
        zoom = int(math.floor(
            0.5 * math.log2(
                self.maxNumTiles * equator * equator / (b-t) / (r-l)
            )
        ))
        if zoom >= self.maxZoom:
            zoom = self.maxZoom
        else:
            _zoom = zoom + 1
            while _zoom <= self.maxZoom:
                # convert <l>, <b>, <r>, <t> to tile coordinates
                _l, _b, _r, _t = tuple(Overlay.toTileCoord(coord, _zoom) for coord in (l, b, r, t))
                if (_r - _l + 1) * (_b - _t + 1) > self.maxNumTiles:
                    break
                zoom = _zoom
                _zoom += 1
        
        self.zoom = zoom
        
        # convert <l>, <b>, <r>, <t> to tile coordinates
        l, b, r, t = tuple(Overlay.toTileCoord(coord, zoom) for coord in (l, b, r, t))
        self.l = l
        self.b = b
        self.r = r
        self.t = t
        self.numTilesX = numTilesX = r - l + 1
        self.numTilesY = numTilesY = b - t + 1
        self.numTiles = numTilesX * numTilesY
        # a numpy array for the resulting image stitched out of all tiles
        self.imageData = numpy.zeros(4*numTilesX*self.tileWidth * numTilesY*self.tileHeight)
        # four because of red, blue, green and opacity
        self.w = 4 * self.tileWidth
        # <self.x> and <self.y> are the current tile coordinates
        self.x = l
        self.y = t
    
    def importNextTile(self):
        w = self.w
        self.tileCounter += 1
        x = self.x
        y = self.y
        tileData = self.getTileData(self.zoom, x, y)
        if not tileData is None:
            for _y in range(self.tileHeight):
                i1 = w * ( (self.numTilesY-1-y+self.t) * self.tileHeight*self.numTilesX + _y*self.numTilesX + x - self.l )
                self.imageData[i1:i1+w] = tileData[_y*w:(_y+1)*w]
        if y == self.b:
            if x == self.r:
                return False
            else:
                self.x += 1
                self.y = self.t
        else:
            self.y += 1
        return True
    
    def finalizeImport(self):
        app.print("Stitching tile images...")
        
        # create the resulting Blender image stitched out of all tiles
        image = bpy.data.images.new(
            self.blenderImageName,
            width = (self.r - self.l + 1) * self.tileWidth,
            height = (self.b - self.t + 1) * self.tileHeight
        )
        image.pixels = self.imageData
        # cleanup
        self.imageData = None
        # pack the image into .blend file
        image.pack(as_png=True)
        
        if app.terrain:
            self.setUvForTerrain(
                app.terrain.terrain,
                Overlay.fromTileCoord(self.l, self.zoom) - halfEquator,
                halfEquator - Overlay.fromTileCoord(self.b+1, self.zoom),
                Overlay.fromTileCoord(self.r+1, self.zoom) - halfEquator,
                halfEquator - Overlay.fromTileCoord(self.t, self.zoom)
            )
        # load and append the default material
        if app.setOverlayMaterial:
            materials = app.terrain.terrain.data.materials
            material = loadMaterialsFromFile(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    os.pardir,
                    self.materialPath
                ),
                False, # i.e. append rather than link
                self.defaultMaterial
            )[0]
            material.node_tree.nodes["Image Texture"].image = image
            if materials:
                # ensure that <material> is placed at the very first material slot
                materials.append(None)
                materials[-1] = materials[0]
                materials[0] = material
            else:
                materials.append(material)
    
    def getTileData(self, zoom, x, y):
        # check if we the tile in the file cache
        j = os.path.join
        tileDir = j(self.overlayDir, str(zoom), str(x))
        tilePath = j(tileDir, "%s.%s" % (y, self.imageExtension))
        tileUrl = self.getTileUrl(zoom, x, y)
        if os.path.exists(tilePath):
            app.print(
                "(%s of %s) Using the cached version of the tile image %s" %
                (self.tileCounter, self.numTiles, tileUrl)
            )
        else:
            app.print(
                "(%s of %s) Downloading the tile image %s" %
                (self.tileCounter, self.numTiles, tileUrl)
            )
            try:
                tileData = request.urlopen(tileUrl).read()
            except:
                app.print(
                    "(%s of %s) Unable to download the tile image %s" %
                    (self.tileCounter, self.numTiles, tileUrl)
                )
                return None
            # ensure that all directories in <tileDir> exist
            if not os.path.exists(tileDir):
                os.makedirs(tileDir)
            # save the tile to file cache
            with open(tilePath, 'wb') as f:
                f.write(tileData)
        # Create a temporary Blender image out of the tile image
        # to create a numpy array out of the image raw data
        tmpImage = bpy.data.images.load(tilePath)
        tileData = numpy.array(tmpImage.pixels)
        # delete the temporary Blender image
        bpy.data.images.remove(tmpImage, True)
        return tileData
    
    def getOverlaySubDir(self):
        urlStart = self.urlStart
        if urlStart[:7] == "http://":
            urlStart = urlStart[7:]
        elif urlStart[:8] == "https://":
            urlStart = urlStart[8:]
        urlStart = urlStart.translate(prohibitedCharacters)
        return\
            "%s%s%s" % (urlStart, ''.join(self.subdomains), self.urlMid[:-1].translate(prohibitedCharacters))\
            if self.subdomains else\
            urlStart
    
    def getTileUrl(self, zoom, x, y):
        if self.subdomains:
            url = "%s%s%s%s/%s/%s%s" % (
                self.urlStart,
                self.subdomains[self.tileCounter % self.numSubdomains],
                self.urlMid,
                zoom,
                x,
                y,
                self.urlEnd
            )
        else:
            url = "%s%s/%s/%s%s" % (
                self.urlStart,
                zoom,
                x,
                y,
                self.urlEnd
            )
        return url
    
    def setUvForTerrain(self, terrain, l, b, r, t):
        bm = getBmesh(terrain)
        uv = bm.loops.layers.uv
        
        uvName = self.uvName
        # create a data UV layer
        if not uvName in uv:
            uv.new(uvName)
        
        width = r - l
        height = t - b
        uvLayer = bm.loops.layers.uv[uvName]
        worldMatrix = terrain.matrix_world
        projection = app.projection
        for vert in bm.verts:
            for loop in vert.link_loops:
                x, y = (worldMatrix * vert.co)[:2]
                lat, lon = projection.toGeographic(x, y)
                lat, lon = Overlay.toSphericalMercator(lat, lon, False)
                loop[uvLayer].uv = (lon - l)/width, (lat - b)/height
        
        setBmesh(terrain, bm)
    
    @staticmethod
    def toSphericalMercator(lat, lon, moveToTopLeft=False):
        lat = earthRadius * math.log(math.tan(math.pi/4 + lat*math.pi/360))
        lon = earthRadius * lon * math.pi / 180
        # move zero to the top left corner
        if moveToTopLeft:
            lat = halfEquator - lat
            lon = lon + halfEquator
        return lat, lon
    
    @staticmethod
    def toTileCoord(coord, zoom):
        """
        An auxiliary method used in the code.
        
        Converts a coordinate <coord> in the spherical Mercator coordinate system
        with the origin moved to the top left corner, to the integer tile coordinate.
        The same tile coordinates are used be the map at http://osm.org
        """
        coord = coord * math.pow(2., zoom) / equator
        return int(math.floor(coord))
    
    @staticmethod
    def fromTileCoord(coord, zoom):
        """
        An auxiliary method used in the code.
        
        Reversed to <Overlay.toTileCoord>
        """
        return coord * equator / math.pow(2., zoom)


from .mapbox import Mapbox


overlayTypeData = {
    'mapbox-satellite': (Mapbox, "mapbox.satellite", 19),
    'osm-mapnik': (Overlay, "http://{a,b,c}.tile.openstreetmap.org", 19),
    'mapbox-streets': (Mapbox, "mapbox.streets", 19),
    'custom': (Overlay, '', 19)
}