import math, os
from urllib import request
import bpy


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
    
    def __init__(self, url, maxZoom, addonName):
        self.maxZoom = maxZoom
        self.subdomains = None
        self.numSubdomains = 0
        self.tileCounter = 0
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
    
    def doImport(self, l, b, r, t):
        def toTileCoords(coord, zoom):
            """
            An auxiliary function used below in the code
            """
            return int(math.floor(coord * math.pow(2, zoom) / equator))
        
        # Convert the coordinates from degrees to spherical Mercator coordinate system
        # and move zero to the top left corner (that's why the 3d argument in the function below)
        b, l = Overlay.toSphericalMercator(b, l, True)
        t, r = Overlay.toSphericalMercator(t, r, True)
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
                _l, _b, _r, _t = tuple(toTileCoords(coord, _zoom) for coord in (l, b, r, t))
                if (_r - _l + 1) * (_b - _t + 1) > self.maxNumTiles:
                    break
                zoom = _zoom
                _zoom += 1
        
        # convert <l>, <b>, <r>, <t> to tile coordinates
        l, b, r, t = tuple(toTileCoords(coord, zoom) for coord in (l, b, r, t))
        # get individual tiles
        for x in range(l, r+1):
            for y in range(t, b+1):
                tile = self.getTileImage(zoom, x, y)
        #bpy.data.images.new(
        #    "MyImage",
        #    width = (r - l + 1) * self.tileWidth,
        #    height = (b - t + 1) * self.tileHeight
        #)
    
    def getTileImage(self, zoom, x, y):
        # check if we the tile in the file cache
        j = os.path.join
        tileDir = j(self.overlayDir, str(zoom), str(x))
        tilePath = j(tileDir, "%s.%s" % (y, self.imageExtension))
        if os.path.exists(tilePath):
            return None
        else:
            tile = request.urlopen(self.getTileUrl(zoom, x, y)).read()
            # ensure that all directories in <tileDir> exist
            if not os.path.exists(tileDir):
                os.makedirs(tileDir)
            # save the tile to file cache
            with open(tilePath, 'wb') as f:
                f.write(tile)
        return tile
        
    
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
            self.tileCounter += 1
        else:
            url = "%s%s/%s/%s%s" % (
                self.urlStart,
                zoom,
                x,
                y,
                self.urlEnd
            )
        return url
    
    @staticmethod
    def toSphericalMercator(lat, lon, moveToTopLeft=False):
        lat = earthRadius * math.log(math.tan(math.pi/4 + lat*math.pi/360))
        lon = earthRadius * lon * math.pi / 180
        # move zero to the top left corner
        if moveToTopLeft:
            lat = halfEquator - lat
            lon = lon + halfEquator
        return lat, lon


from .mapbox import Mapbox


overlayTypeData = {
    'mapbox-satellite': (Mapbox, "mapbox.satellite", 19),
    'osm-mapnik': (Overlay, "http://{a,b,c}.tile.openstreetmap.org", 19),
    'mapbox-streets': (Mapbox, "mapbox.streets", 19),
    'custom': (Overlay, '', 19)
}