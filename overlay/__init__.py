"""
This file is part of blender-osm (OpenStreetMap importer for Blender).
Copyright (C) 2014-2018 Vladimir Elistratov
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
import math, os, sys, ssl
import numpy
from threading import Thread
from urllib import request


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
    
    minZoom = 1
    
    tileWidth = 256
    tileHeight = 256
    
    # default template
    tileCoordsTemplate = "{z}/{x}/{y}.png"
    
    blenderImageName = "overlay.png"
    
    # the name for the base UV map
    uvName = "UVMap"
    
    # relative path to default materials
    materialPath = "assets/base.blend"
    
    # name of the default material from <Overlay.materialPath>
    defaultMaterial = "overlay"
    
    def __init__(self, url, maxZoom, app):
        self.maxZoom = maxZoom
        self.app = app
        self.subdomains = None
        self.numSubdomains = 0
        self.tileCounter = 0
        self.numTiles = 0
        # four because of red, blue, green and opacity
        self.numComponents = 4
        self.imageExtension = "png"
        # Defines if the origin of an image is located at its top left corner (True) or
        # bottom left corner (False)
        self.originAtTop = False
        
        self.checkImageFormat = False 
        
        if app.overlayType == "custom":
            if "jpg" in url:
                self.imageExtension = "jpg"
            elif "jpeg" in url:
                self.imageExtension = "jpeg"
            elif not "png" in url:
                # The image tile format is unknown. We have to check it for the command line mode
                self.checkImageFormat = True
        
        url = url.strip()
        
        if url.startswith("mapbox://styles/"):
            # the special case of a Mapbox style URL
            url = "https://api.mapbox.com/styles/v1/%s/tiles/256/{z}/{x}/{y}?access_token=%s" %\
                (url[16:], self.app.getMapboxAccessToken())
        else:
            # check we have subdomains
            leftBracketPosition = url.find("[")
            rightBracketPosition = url.find("]", leftBracketPosition+2)
            if leftBracketPosition > -1 and rightBracketPosition > -1: 
                subdomains = tuple(
                    s.strip() for s in url[leftBracketPosition+1:rightBracketPosition].split(',')
                )
                if subdomains:
                    self.subdomains = subdomains
                    self.numSubdomains = len(self.subdomains)
                    url = "%s{sub}%s" % (url[:leftBracketPosition], url[rightBracketPosition+1:])
            # check if have {z} and {x} and {y} in <url> (i.e. tile coords)
            if not ("{z}" in url and "{x}" in url and "{y}" in url):
                if url[-1] != '/':
                    url = url + '/'
                url = url + self.tileCoordsTemplate
        
        self.url = url
    
    def prepareImport(self, left, bottom, right, top):
        self.app.print("Preparing overlay for import...")
        
        maxNumTiles = self.app.maxNumTiles
        # Convert the coordinates from degrees to spherical Mercator coordinate system
        # and move zero to the top left corner (that's why the 3d argument in the function below)
        b, l = Overlay.toSphericalMercator(bottom, left, True)
        t, r = Overlay.toSphericalMercator(top, right, True)
        # remember <l>, <b>, <r>, <t>
        self.left = l
        self.bottom = b
        self.right = r
        self.top = t
        # find the maximum zoom
        zoom = int(math.floor(
            0.5 * math.log2(
                maxNumTiles * equator * equator / (b-t) / (r-l)
            )
        ))
        if zoom >= self.maxZoom:
            zoom = self.maxZoom
        else:
            _zoom = zoom + 1
            while _zoom <= self.maxZoom:
                # convert <l>, <b>, <r>, <t> to tile coordinates
                _l, _b, _r, _t = tuple(Overlay.toTileCoord(coord, _zoom) for coord in (l, b, r, t))
                if (_r - _l + 1) * (_b - _t + 1) > maxNumTiles:
                    break
                zoom = _zoom
                _zoom += 1
        
        self.setParameters(zoom)
        
    def setParameters(self, zoom):
        self.tileCounter = 0
        self.zoom = zoom
        
        # convert <l>, <b>, <r>, <t> to tile coordinates
        l, b, r, t = tuple(
            Overlay.toTileCoord(coord, zoom) for coord in (self.left, self.bottom, self.right, self.top)
        )
        self.l = l
        self.b = b
        self.r = r
        self.t = t
        self.numTilesX = numTilesX = r - l + 1
        self.numTilesY = numTilesY = b - t + 1
        self.numTiles = numTilesX * numTilesY
        self.initImageData()
        # <self.x> and <self.y> are the current tile coordinates
        self.x = l
        self.y = t
    
    def initImageData(self):
        self.w = self.numComponents * self.tileWidth
        # a numpy array for the resulting image stitched out of all tiles
        self.imageData = numpy.zeros(
            self.w * self.numTilesX * self.numTilesY * self.tileHeight,
            dtype = numpy.float64 if self.numComponents == 4 else numpy.uint8
        )
    
    def importNextTile(self):
        self.tileCounter += 1
        x = self.x
        y = self.y
        tileData = self.getTileData(self.zoom, x, y)
        w = self.w
        if tileData is True or tileData is False:
            # Boolean in <tileData> means that we have to restart everything from the beginning since 
            # the tiles aren't available for the given zoom.
            # Return immediately.
            return tileData
        if not tileData is None:
            for _y in range(self.tileHeight):
                i1 = w * ( (y-self.t) * self.tileHeight*self.numTilesX + _y*self.numTilesX + x - self.l )\
                if self.originAtTop else\
                w * ( (self.numTilesY-1-y+self.t) * self.tileHeight*self.numTilesX + _y*self.numTilesX + x - self.l )
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
    
    def getTileData(self, zoom, x, y):
        # check if we the tile in the file cache
        tileDir = os.path.join(self.overlayDir, str(zoom), str(x))
        if self.checkImageFormat and os.path.exists(tileDir):
            # Tile image format is unknown. We'll get from the the file extension
            # of the tile image
            self.getImageExtensionFromFile(tileDir)
        tilePath = os.path.join(tileDir, "%s.%s" % (y, self.imageExtension))
        tileUrl = self.getTileUrl(zoom, x, y)
        if os.path.exists(tilePath):
            self.app.print(
                "(%s of %s) Using the cached version of the tile image %s" %
                (self.tileCounter, self.numTiles, tileUrl)
            )
        else:
            self.app.print(
                "(%s of %s) Downloading the tile image %s" %
                (self.tileCounter, self.numTiles, tileUrl)
            )
            req = request.Request(
                tileUrl,
                data = None,
                headers = {
                    "User-Agent": "custom"
                }
            )
            try:
                # a hack to avoid CERTIFICATE_VERIFY_FAILED error
                ctx = ssl._create_unverified_context()
                tileData = request.urlopen(req, context=ctx).read()
            except:
                if getattr(sys.exc_info()[1], "code", None) == 404:
                    # The error code 404 means that the tile doesn't exist
                    # Probably the tiles for <self.zoom> aren't available at all
                    # Let's try to decrease <self.zoom>
                    if self.zoom == self.minZoom:
                        # probably something is wrong with the server
                        self.imageData = None
                        return False
                    self.setParameters(self.zoom-1)
                    # Returning Python boolean means that we have to restart everything from the beginning since 
                    # the tiles aren't available for the given zoom.
                    return True
                self.app.print(
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
        
        return self.getTileDataFromImage(tilePath)
    
    def getOverlaySubDir(self):
        url = self.url
        url = self.removeAccessToken(url)
        urlOffset = 0
        if url[:7] == "http://":
            urlOffset = 7
        elif url[:8] == "https://":
            urlOffset = 8
        else:
            urlOffset = 0
        return url[urlOffset:].translate(prohibitedCharacters)
    
    def removeAccessToken(self, url):
        # nothing to do here
        return url
    
    def getTileUrl(self, zoom, x, y):
        return self.url.format(
                sub=self.subdomains[self.tileCounter % self.numSubdomains],
                z=zoom, x=x, y=y
            ) if self.subdomains else\
            self.url.format(z=zoom, x=x, y=y)
    
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
    def fromSphericalMercator(lat, lon):
        return\
            math.degrees(
                math.atan( math.exp(lat/earthRadius) ) * 2. - math.pi/2.
            ),\
            math.degrees(lon/earthRadius)
    
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
    
    def fromTileCoordsToAppCoords(self):
        # Convert tile coordinates to the coordinates in the spherical Mercator coordinate system
        l = Overlay.fromTileCoord(self.l, self.zoom) - halfEquator
        b = halfEquator - Overlay.fromTileCoord(self.b+1, self.zoom)
        r = Overlay.fromTileCoord(self.r+1, self.zoom) - halfEquator
        t = halfEquator - Overlay.fromTileCoord(self.t, self.zoom)
        
        # Convert the coordinates in the spherical Mercator coordinate system
        # to geographical coordinates (latitude and longitude)
        b, l = Overlay.fromSphericalMercator(b, l)
        t, r = Overlay.fromSphericalMercator(t, r)
        
        # Convert the geographical coordinates to the coordinates in application's system of reference
        l, b = self.app.projection.fromGeographic(b, l)
        r, t = self.app.projection.fromGeographic(t, r)
        
        return (l, r, b, t)
    
    def getImageExtensionFromFile(self, tileDir):
        images = os.listdir(tileDir)
        if images:
            # we use the extension of the first file in <tileDir>
            self.imageExtension = os.path.splitext(images[0])[1][1:]
            if self.imageExtension == "jpg":
                self.numComponents = 3
                self.initImageData()
            self.checkImageFormat = False


from .mapbox import Mapbox
from .arcgis import Arcgis


overlayTypeData = {
    'mapbox-satellite': (Mapbox, "mapbox.satellite", 19),
    'arcgis-satellite': (Arcgis, "World_Imagery", 19),
    'osm-mapnik': (Overlay, "http://[a,b,c].tile.openstreetmap.org", 19),
    'mapbox-streets': (Overlay, "mapbox://styles/mapbox/streets-v11", 19),
    'custom': (Overlay, '', 20)
}