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

import math, os, sys
from threading import Thread
import numpy
from urllib import request
import bpy

from util.blender import getBmesh, setBmesh, loadMaterialsFromFile
from app import app

_isBlender280 = bpy.app.version[1] >= 80


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
    
    def __init__(self, url, maxZoom, addonName):
        self.maxZoom = maxZoom
        self.subdomains = None
        self.numSubdomains = 0
        self.tileCounter = 0
        self.numTiles = 0
        self.imageExtension = "png"
        
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
        app.print("Preparing overlay for import...")
        
        maxNumTiles = app.maxNumTiles
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
        if tileData is True or tileData is False:
            # Boolean in <tileData> means that we have to restart everything from the beginning since 
            # the tiles aren't available for the given zoom.
            # Return immediately.
            return tileData
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
        if self.imageData is None:
            return False 
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
        if _isBlender280:
            image.pack()
        else:
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
        return True
    
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
            req = request.Request(
                tileUrl,
                data = None,
                headers = {
                    "User-Agent": "custom"
                }
            )
            try:
                tileData = request.urlopen(req).read()
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
        bpy.data.images.remove(tmpImage, do_unlink=True)
        return tileData
    
    def getOverlaySubDir(self):
        url = self.url
        urlOffset = 0
        if url[:7] == "http://":
            urlOffset = 7
        elif url[:8] == "https://":
            urlOffset = 8
        else:
            urlOffset = 0
        return url[urlOffset:].translate(prohibitedCharacters)
    
    def getTileUrl(self, zoom, x, y):
        return self.url.format(
                sub=self.subdomains[self.tileCounter % self.numSubdomains],
                z=zoom, x=x, y=y
            ) if self.subdomains else\
            self.url.format(z=zoom, x=x, y=y)
    
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
                x, y = (worldMatrix @ vert.co)[:2] if _isBlender280 else (worldMatrix * vert.co)[:2]
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
    'osm-mapnik': (Overlay, "http://[a,b,c].tile.openstreetmap.org", 19),
    'mapbox-streets': (Mapbox, "mapbox.streets", 19),
    'custom': (Overlay, '', 19)
}