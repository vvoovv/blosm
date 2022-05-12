import os
import numpy
import bpy
from . import Overlay, halfEquator
from util.blender import getBmesh, setBmesh, loadMaterialsFromFile


class OverlayMixin:

    def finalizeImport(self):
        app = self.app
        
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
        
        if app.saveOverlayToFile:
            path = os.path.join(app.dataDir, "texture", f"overlay.{image.file_format.lower()}")
            image.save_render(path)
            image.source = 'FILE'
            image.filepath = path
        else:
            # pack the image into .blend file
            image.pack()
        
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
        projection = self.app.projection
        for vert in bm.verts:
            for loop in vert.link_loops:
                x, y = (worldMatrix @ vert.co)[:2]
                lat, lon = projection.toGeographic(x, y)
                lat, lon = Overlay.toSphericalMercator(lat, lon, False)
                loop[uvLayer].uv = (lon - l)/width, (lat - b)/height
        
        setBmesh(terrain, bm)
    
    def getTileDataFromImage(self, tilePath):
        # Create a temporary Blender image out of the tile image
        # to create a numpy array out of the image raw data
        tmpImage = bpy.data.images.load(tilePath)
        tileData = numpy.array(tmpImage.pixels)
        # delete the temporary Blender image
        bpy.data.images.remove(tmpImage, do_unlink=True)
        return tileData