#!/usr/bin/env python3

import bpy, mathutils
import math, os, subprocess
from spherical_mercator import SphericalMercator

from map_25d import Map25D

class CommonImage(Map25D):

	outputImagesDir = "."

	gdalDir = "."

	# position of the Blender zero point in the Mercator projection 
	x = 0
	y = 0
	# the Blender zero point is set to the first object
	zeroPointSet = False
	
	projection = SphericalMercator()
	
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		# check if outputImagesDir exists
		if not os.path.exists(self.outputImagesDir):
			os.makedirs(self.outputImagesDir)

	def render(self):
		# self.blenderBaseFile is located next to this script
		bpy.ops.wm.open_mainfile(filepath=os.path.join(os.path.dirname(os.path.realpath(__file__)), self.blenderBaseFile))
		self.camera = bpy.data.objects["Camera"]
		# setting the dummy object; it will help to perform transformations later in the code
		# setting pivot_point doesn't work if done from a python script
		self.dummyObject = bpy.data.objects["Empty"]

		for filename in self.files:
			bpy.ops.object.select_all(action="DESELECT")
			self.dummyObject.location = (0,0,0)
			bpy.context.scene.cursor_location = (0, 0, 0)
			self.addFile(filename)
		# apply lattice modidier to all mesh and curve objects
		self.applyLatticeModifier()
		# calculating scene bounding box in local coordinates, "l" stands for "local"
		lbb = self.getBoundingBox()
		# place camera at the center of bbox
		self.camera.location.x = (lbb["xmin"]+lbb["xmax"])/2
		self.camera.location.y = (lbb["ymin"]+lbb["ymax"])/2
		# render resolution
		bpy.context.scene.render.resolution_percentage = 100
		# converting lbb to spherical Mercator coordinates, "m" stands for Mercator
		mbb = {
			"xmin": lbb["xmin"] + self.x,
			"ymin": lbb["ymin"] + self.y,
			"xmax": lbb["xmax"] + self.x,
			"ymax": lbb["ymax"] + self.y
		}
		render = bpy.context.scene.render
		for zoom in range(self.zoomMin, self.zoomMax+1):
			# adding self.extraPixels
			multiplier = 256*math.pow(2, zoom) / (2*math.pi*self.radius)
			bb = {
				"xmin": mbb["xmin"] - self.extraPixels/multiplier,
				"ymin": mbb["ymin"] - self.extraPixels/multiplier,
				"xmax": mbb["xmax"] + self.extraPixels/multiplier,
				"ymax": mbb["ymax"] + self.extraPixels/multiplier
			}
			(imageWidth, imageHeight, width, height, imageFile) = self.setSizes(bb, zoom, multiplier)
			bpy.ops.render.render(write_still=True)
			# perform georeferencing with gdal_translate
			imageFile = os.path.join(self.outputImagesDir, imageFile)
			subprocess.call([
				os.path.join(self.gdalDir, "gdal_translate"),
				"-a_ullr", str(bb["xmin"]), str(bb["ymax"]), str(bb["xmax"]), str(bb["ymin"]),
				imageFile,
				os.path.join(self.outputImagesDir, imageFile[0:imageFile.rfind(".")]+".tif") # TIFF file
			])
			# get rid of the imageFile
			os.remove(imageFile)

	def addFile(self, filename):
		# latitude and longitude in degrees, heading in radians
		(latitude, longitude, heading) = self.loadFile(os.path.join(self.blenderFilesDir, filename))
		self.dummyObject.select = True
		# all needed objects from the loaded file and the dummyObject are selected
		# now apply heading
		bpy.ops.transform.rotate(value=heading, axis=self.zAxis)
		# apply scaling
		scale = 1/math.cos(math.radians(latitude))
		bpy.ops.transform.resize(value=(scale, scale, scale))
		bpy.ops.transform.translate(value=-self.dummyObject.location)
		# check if the Blender zero point is set
		if self.zeroPointSet:
			# move object to its position according to the Mercator projection
			(x, y) = self.projection.fromGeographic((latitude, longitude))
			bpy.ops.transform.translate(value=(x-self.x, y-self.y, 0))
		else:
			(self.x, self.y) = self.projection.fromGeographic((latitude, longitude))
			self.zeroPointSet = True

	def loadFile(self, filename):
		with bpy.data.libraries.load(filename, link=False) as (data_from, data_to):
			data_to.objects = data_from.objects
			data_to.scenes = data_from.scenes
		# reading latitude, longitude and heading from the first scene
		scene = data_to.scenes[0]
		latitude = scene["latitude"]
		longitude = scene["longitude"]
		heading = math.radians(scene["heading"])
		# clean up
		bpy.data.scenes.remove(scene)
		# adding objects and curves to the scene
		for o in data_to.objects:
			if o.type == "MESH" or o.type == "CURVE":
				bpy.context.scene.objects.link(o)
				o.select = True
		return (latitude, longitude, heading)

	def getImageName(self, zoom):
		return "raster_%s.png" % zoom