#!/usr/bin/env python3

import bpy, mathutils
import math, os
from spherical_mercator import SphericalMercator

from map_25d import Map25D

class CommonImage(Map25D):

	# position of the Blender zero point in the Mercator projection 
	x = 0
	y = 0
	# the Blender zero point is set to the first object
	zeroPointSet = False

	outputDir = "models"
	
	projection = SphericalMercator()

	fileCounter = 0
	
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		# check if outputDir exists
		if not os.path.exists(self.outputDir):
			os.makedirs(self.outputDir)

	def render(self):
		for z in range(self.zoomMin, self.zoomMax+1):
			pass

		# self.blenderBaseFile is located next to this script
		#bpy.ops.wm.open_mainfile(filepath=os.path.join(os.path.dirname(os.path.realpath(__file__)), self.blenderBaseFile))
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
		# calculating scene bounding box
		bb = self.getBoundingBox()
		print(bb)

	def addFile(self, filename):
		self.fileCounter += 1
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

	def renderImages(self, zoomInfo):
		# calculating scene bounding box
		bb = self.getBoundingBox()
		# place camera at the center of bbox
		self.camera.location.x = (bb["xmin"]+bb["xmax"])/2
		self.camera.location.y = (bb["ymin"]+bb["ymax"])/2
		# render resolution
		bpy.context.scene.render.resolution_percentage = 100
		for z in range(self.zoomMin, self.zoomMax+1):
			self.renderImage(z, bb, zoomInfo)

	def renderImage(self, zoom, bbox, zoomInfo):
		multiplier = self.multiplier * math.pow(2, zoom)

		# correcting bbox, taking into account self.extraPixels
		bb = {}
		extraMeters = self.extraPixels/multiplier
		bb["xmin"] = bbox["xmin"] - extraMeters
		bb["ymin"] = bbox["ymin"] - extraMeters
		bb["xmax"] = bbox["xmax"] + extraMeters
		bb["ymax"] = bbox["ymax"] + extraMeters
		# setting resulting image size
		render = bpy.context.scene.render
		# bbox dimensions
		width = bb["xmax"]-bb["xmin"]
		height = bb["ymax"]-bb["ymin"]
		# camera's ortho_scale property
		self.camera.data.ortho_scale = width if width > height else height
		# image width and height
		imageWidth = multiplier * width
		imageHeight = multiplier * height
		render.resolution_x = imageWidth
		render.resolution_y = imageHeight
		# image name
		imageFile = self.getImageName(zoom)
		render.filepath = os.path.join(self.outputImagesDir, imageFile)
		bpy.ops.render.render(write_still=True)
		# shift between image center and object center (in pixels)
		dx = imageWidth * (bb["xmin"]+bb["xmax"]) / (2*width)
		dy = -imageHeight * (bb["ymin"]+bb["ymax"]) / (2*height)
		zoomInfo.append((imageFile, round(dx,2), round(dy,2)))

	def getImageName(self, zoom):
		return "%s_%s" % (self.fileCounter, zoom)