#!/usr/bin/env python3

import bpy, mathutils
import math, os

from map_25d import Map25D

class MultipleImages(Map25D):

	latitude = 0
	longitude = 0
	heading = 0
	# multiplier is equal to 256/(2*math.pi*self.radius*math.cos(math.radians(self.latitude)))
	multiplier = 0

	outputImagesDir = "models"
	csvFileDir = "."
	csvFile = "models.csv"
	csvFileHandle = None

	fileCounter = 0
	
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		# check if outputImagesDir exists
		if not os.path.exists(self.outputImagesDir):
			os.makedirs(self.outputImagesDir)
		# check if csvFileDir exists
		if not os.path.exists(self.csvFileDir):
			os.makedirs(self.csvFileDir)

	def render(self):
		# start csvFile
		self.csvFileHandle = open(os.path.join(self.csvFileDir, self.csvFile), "w")
		# compose headers, i.e. the first line in the csvFile
		self.csvFileHandle.write("modelId,lat,lon")
		for z in range(self.zoomMin, self.zoomMax+1):
			self.csvFileHandle.write(",image_z%s,dx_z%s,dy_z%s" % (z,z,z))

		for filename in self.files:
			# self.blenderBaseFile is located next to this script
			bpy.ops.wm.open_mainfile(filepath=os.path.join(os.path.dirname(os.path.realpath(__file__)), self.blenderBaseFile))
			self.camera = bpy.data.objects["Camera"]
			# setting the dummy object; it will help to perform transformations later in the code
			# setting pivot_point doesn't work if done from a python script
			self.dummyObject = bpy.data.objects["Empty"]
	
			bpy.ops.object.select_all(action="DESELECT")
			bpy.context.scene.cursor_location = (0, 0, 0)
			
			self.renderFile(filename)

		self.csvFileHandle.close()

	def renderFile(self, filename):
		self.fileCounter += 1
		self.loadFile(os.path.join(self.blenderFilesDir, filename))
		self.dummyObject.select = True
		# all needed objects from the loaded file and the dummyObject are selected
		# now apply heading
		bpy.ops.transform.rotate(value=self.heading, axis=self.zAxis)
		bpy.ops.transform.translate(value=-self.dummyObject.location)
		# apply lattice modidier to all mesh and curve objects
		self.applyLatticeModifier()
		# render images
		zoomInfo = [] # used to compose self.csvFile
		self.renderImages(zoomInfo)
		# compose a line for self.csvFile
		self.csvFileHandle.write("\n%s,%s,%s" % (self.fileCounter, self.latitude, self.longitude))
		for z in zoomInfo:
			self.csvFileHandle.write(",%s,%s,%s" % (z[0], z[1], z[2]))

	def loadFile(self, filename):
		with bpy.data.libraries.load(filename, link=False) as (data_from, data_to):
			data_to.objects = data_from.objects
			data_to.scenes = data_from.scenes
		# reading latitude, longitude and heading from the first scene
		scene = data_to.scenes[0]
		self.latitude = scene["latitude"]
		self.longitude = scene["longitude"]
		self.heading = math.radians(scene["heading"])
		# calculate multiplier
		self.multiplier = 256/(2*math.pi*self.radius*math.cos(math.radians(self.latitude)))
		# clean up
		bpy.data.scenes.remove(scene)
		# adding objects and curves to the scene
		for o in data_to.objects:
			if o.type == "MESH" or o.type == "CURVE":
				bpy.context.scene.objects.link(o)
				o.select = True

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
		extraMeters = self.extraPixels/multiplier
		bb = {
			"xmin": bbox["xmin"] - extraMeters,
			"ymin": bbox["ymin"] - extraMeters,
			"xmax": bbox["xmax"] + extraMeters,
			"ymax": bbox["ymax"] + extraMeters
		}
		(imageWidth, imageHeight, width, height, imageFile) = self.setSizes(bb, zoom, multiplier)
		bpy.ops.render.render(write_still=True)
		# shift between image center and object center (in pixels)
		dx = imageWidth * (bb["xmin"]+bb["xmax"]) / (2*width)
		dy = -imageHeight * (bb["ymin"]+bb["ymax"]) / (2*height)
		zoomInfo.append((imageFile, round(dx,2), round(dy,2)))

	def getImageName(self, zoom):
		return "%s_%s" % (self.fileCounter, zoom)