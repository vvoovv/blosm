#!/usr/bin/env python3

import bpy, mathutils
import math, os

class MultipleImages():
	
	angleX = 45
	angleY = 45
	
	files = None
	camera = None
	dummyObject = None
	lattice = None
	zAxis = (0,0,1)
	
	# number of extra pixels to the left, top, right, bottom 
	extraPixels = 2

	latitude = 0
	longitude = 0
	heading = 0
	# multiplier is equal to 256/(2*math.pi*6378137*math.cos(math.radians(self.latitude)))
	multiplier = 0

	zoomMin = 17
	zoomMax = 19
	
	blenderBaseFile = "initial.blend" 
	blenderFilesDir = "."
	outputImagesDir = "models"
	csvFileDir = "."
	csvFile = "models.csv"
	csvFileHandle = None

	fileCounter = 0
	
	def __init__(self, **kwargs):
		for k in kwargs:
			setattr(self, k, kwargs[k])
		# check if outputImagesDir exists
		if not os.path.exists(self.outputImagesDir):
			os.makedirs(self.outputImagesDir)
		print(self.outputImagesDir)
		print(self.csvFileDir)
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
		# add lattice
		self.createLattice()
		# apply lattice modidier to all mesh and curve objects
		for o in bpy.context.scene.objects:
			if o.type == "MESH" or o.type == "CURVE":
				modifier = o.modifiers.new(name="Lattice", type="LATTICE")
				modifier.object = self.lattice
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
		self.multiplier = 256/(2*math.pi*6378137*math.cos(math.radians(self.latitude)))
		# clean up
		bpy.data.scenes.remove(scene)
		# adding objects and curves to the scene
		for o in data_to.objects:
			if o.type == "MESH" or o.type == "CURVE":
				bpy.context.scene.objects.link(o)
				o.select = True

	def createLattice(self):
		bpy.ops.object.select_all(action="DESELECT")
		bpy.ops.object.add(type="LATTICE")
		lattice = bpy.context.active_object
		lattice.data.points_u = 1
		lattice.data.points_v = 1
		bpy.ops.transform.resize(value=(100, 100, 100))
		# deform the lattice
		bpy.ops.object.mode_set(mode="EDIT")
		bpy.ops.lattice.select_all(action="SELECT")
		rotationMatrix = mathutils.Matrix.Rotation(math.radians(self.angleY), 4, "Y") * mathutils.Matrix.Rotation(math.radians(self.angleX), 4, "X")
		for p in lattice.data.points:
			p.co_deform = rotationMatrix * p.co_deform
		bpy.ops.object.mode_set(mode="OBJECT")
		self.lattice = lattice

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

	def getBoundingBox(self):
		# perform context.scene.update(), otherwise o.matrix_world or o.bound_box are incorrect
		bpy.context.scene.update()
		xmin = float("inf")
		ymin = float("inf")
		xmax = float("-inf")
		ymax = float("-inf")
		for o in bpy.context.scene.objects:
			if o.type == "MESH" or o.type == "CURVE":
				for v in o.bound_box:
					(x,y,z) = o.matrix_world * mathutils.Vector(v)
					if x<xmin: xmin = x
					elif x>xmax: xmax = x
					if y<ymin: ymin = y
					elif y>ymax: ymax = y
		return{"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}