#!/usr/bin/env python3

import bpy, mathutils
import math, os

from render_manager import RenderManager

class Map25D(RenderManager):

	angleX = 45
	angleY = 45
	
	zoomMin = 17
	zoomMax = 19
	
	# number of extra pixels to the left, top, right, bottom 
	extraPixels = 2

	blenderBaseFile = "initial.blend"

	lattice = None

	def __init__(self, **kwargs):
		super().__init__(**kwargs)

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

	def applyLatticeModifier(self):
		# add lattice
		self.createLattice()
		# apply lattice modidier to all mesh and curve objects
		for o in bpy.context.scene.objects:
			if o.type == "MESH" or o.type == "CURVE":
				modifier = o.modifiers.new(name="Lattice", type="LATTICE")
				modifier.object = self.lattice

	def setSizes(self, bbox, zoom, multiplier):
		# setting resulting image size
		render = bpy.context.scene.render
		# bbox dimensions
		width = bbox["xmax"]-bbox["xmin"]
		height = bbox["ymax"]-bbox["ymin"]
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
		return (imageWidth, imageHeight, width, height, imageFile)