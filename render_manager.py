#!/usr/bin/env python3

import bpy, mathutils

class RenderManager():
	
	files = None
	camera = None
	dummyObject = None
	zAxis = (0,0,1)

	blenderFilesDir = "."
	
	radius = 6378137
	
	def __init__(self, **kwargs):
		for k in kwargs:
			setattr(self, k, kwargs[k])

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
		return {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}