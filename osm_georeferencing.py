#!/usr/bin/env python3

import bpy
import math
from transverse_mercator import TransverseMercator

bl_info = {
	"name": "OpenStreetMap Georeferencing",
	"author": "Vladimir Elistratov <vladimir.elistratov@gmail.com>",
	"version": (1, 0, 0),
	"blender": (2, 6, 8),
	"location": "View 3D > Object Mode > Tool Shelf",
	"description" : "OpenStreetMap based object georeferencing",
	"warning": "",
	"wiki_url": "",
	"tracker_url": "",
	"support": "COMMUNITY",
	"category": "3D View",
}

class OsmGeoreferencingPanel(bpy.types.Panel):
	bl_space_type = "VIEW_3D"
	bl_region_type = "TOOLS"
	#bl_context = "object"
	bl_label = "Georeferencing"

	def draw(self, context):
		l = self.layout
		c = l.column()
		c.operator("object.set_main_position")
		c.operator("object.do_georeferencing")

class SetMainPosition(bpy.types.Operator):
	bl_idname = "object.set_main_position"
	bl_label = "Active object postion"
	bl_description = "Remember active object position"

	def execute(self, context):
		bpy.p = bpy.context.active_object.location.copy()
		return {"FINISHED"}

class DoGeoreferencing(bpy.types.Operator):
	bl_idname = "object.do_georeferencing"    
	bl_label = "Perform georeferencing"
	bl_description = "Perform georeferencing"

	def execute(self, context):
		# calculationg new position of the object center
		p = bpy.context.active_object.matrix_world * (-bpy.p)
		projection = TransverseMercator(lon=bpy.longitude, lat=bpy.latitude)
		(lat, lon) = projection.toGeographic(p[0], p[1])
		bpy.context.scene["longitude"] = lon
		bpy.context.scene["latitude"] = lat
		bpy.context.scene["heading"] = bpy.context.active_object.rotation_euler[2]*180/math.pi
		return {"FINISHED"}

def register():
	bpy.utils.register_module(__name__)

def unregister():
	bpy.utils.unregister_module(__name__)