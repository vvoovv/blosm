bl_info = {
	"name": "OpenStreetMap Georeferencing",
	"author": "Vladimir Elistratov <vladimir.elistratov@gmail.com>",
	"version": (1, 0, 0),
	"blender": (2, 6, 9),
	"location": "View 3D > Object Mode > Tool Shelf",
	"description": "OpenStreetMap based object georeferencing",
	"warning": "",
	"wiki_url": "https://github.com/vvoovv/blender-geo/wiki/OpenStreetMap-Georeferencing",
	"tracker_url": "https://github.com/vvoovv/blender-geo/issues",
	"support": "COMMUNITY",
	"category": "3D View",
}

import bpy
import math

import sys
sys.path.append("D:\\projects\\blender\\blender-geo")
from transverse_mercator import TransverseMercator

class OsmGeoreferencingPanel(bpy.types.Panel):
	bl_space_type = "VIEW_3D"
	bl_region_type = "TOOLS"
	bl_context = "objectmode"
	bl_category = "Geo"
	bl_label = "Georeferencing"
	
	def draw(self, context):
		layout = self.layout
		
		layout.row().operator("object.set_original_position")
		row = layout.row()
		if not (
			# the original position is set
			_.refObjectData and
			# custom properties "latitude" and "longitude" are set for the active scene
			"latitude" in context.scene and "longitude" in context.scene and
			# objects belonging to the model are selected
			len(context.selected_objects)>0
		):
			row.enabled = False
		row.operator("object.do_georeferencing")

class SetOriginalPosition(bpy.types.Operator):
	bl_idname = "object.set_original_position"
	bl_label = "Set original position"
	bl_description = "Remember original position"

	def execute(self, context):
		if len(context.selected_objects)==0:
			self.report({"ERROR"}, "Select objects belonging to your model")
			return {"FINISHED"}
		# remember the location and orientation of the reference object
		# take the first selected object as a reference object
		refObject = context.selected_objects[0]
		refObjectData = (refObject, refObject.location.copy(), refObject.rotation_euler[2])
		_.refObjectData = refObjectData
		return {"FINISHED"}

class DoGeoreferencing(bpy.types.Operator):
	bl_idname = "object.do_georeferencing"    
	bl_label = "Perform georeferencing"
	bl_description = "Perform georeferencing"
	bl_options = {"UNDO"}

	def execute(self, context):
		scene = context.scene
		refObjectData = _.refObjectData
		refObject = refObjectData[0]
		# calculationg new position of the reference object center
		p = refObject.matrix_world * (-refObjectData[1])
		projection = TransverseMercator(lat=scene["latitude"], lon=scene["longitude"])
		(lat, lon) = projection.toGeographic(p[0], p[1])
		scene["longitude"] = lon
		scene["latitude"] = lat
		scene["heading"] = (refObject.rotation_euler[2]-refObjectData[2])*180/math.pi
		
		# restoring original objects location and orientation
		bpy.ops.transform.rotate(value=-(refObject.rotation_euler[2]-refObjectData[2]), axis=(0,0,1))
		bpy.ops.transform.translate(value=-(refObject.location-refObjectData[1]))
		# cleaning up
		_.refObjectData = None
		return {"FINISHED"}

class _:
	"""An auxiliary class to store plugin data"""
	refObjectData = None

def register():
	bpy.utils.register_module(__name__)

def unregister():
	bpy.utils.unregister_module(__name__)