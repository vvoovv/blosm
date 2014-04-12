bl_info = {
	"name": "Target Scaling",
	"author": "Vladimir Elistratov <vladimir.elistratov@gmail.com>",
	"version": (1, 0, 0),
	"blender": (2, 6, 9),
	"location": "View 3D > Edit Mode > Tool Shelf",
	"description": "Scale your model to the correct target size",
	"warning": "",
	"wiki_url": "https://github.com/vvoovv/blender-geo/wiki/Target-Scaling",
	"tracker_url": "https://github.com/vvoovv/blender-geo/issues",
	"support": "COMMUNITY",
	"category": "3D View",
}

import bpy, bmesh

def getSelectedEdgeLength(context):
	# getting active edge via bmesh and its select_history
	obj = context.active_object
	bm = bmesh.from_edit_mesh(obj.data)
	edge = bm.select_history.active
	l = -1
	if isinstance(edge, bmesh.types.BMEdge):
		# calculating edge length in the world space using obj.matrix_world
		l = obj.matrix_world * edge.verts[1].co - obj.matrix_world * edge.verts[0].co
		l = round(l.length, 5)
	return l

class TargetScalingPanel(bpy.types.Panel):
	bl_space_type = "VIEW_3D"
	bl_region_type = "TOOLS"
	bl_context = "mesh_edit"
	bl_label = "Target Scaling"

	def draw(self, context):
		layout = self.layout
		
		layout.row().operator("edit.select_target_edge")
		row = layout.row()
		if _.target_length <= 0:
			row.enabled = False
		row.operator("edit.do_target_scaling")

class SelectTargetEdge(bpy.types.Operator):
	bl_idname = "edit.select_target_edge"
	bl_label = "Select a target edge"
	bl_description = "Select a target edge"

	def execute(self, context):
		# getting active edge via bmesh and its select_history
		l = getSelectedEdgeLength(context)
		if l > 0:
			_.target_length = l
			self.report({"INFO"}, "The target edge length is {}".format(l))
		else:
			self.report({"ERROR"}, "Select a single target edge!")
		return {"FINISHED"}

class DoTargetScaling(bpy.types.Operator):
	bl_idname = "edit.do_target_scaling"    
	bl_label = "Perform mesh scaling"
	bl_description = "Perform whole mesh scaling, so the selected edge will be equal to the target one."
	bl_options = {"UNDO"}

	def execute(self, context):
		l = getSelectedEdgeLength(context)
		if l > 0:
			scale = _.target_length/l
			
			# do scaling
			bpy.ops.mesh.select_all(action="SELECT")
			bpy.ops.transform.resize(value=(scale, scale, scale))
			
			# deselect everything
			bpy.ops.mesh.select_all(action="DESELECT")
		else:
			self.report({"ERROR"}, "Select a single edge for target scaling!")
		return {"FINISHED"}

class _:
	"""An auxiliary class to store plugin data"""
	target_length = -1


def register():
	bpy.utils.register_module(__name__)

def unregister():
	bpy.utils.unregister_module(__name__)