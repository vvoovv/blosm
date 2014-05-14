bl_info = {
	"name": "Import OpenStreetMap (.osm)",
	"author": "Vladimir Elistratov <vladimir.elistratov@gmail.com>",
	"version": (1, 0, 0),
	"blender": (2, 6, 9),
	"location": "File > Import > OpenStreetMap (.osm)",
	"description": "Import a file in the OpenStreetMap format (.osm)",
	"warning": "",
	"wiki_url": "https://github.com/vvoovv/blender-geo/wiki/Import-OpenStreetMap-(.osm)",
	"tracker_url": "https://github.com/vvoovv/blender-geo/issues",
	"support": "COMMUNITY",
	"category": "Import-Export",
}

import bpy, bmesh
# ImportHelper is a helper class, defines filename and invoke() function which calls the file selector
from bpy_extras.io_utils import ImportHelper

import sys, os
sys.path.append("D:\\projects\\blender\\blender-geo")
from transverse_mercator import TransverseMercator
from osm_parser import OsmParser
from osm_import_handlers import *
import osm_import_handlers
import utils

class ImportOsm(bpy.types.Operator, ImportHelper):
	"""Import a file in the OpenStreetMap format (.osm)"""
	bl_idname = "import_scene.osm"  # important since its how bpy.ops.import_scene.osm is constructed
	bl_label = "Import OpenStreetMap"
	bl_options = {"UNDO"}

	# ImportHelper mixin class uses this
	filename_ext = ".osm"

	filter_glob = bpy.props.StringProperty(
		default="*.osm",
		options={"HIDDEN"},
	)

	ignoreGeoreferencing = bpy.props.BoolProperty(
		name="Ignore existing georeferencing",
		description="Ignore existing georeferencing and make a new one",
		default=False,
	)
	
	singleMesh = bpy.props.BoolProperty(
		name="Import as a single mesh",
		description="Import OSM objects as a single mesh instead of separate Blender objects",
		default=False,
	)

	importBuildings = bpy.props.BoolProperty(
		name="Import buildings",
		description="Import building outlines",
		default=True,
	)

	importHighways = bpy.props.BoolProperty(
		name="Import roads and paths",
		description="Import roads and paths",
		default=False,
	)

	thickness = bpy.props.FloatProperty(
		name="Thickness",
		description="Set thickness to make OSM building outlines extruded",
		default=0,
	)

	def execute(self, context):
		# setting active object if there is no active object
		if context.mode != "OBJECT":
			# if there is no object in the scene, only "OBJECT" mode is provided
			if not context.scene.objects.active:
				context.scene.objects.active = context.scene.objects[0]
			bpy.ops.object.mode_set(mode="OBJECT")
		
		bpy.ops.object.select_all(action="DESELECT")
		
		name = os.path.basename(self.filepath)
		
		if self.singleMesh:
			self.bm = bmesh.new()
		else:
			self.bm = None
			# create an empty object to parent all imported OSM objects
			bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0, 0, 0))
			parentObject = context.active_object
			self.parentObject = parentObject
			parentObject.name = name
		
		self.read_osm_file(context)
		
		if self.singleMesh:
			bm = self.bm
			# extrude
			if self.thickness>0:
				utils.extrudeMesh(bm, self.thickness)
			
			bm.normal_update()
			
			mesh = bpy.data.meshes.new(name)
			bm.to_mesh(mesh)
			
			obj = bpy.data.objects.new(name, mesh)
			bpy.context.scene.objects.link(obj)
			
			# remove double vertices
			context.scene.objects.active = obj
			bpy.ops.object.mode_set(mode="EDIT")
			bpy.ops.mesh.select_all(action="SELECT")
			bpy.ops.mesh.remove_doubles()
			bpy.ops.mesh.select_all(action="DESELECT")
			bpy.ops.object.mode_set(mode="OBJECT")
			
			bpy.context.scene.update()
		else:
			# perform parenting
			context.scene.objects.active = parentObject
			bpy.ops.object.parent_set()
		
		bpy.ops.object.select_all(action="DESELECT")
		return {"FINISHED"}

	def read_osm_file(self, context):
		scene = context.scene
		
		wayHandlers = []
		if self.importBuildings: wayHandlers.append(buildings)
		if self.importHighways: wayHandlers.append(highways)
		
		osm = OsmParser(self.filepath,
			# possible values for wayHandlers and nodeHandlers list elements:
			#	1) a string name for the module containing classes (all classes from the modules will be used as handlers)
			#	2) a python variable representing the module containing classes (all classes from the modules will be used as handlers)
			#	3) a python variable representing the class
			# Examples:
			# wayHandlers = [buildings, highways]
			# wayHandlers = [handlers.buildings]
			# wayHandlers = [handlers]
			# wayHandlers = ["handlers"]
			wayHandlers = wayHandlers
		)
		
		if "latitude" in scene and "longitude" in scene and not self.ignoreGeoreferencing:
			lat = scene["latitude"]
			lon = scene["longitude"]
		else:
			if osm.bounds and self.importHighways:
				# If the .osm file contains the bounds tag,
				# use its values as the extent of the imported area.
				# Highways may go far beyond the values of the bounds tag.
				# A user might get confused if higways are used in the calculation of the extent of the imported area.
				bounds = osm.bounds
				lat = (bounds["minLat"] + bounds["maxLat"])/2
				lon = (bounds["minLon"] + bounds["maxLon"])/2
			else:
				lat = (osm.minLat + osm.maxLat)/2
				lon = (osm.minLon + osm.maxLon)/2
			scene["latitude"] = lat
			scene["longitude"] = lon
		
		osm.parse(
			projection = TransverseMercator(lat=lat, lon=lon),
			thickness = self.thickness,
			bm = self.bm # if present, indicates the we need to create as single mesh
		)


# Only needed if you want to add into a dynamic menu
def menu_func_import(self, context):
	self.layout.operator(ImportOsm.bl_idname, text="OpenStreetMap (.osm)")

def register():
	bpy.utils.register_class(ImportOsm)
	bpy.types.INFO_MT_file_import.append(menu_func_import)

def unregister():
	bpy.utils.unregister_class(ImportOsm)
	bpy.types.INFO_MT_file_import.remove(menu_func_import)