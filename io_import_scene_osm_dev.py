bl_info = {
	"name": "Import OpenStreetMap (.osm)",
	"author": "Vladimir Elistratov <vladimir.elistratov@gmail.com>",
	"version": (1, 0, 0),
	"blender": (2, 6, 9),
	"location": "File > Import > OpenStreetMap (.osm)",
	"description" : "Import a file in the OpenStreetMap format (.osm)",
	"warning": "",
	"wiki_url": "https://github.com/vvoovv/blender-geo/wiki/Import-OpenStreetMap-(.osm)",
	"tracker_url": "https://github.com/vvoovv/blender-geo/issues",
	"support": "COMMUNITY",
	"category": "Import-Export",
}

import bpy
# ImportHelper is a helper class, defines filename and invoke() function which calls the file selector
from bpy_extras.io_utils import ImportHelper

import sys, os
sys.path.append("D:\\projects\\blender\\blender-geo")
from transverse_mercator import TransverseMercator
from osm_parser import OsmParser
from osm_import_handlers import buildings
import utils

class ImportOsm(bpy.types.Operator, ImportHelper):
	"""Import a file in the OpenStreetMap format (.osm)"""
	bl_idname = "import_scene.osm"  # important since its how bpy.ops.import_scene.osm is constructed
	bl_label = "Import OpenStreetMap"

	# ImportHelper mixin class uses this
	filename_ext = ".osm"

	filter_glob = bpy.props.StringProperty(
		default="*.osm",
		options={"HIDDEN"},
	)

	thickness = bpy.props.FloatProperty(
		name="Thickness",
		description="Set thickness to make OSM objects extruded",
		default=0,
	)

	def execute(self, context):
		# setting active object if there is no active object
		if not context.scene.objects.active:
			context.scene.objects.active = context.scene.objects[0]
		bpy.ops.object.mode_set(mode="OBJECT")
		
		bpy.ops.object.select_all(action="DESELECT")
		
		# try to find a Blender object with "latitude" and "longitude" as custom properties
		geoObject = utils.findGeoObject(context)
		
		# create an empty object to parent all imported OSM objects
		bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0, 0, 0))
		parentObject = context.active_object
		self.parentObject = parentObject
		parentObject.name = os.path.basename(self.filepath)
		#parentObject.hide = True
		#parentObject.hide_select = True
		parentObject.hide_render = True
		
		self.read_osm_file(geoObject, parentObject)
		
		# perform parenting
		context.scene.objects.active = parentObject
		bpy.ops.object.parent_set()
		bpy.ops.object.select_all(action="DESELECT")
		return {"FINISHED"}

	def read_osm_file(self, geoObject, parentObject):
		osm = OsmParser(self.filepath)
		if geoObject:
			lat = geoObject["latitude"]
			lon = geoObject["longitude"]
		else:
			lat = (osm.minLat + osm.maxLat)/2
			lon = (osm.minLon + osm.maxLon)/2
			parentObject["latitude"] = lat
			parentObject["longitude"] = lon
		projection = TransverseMercator(lat=lat, lon=lon)
		osm.parse(
			projection=projection,
			thickness=self.thickness,
			# possible values for wayHandlers and nodeHandlers list elements:
			#	1) a string name for the module containing functions (all functions from the modules will be used as handlers)
			#	2) a python variable representing the module containing functions (all functions from the modules will be used as handlers)
			#	3) a python variable representing the function
			wayHandlers = [buildings] #[handlers.buildings] #[handlers] #["handlers"]
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