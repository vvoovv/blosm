bl_info = {
	"name": "Import OpenStreetMap (.osm)",
	"author": "Vladimir Elistratov <vladimir.elistratov@gmail.com>",
	"version": (1, 0, 0),
	"blender": (2, 6, 9),
	"location": "File > Import > OpenStreetMap (.osm)",
	"description" : "Import a file in the OpenStreetMap format (.osm)",
	"warning": "",
	"wiki_url": "",
	"tracker_url": "https://github.com/vvoovv/blender-geo/issues",
	"support": "COMMUNITY",
	"category": "Import-Export",
}

import bpy
# ImportHelper is a helper class, defines filename and invoke() function which calls the file selector
from bpy_extras.io_utils import ImportHelper

import sys
sys.path.append("D:\\projects\\blender\\blender-geo")
from transverse_mercator import TransverseMercator
from osm_parser import OsmParser
from osm_import_handlers import buildings

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
		description="Set some thickness to make OSM objects extruded",
		default=0,
	)

	# empty object used to parent all imported OSM objects
	parentObject = None

	def execute(self, context):
		# setting active object if there is no active object
		if not context.scene.objects.active:
			context.scene.objects.active = context.scene.objects[0]
		bpy.ops.object.mode_set(mode="OBJECT")

		if not self.parentObject:
			bpy.ops.object.empty_add(type="PLAIN_AXES",location=(0, 0, 0))
			parentObject = context.active_object
			self.parentObject = parentObject
			parentObject.name = "OpenStreetMap data"
			parentObject.hide = True
			parentObject.hide_select = True
			parentObject.hide_render = True

		bpy.ops.object.select_all(action="DESELECT")
		self.read_osm_file(context)
		# perform parenting
		parentObject = self.parentObject
		context.scene.objects.active = parentObject
		# temporary unhiding self.parentObject, otherwise parenting doesn't work
		parentObject.hide = False
		bpy.ops.object.parent_set()
		# hiding self.parentObject again
		parentObject.hide = True
		bpy.ops.object.select_all(action="DESELECT")
		return {"FINISHED"}

	def read_osm_file(self, context):
		osm = OsmParser(self.filepath)
		lat = (osm.minLat + osm.maxLat)/2
		lon = (osm.minLon + osm.maxLon)/2
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
		# saving geo reference information for the scene
		bpy.longitude = lon
		bpy.latitude = lat

# Only needed if you want to add into a dynamic menu
def menu_func_import(self, context):
	self.layout.operator(ImportOsm.bl_idname, text="OpenStreetMap (.osm)")


def register():
	bpy.utils.register_class(ImportOsm)
	bpy.types.INFO_MT_file_import.append(menu_func_import)


def unregister():
	bpy.utils.unregister_class(ImportOsm)
	bpy.types.INFO_MT_file_import.remove(menu_func_import)


if __name__ == "__main__":
	register()

	# test call
	bpy.ops.import_scene.osm("INVOKE_DEFAULT")
