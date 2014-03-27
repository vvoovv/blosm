import bpy


def read_some_data(context, filepath, use_some_setting):
    print("running read_some_data...")
    f = open(filepath, 'r', encoding='utf-8')
    data = f.read()
    f.close()

    # would normally load the data here
    print(data)

    return {'FINISHED'}


# ImportHelper is a helper class, defines filename and
# invoke() function which calls the file selector.
from bpy_extras.io_utils import ImportHelper
from bpy.props import StringProperty, BoolProperty, EnumProperty
from bpy.types import Operator

bl_info = {
	"name": "Import OpenStreetMap (.osm)",
	"author": "Vladimir Elistratov <vladimir.elistratov@gmail.com>",
	"version": (1, 0, 0),
	"blender": (2, 6, 8),
	"location": "File > Import > OpenStreetMap (.osm)",
	"description" : "Import a file in the OpenStreetMap format (.osm)",
	"warning": "",
	"wiki_url": "",
	"tracker_url": "https://github.com/vvoovv/blender-geo/issues",
	"support": "COMMUNITY",
	"category": "Import-Export",
}


class ImportOsm(Operator, ImportHelper):
    """Import a file in the OpenStreetMap format (.osm)"""
    bl_idname = "import_scene.osm"  # important since its how bpy.ops.import_scene.osm is constructed
    bl_label = "Import OpenStreetMap"

    # ImportHelper mixin class uses this
    filename_ext = ".osm"

    filter_glob = StringProperty(
            default="*.osm",
            options={'HIDDEN'},
            )

    # List of operator properties, the attributes will be assigned
    # to the class instance from the operator settings before calling.
    use_setting = BoolProperty(
            name="Example Boolean",
            description="Example Tooltip",
            default=True,
            )

    type = EnumProperty(
            name="Example Enum",
            description="Choose between two items",
            items=(('OPT_A', "First Option", "Description one"),
                   ('OPT_B', "Second Option", "Description two")),
            default='OPT_A',
            )

    def execute(self, context):
        return read_some_data(context, self.filepath, self.use_setting)


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
    bpy.ops.import_scene.osm('INVOKE_DEFAULT')
