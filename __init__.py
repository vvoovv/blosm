bl_info = {
    "name": "Import OpenStreetMap (.osm)",
    "author": "Vladimir Elistratov <prokitektura+support@gmail.com>",
    "version": (2, 0, 1),
    "blender": (2, 7, 8),
    "location": "File > Import > OpenStreetMap (.osm)",
    "description": "Import a file in the OpenStreetMap format (.osm)",
    "warning": "",
    "wiki_url": "https://github.com/vvoovv/blender-osm/wiki/Documentation",
    "tracker_url": "https://github.com/vvoovv/blender-osm/issues",
    "support": "COMMUNITY",
    "category": "Import-Export",
}

import os,sys

def _checkPath():
    path = os.path.dirname(__file__)
    if path not in sys.path:
        sys.path.append(path)
_checkPath()

import bpy, bmesh
# ImportHelper is a helper class, defines filename and invoke() function which calls the file selector
from bpy_extras.io_utils import ImportHelper

from util.transverse_mercator import TransverseMercator
from renderer import Renderer
from parse import Osm
import app
from defs import Keys

from setup import setup


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
    
    layers = ("buildings", "highways", "railways", "water", "forests", "vegetation")
    
    # diffuse colors for some layers
    colors = {
        "buildings": (0.309, 0.013, 0.012),
        "highways": (0.1, 0.1, 0.1),
        "water": (0.009, 0.002, 0.8),
        "forests": (0.02, 0.208, 0.007),
        "vegetation": (0.007, 0.558, 0.005),
        "railways": (0.2, 0.2, 0.2)
    }
    
    mode = bpy.props.EnumProperty(
        name = "Mode: 3D or 2D",
        items = (("3D","3D","3D"), ("2D","2D","2D")),
        description = "Import data in 3D or 2D mode",
        default = "3D"
    )
    
    buildings = bpy.props.BoolProperty(
        name = "Import buildings",
        description = "Import building outlines",
        default = True
    )

    highways = bpy.props.BoolProperty(
        name = "Import roads and paths",
        description = "Import roads and paths",
        default = False
    )
    
    railways = bpy.props.BoolProperty(
        name = "Import railways",
        description = "Import railways",
        default = False
    )
    
    water = bpy.props.BoolProperty(
        name = "Import water objects",
        description = "Import water objects (rivers and lakes)",
        default = False
    )
    
    forests = bpy.props.BoolProperty(
        name = "Import forests",
        description = "Import forests and woods",
        default = False
    )
    
    vegetation = bpy.props.BoolProperty(
        name = "Import other vegetation",
        description = "Import other vegetation (grass, meadow, scrub)",
        default = False
    )
    
    singleObject = bpy.props.BoolProperty(
        name = "Import as a single object",
        description = "Import OSM objects as a single Blender mesh objects instead of separate ones",
        default = True
    )
    
    layered = bpy.props.BoolProperty(
        name = "Arrange into layers",
        description = "Arrange imported OSM objects into layers (buildings, highways, etc)",
        default = True
    )

    ignoreGeoreferencing = bpy.props.BoolProperty(
        name = "Ignore existing georeferencing",
        description = "Ignore existing georeferencing and make a new one",
        default = False
    )

    defaultBuildingHeight = bpy.props.FloatProperty(
        name = "Default building height",
        description = "Default height in meters if the building height isn't set in OSM tags",
        default = 5.
    )
    
    levelHeight = bpy.props.FloatProperty(
        name = "Level height",
        description = "Height of a level in meters to use for OSM tags building:levels and building:min_level",
        default = 3.
    )
    
    app = app.App()

    def execute(self, context):
        # <self.logger> may be set in <setup(..)>
        self.logger = None
        # defines for which layerId's material is set per item instead of per layer
        self.materialPerItem = set()
        
        scene = context.scene
        kwargs = {}
        
        # setting active object if there is no active object
        if context.mode != "OBJECT":
            # if there is no object in the scene, only "OBJECT" mode is provided
            if not context.scene.objects.active:
                context.scene.objects.active = context.scene.objects[0]
            bpy.ops.object.mode_set(mode="OBJECT")
            
        if not self.app.has(Keys.mode3d):
            self.mode = '2D'
        
        # manager (derived from manager.Manager) performing some processing
        self.managers = []
        
        self.prepareLayers()
        if not len(self.layerIndices):
            self.layered = False
        
        osm = Osm(self)
        setup(self, osm)
        
        if "lat" in scene and "lon" in scene and not self.ignoreGeoreferencing:
            kwargs["projection"] = TransverseMercator(lat=scene["lat"], lon=scene["lon"])
        else:
            kwargs["projectionClass"] = TransverseMercator
        
        osm.parse(**kwargs)
        self.process()
        self.render()
        
        # setting 'lon' and 'lat' attributes for <scene> if necessary
        if not "projection" in kwargs:
            # <kwargs["lat"]> and <kwargs["lon"]> have been set in osm.parse(..)
            scene["lat"] = osm.lat
            scene["lon"] = osm.lon
        
        if not self.app.has(Keys.mode3d):
            self.app.show()
        
        return {"FINISHED"}
    
    def prepareLayers(self):
        layerIndices = {}
        layerIds = []
        i = 0
        for layerId in self.layers:
            if getattr(self, layerId):
                layerIndices[layerId] = i
                layerIds.append(layerId)
                i += 1
        self.layerIndices = layerIndices
        self.layerIds = layerIds
    
    def process(self):
        logger = self.logger
        if logger: logger.processStart()
        
        for m in self.managers:
            m.process()
        
        if logger: logger.processEnd()
    
    def render(self):
        logger = self.logger
        if logger: logger.renderStart()
        
        Renderer.begin(self)
        for m in self.managers:
            m.render()
        Renderer.end(self)
        
        if logger: logger.renderEnd()
    
    def draw(self, context):
        layout = self.layout
        
        if self.app.has(Keys.mode3d):
            layout.prop(self, "mode", expand=True)
        box = layout.box()
        box.prop(self, "buildings")
        box.prop(self, "highways")
        box.prop(self, "railways")
        box.prop(self, "water")
        box.prop(self, "forests")
        box.prop(self, "vegetation")
        box = layout.box()
        box.prop(self, "defaultBuildingHeight")
        box.prop(self, "levelHeight")
        box = layout.box()
        box.prop(self, "singleObject")
        box.prop(self, "layered")
        layout.box().prop(self, "ignoreGeoreferencing")


# Only needed if you want to add into a dynamic menu
def menu_func_import(self, context):
    self.layout.operator(ImportOsm.bl_idname, text="OpenStreetMap (.osm)")

def register():
    bpy.utils.register_class(ImportOsm)
    app.register()
    bpy.types.INFO_MT_file_import.append(menu_func_import)

def unregister():
    bpy.utils.unregister_class(ImportOsm)
    app.unregister()
    bpy.types.INFO_MT_file_import.remove(menu_func_import)

# This allows you to run the script directly from blenders text editor
# to test the addon without having to install it.
if __name__ == "__main__":
    register()