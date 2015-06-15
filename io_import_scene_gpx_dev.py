bl_info = {
    "name": "Import GPX (.gpx)",
    "author": "Vladimir Elistratov <vladimir.elistratov@gmail.com>",
    "version": (1, 0, 0),
    "blender": (2, 7, 2),
    "location": "File > Import > GPX (.gpx)",
    "description": "Import a file in the GPX format (.gpx)",
    "warning": "",
    "wiki_url": "",
    "tracker_url": "https://github.com/vvoovv/blender-geo/issues",
    "support": "COMMUNITY",
    "category": "Import-Export",
}

import bpy, bmesh
# ImportHelper is a helper class, defines filename and invoke() function which calls the file selector
from bpy_extras.io_utils import ImportHelper

import xml.etree.cElementTree as etree

import sys, os
sys.path.append("D:\\projects\\blender\\blender-geo")
from transverse_mercator import TransverseMercator

class ImportGpx(bpy.types.Operator, ImportHelper):
    """Import a file in the GPX format (.gpx)"""
    bl_idname = "import_scene.gpx"  # important since its how bpy.ops.import_scene.gpx is constructed
    bl_label = "Import GPX"
    bl_options = {"UNDO"}

    # ImportHelper mixin class uses this
    filename_ext = ".gpx"

    filter_glob = bpy.props.StringProperty(
        default="*.gpx",
        options={"HIDDEN"},
    )

    ignoreGeoreferencing = bpy.props.BoolProperty(
        name="Ignore existing georeferencing",
        description="Ignore existing georeferencing and make a new one",
        default=False,
    )
    
    useElevation = bpy.props.BoolProperty(
        name="Use elevation for z-coordinate",
        description="Use elevation from the track for z-coordinate if checked or make the track flat otherwise",
        default=True,
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
        
        self.bm = bmesh.new()
        
        self.read_gpx_file(context)
        
        mesh = bpy.data.meshes.new(name)
        self.bm.to_mesh(mesh)
        
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.scene.objects.link(obj)
        
        # remove double vertices
        context.scene.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.remove_doubles()
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.object.mode_set(mode="OBJECT")
        
        obj.select = True
        
        bpy.context.scene.update()
        
        return {"FINISHED"}

    def read_gpx_file(self, context):
        scene = context.scene
        
        # a list of track segments (trkseg)
        segments = []

        minLat = 90
        maxLat = -90
        minLon = 180
        maxLon = -180
        
        gpx = etree.parse(self.filepath).getroot()
        
        for e1 in gpx: # e stands for element
            # Each tag may have the form {http://www.topografix.com/GPX/1/1}tag
            # That's whay we skip curly brackets
            if e1.tag[e1.tag.find("}")+1:] == "trk":
                for e2 in e1:
                    if e2.tag[e2.tag.find("}")+1:] == "trkseg":
                        segment = []
                        for e3 in e2:
                            if e3.tag[e3.tag.find("}")+1:] == "trkpt":
                                lat = float(e3.attrib["lat"])
                                lon = float(e3.attrib["lon"])
                                # calculate track extent
                                if lat<minLat: minLat = lat
                                elif lat>maxLat: maxLat = lat
                                if lon<minLon: minLon = lon
                                elif lon>maxLon: maxLon = lon
                                # check if <trkpt> has <ele>
                                ele = None
                                for e4 in e3:
                                    if e4.tag[e4.tag.find("}")+1:] == "ele":
                                        ele = e4
                                        break
                                point = (lat, lon, float(ele.text)) if self.useElevation and ele is not None else (lat, lon)
                                segment.append(point)
                        segments.append(segment)
        
        if "latitude" in scene and "longitude" in scene and not self.ignoreGeoreferencing:
            lat = scene["latitude"]
            lon = scene["longitude"]
        else:
            lat = (minLat + maxLat)/2
            lon = (minLon + maxLon)/2
            scene["latitude"] = lat
            scene["longitude"] = lon
        projection = TransverseMercator(lat=lat, lon=lon)
        
        # create vertices and edges for the track segments
        for segment in segments:
            prevVertex = None
            for point in segment:
                v = projection.fromGeographic(point[0], point[1])
                v = self.bm.verts.new((v[0], v[1], point[2] if self.useElevation and len(point)==3 else 0))
                if prevVertex:
                    self.bm.edges.new([prevVertex, v])
                prevVertex = v


# Only needed if you want to add into a dynamic menu
def menu_func_import(self, context):
    self.layout.operator(ImportGpx.bl_idname, text="GPX (.gpx)")

def register():
    bpy.utils.register_class(ImportGpx)
    bpy.types.INFO_MT_file_import.append(menu_func_import)

def unregister():
    bpy.utils.unregister_class(ImportGpx)
    bpy.types.INFO_MT_file_import.remove(menu_func_import)