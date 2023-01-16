import os
import bpy

from renderer import Renderer
from renderer.curve_renderer import CurveRenderer

from util.blender import createMeshObject, createCollection, getBmesh, setBmesh, loadMaterialsFromFile

from mathutils import Vector


class StreetRenderer:
    
    def __init__(self, app):
        self.app = app
        self.streetSectionsCollection = None
    
    def prepare(self):
        self.streetSectionsCollection = createCollection("Street sections", Renderer.collection)
        
        self.intersectionAreasObj = createMeshObject(
            "Intersections",
            collection = Renderer.collection
        )
        
        # check if the Geometry Nodes setup with the name "blosm_gn_street_no_terrain" is already available
        _gnNameStreet = "blosm_gn_street_no_terrain"
        _gnNameLamps = "blosm_street_lamps"
        node_groups = bpy.data.node_groups
        if _gnNameStreet in node_groups:
            self.gnStreet = node_groups[_gnNameStreet]
            self.gnLamps = node_groups[_gnNameLamps]
        else:
            #
            # TEMPORARY CODE
            #
            # load the Geometry Nodes setup with the name "blosm_gn_street_no_terrain" from
            # the file <parent_directory of_<self.app.baseAssetPath>/prochitecture_carriageway_with_sidewalks.blend>
            with bpy.data.libraries.load(os.path.join(os.path.dirname(self.app.baseAssetPath), "prochitecture_carriageway_with_sidewalks.blend")) as (_, data_to):
                data_to.node_groups = [_gnNameStreet, _gnNameLamps]
            self.gnStreet, self.gnLamps = data_to.node_groups
    
    def render(self, manager, data):
        obj = CurveRenderer.createBlenderObject(
            "Street Sections",
            Vector((0., 0., 0.)),
            self.streetSectionsCollection,
            None
        )
        # street sections without a cluster
        self.renderStreetSections(manager.waySectionLines, obj)
        # street sections clustered
        self.renderStreetSections(manager.wayClusters, obj)
        self.renderIntersections(manager)
        
        #self.setGnModifiers(obj)
    
    def renderStreetSections(self, streetSections, obj):
        for streetSection in streetSections.values():
            centerline = streetSection.centerline
            spline = obj.data.splines.new('POLY')
            spline.points.add(len(centerline)-1)
            for index,point in enumerate(centerline):
                spline.points[index].co = (point[0], point[1], 0., 1.)

    def renderIntersections(self, manager):
        bm = getBmesh(self.intersectionAreasObj)
        
        for intersectionArea in manager.intersectionAreas:
            polygon = intersectionArea.polygon
            bm.faces.new(
                bm.verts.new(Vector((vert[0], vert[1], 0.))) for vert in polygon
            )
        
        setBmesh(self.intersectionAreasObj, bm)
    
    def setGnModifiers(self, obj):
        # create a modifier for the Geometry Nodes setup
        m = obj.modifiers.new("Street section", "NODES")
        m.node_group = self.gnStreet
        m["Input_25"] = streetSection.startWidths[0] + streetSection.startWidths[1]
        m["Input_26"] = 2.5
        m["Input_9"] = self.getMaterial("blosm_carriageway")
        m["Input_24"] = self.getMaterial("blosm_sidewalk")
        m["Input_28"] = self.getMaterial("blosm_zebra")
        
        m = obj.modifiers.new("Street Lamps", "NODES")
        m.node_group = self.gnLamps
        m["Input_26"] = 10.    
    
    def finalize(self):
        return
    
    def cleanup(self):
        return
    
    def getMaterial(self, name):
        """
        TEMPORARY CODE
        """
        if name in bpy.data.materials:
            return bpy.data.materials[name]
        else:
            return loadMaterialsFromFile(os.path.join(os.path.dirname(self.app.baseAssetPath), "prochitecture_carriageway_with_sidewalks.blend"), False, name)[0]