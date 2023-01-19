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
        obj = self.setupStreetObject()
        bm = getBmesh(obj)
        # street sections without a cluster
        self.generateStreetSections(manager.waySectionLines, bm)
        # street sections clustered
        self.generateStreetSections(manager.wayClusters, bm)
        self.renderIntersections(manager)
        
        #self.setGnModifiers(obj)
        setBmesh(obj, bm)
        
        self.setAttributes(manager.waySectionLines, obj.data.attributes)
    
    def setupStreetObject(self):
        obj = createMeshObject(
            "Street Sections",
            collection = self.streetSectionsCollection
        )
        obj.data.attributes.new("offset1", 'FLOAT', 'POINT')
        obj.data.attributes.new("width1", 'FLOAT', 'POINT')
        obj.data.attributes.new("texture_offset1", 'INT', 'POINT')
        
        return obj
    
    def generateStreetSections(self, streetSections, bm):
        for streetSection in streetSections.values():
            centerline = streetSection.centerline
            # create verts and edges
            prevVert = bm.verts.new((centerline[0][0], centerline[0][1], 0.))
            for i in range(1, len(centerline)):
                vert = bm.verts.new((centerline[i][0], centerline[i][1], 0.))
                bm.edges.new((prevVert, vert))
                prevVert = vert
    
    def setAttributes(self, streetSections, attributes):
        index = 0
        for streetSection in streetSections.values():
            for _ in streetSection.centerline:
                attributes['width1'].data[index].value = streetSection.startWidths[0] + streetSection.startWidths[1]
                index += 1

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