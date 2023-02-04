import os
import bpy

from renderer import Renderer
from renderer.curve_renderer import CurveRenderer

from util.blender import createMeshObject, createCollection, getBmesh, setBmesh, loadMaterialsFromFile,\
    addGeometryNodesModifier, useAttributeForGnInput

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
        _gnNameStreet = "blosm_street"
        _gnNameSidewalk = "blosm_sidewalk"
        _gnNameSeparator = "blosm_str_separator"
        _gnNameLamps = "blosm_street_lamps"
        _gnNameTerrainPatches = "blosm_terrain_patches"
        node_groups = bpy.data.node_groups
        if _gnNameStreet in node_groups:
            self.gnStreet = node_groups[_gnNameStreet]
            self.gnSidewalk = node_groups[_gnNameSidewalk]
            self.gnSeparator = node_groups[_gnNameSeparator]
            self.gnLamps = node_groups[_gnNameLamps]
            self.gnTerrainPatches = node_groups[_gnNameTerrainPatches]
        else:
            #
            # TEMPORARY CODE
            #
            # load the Geometry Nodes setup with the name "blosm_gn_street_no_terrain" from
            # the file <parent_directory of_<self.app.baseAssetPath>/prochitecture_carriageway_with_sidewalks.blend>
            with bpy.data.libraries.load(os.path.join(os.path.dirname(self.app.baseAssetPath), "prochitecture_streets.blend")) as (_, data_to):
                data_to.node_groups = [_gnNameStreet, _gnNameSidewalk, _gnNameSeparator, _gnNameLamps, _gnNameTerrainPatches]
            self.gnStreet, self.gnSidewalk, self.gnSeparator, self.gnLamps, self.gnTerrainPatches = data_to.node_groups
    
    def render(self, manager, data):
        self.terrainRenderer = TerrainPatchesRenderer(self.gnTerrainPatches)
        
        # street sections without a cluster
        self.generateStreetSectionsSimple(manager.waySectionLines)
        # street sections clustered
        self.generateStreetSectionsClusters(manager.wayClusters)
        self.renderIntersections(manager)
        
        self.terrainRenderer.setAttributes(manager.waySectionLines, manager.wayClusters)
    
    def generateStreetSection(self, streetSection, waySection, location):
        centerline = streetSection.centerline
        
        obj = CurveRenderer.createBlenderObject(
            waySection.tags.get("name", "Street section"),
            location,
            self.streetSectionsCollection,
            None
        )
        spline = obj.data.splines.new('POLY')
        spline.points.add(len(centerline)-1)
        for index,point in enumerate(centerline):
            spline.points[index].co = (point[0], point[1], 0., 1.)
        
        return obj
    
    def generateStreetSectionsSimple(self, streetSections):
        location = Vector((0., 0., 0.))
        
        for streetSection in streetSections.values():
            obj = self.generateStreetSection(streetSection, streetSection, location)
            
            self.setModifierCarriageway(obj, streetSection)
            
            # sidewalk on the left
            #self.setModifierSidewalk(obj, -0.5*streetSection.width, 5.)
            
            # sidewalk on the right
            #self.setModifierSidewalk(obj, 0.5*streetSection.width, 5.)
            
            # Use the street centerlines to create terrain patches to the left and to the right
            # from the street centerline
            self.terrainRenderer.addStreetCenterline(streetSection.centerline)
    
    def generateStreetSectionsClusters(self, streetSections):
        location = Vector((0., 0., 0.))
        
        for streetSection in streetSections.values():
            obj = self.generateStreetSection(streetSection, streetSection.waySections[0], location)
            
            waySections = streetSection.waySections
            
            # carriageways
            for waySection in waySections:
                self.setModifierCarriageway(obj, waySection)

            # sidewalk on the left
            #self.setModifierSidewalk(
            #    obj,
            #    streetSection.waySections[0].offset - 0.5*streetSection.waySections[0].width,
            #    5.
            #)
            
            # sidewalk on the right
            #self.setModifierSidewalk(
            #    obj,
            #    streetSection.waySections[-1].offset + 0.5*streetSection.waySections[-1].width,
            #    5.
            #)
            
            # separators between the carriageways
            for i in range(1, len(waySections)):
                self.setModifierSeparator(
                    obj,
                    0.5*(waySections[i-1].offset + 0.5*waySections[i-1].width + waySections[i].offset - 0.5*waySections[i].width),
                    waySections[i].offset - 0.5*waySections[i].width - (waySections[i-1].offset + 0.5*waySections[i-1].width)
                )
            
            # Use the street centerlines to create terrain patches to the left and to the right
            # from the street centerline
            self.terrainRenderer.addStreetCenterline(streetSection.centerline)
    
    def setModifierCarriageway(self, obj, waySection):
        m = addGeometryNodesModifier(obj, self.gnStreet, "Carriageway")
        m["Input_2"] = waySection.offset
        m["Input_3"] = waySection.width
        #m["Input_9"] = self.getMaterial("blosm_carriageway")
        #m["Input_24"] = self.getMaterial("blosm_sidewalk")
        #m["Input_28"] = self.getMaterial("blosm_zebra")
    
    def setModifierSeparator(self, obj, offset, width):
        m = addGeometryNodesModifier(obj, self.gnSeparator, "Carriageway separator")
        m["Input_2"] = offset
        m["Input_3"] = width
    
    def setModifierSidewalk(self, obj, offset, width):
        m = addGeometryNodesModifier(obj, self.gnSidewalk, "Sidewalk")
        m["Input_3"] = offset
        m["Input_4"] = width
    
    def renderIntersections(self, manager):
        bm = getBmesh(self.intersectionAreasObj)
        
        for intersectionArea in manager.intersectionAreas:
            polygon = intersectionArea.polygon
            bm.faces.new(
                bm.verts.new(Vector((vert[0], vert[1], 0.))) for vert in polygon
            )
            
            self.terrainRenderer.processIntersection(intersectionArea)
        
        setBmesh(self.intersectionAreasObj, bm) 
    
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
        

class TerrainPatchesRenderer:
    # A renderer for the terrain patches made between street graph cycles
    
    def __init__(self, gnTerrainPatches):
        self.obj = obj = createMeshObject("Terrain between streets", collection = Renderer.collection)
        obj.data.attributes.new("offset_l", 'FLOAT', 'POINT')
        obj.data.attributes.new("offset_r", 'FLOAT', 'POINT')
        
        m = addGeometryNodesModifier(obj, gnTerrainPatches, "Terrain patches")
        useAttributeForGnInput(m, "Input_2", "offset_l")
        useAttributeForGnInput(m, "Input_3", "offset_r")
        
        self.bm = getBmesh(obj)
    
    def addStreetCenterline(self, centerline):
        prevVert = self.bm.verts.new((centerline[0][0], centerline[0][1], 0.))
        for i in range(1, len(centerline)):
            vert = self.bm.verts.new((centerline[i][0], centerline[i][1], 0.))
            self.bm.edges.new((prevVert, vert))
            prevVert = vert
    
    def setAttributes(self, waySectionLines, wayClusters):
        setBmesh(self.obj, self.bm)
        
        index = 0
        
        for streetSection in waySectionLines.values():
            offsetL = -0.5*streetSection.width
            offsetR = 0.5*streetSection.width
            for _ in streetSection.centerline:
                self.obj.data.attributes["offset_l"].data[index].value = offsetL
                self.obj.data.attributes["offset_r"].data[index].value = offsetR
                index += 1
        
        for streetSection in wayClusters.values():
            offsetL = streetSection.waySections[0].offset - 0.5*streetSection.waySections[0].width
            offsetR = streetSection.waySections[-1].offset + 0.5*streetSection.waySections[-1].width
            for _ in streetSection.centerline:
                self.obj.data.attributes["offset_l"].data[index].value = offsetL
                self.obj.data.attributes["offset_r"].data[index].value = offsetR
                index += 1
    
    def processIntersection(self, intersectionArea):
        # Form a Python list of starting point of connectors with the adjacent way sections or
        # way clusters
        connectorStarts = []
        if intersectionArea.connectors:
            connectorStarts.extend(c[0] for c in intersectionArea.connectors.values())
        if intersectionArea.clusterConns:
            connectorStarts.extend(c[0] for c in intersectionArea.clusterConns.values())
        
        connectorStarts.sort()
        
        polygon = intersectionArea.polygon
        
        # all but the final segments
        for i in range(len(connectorStarts)-1):
            polylineStartIndex, polylineEndIndex = connectorStarts[i]+1, connectorStarts[i+1]
            # Create a polyline out of <polygon> that starts at the polygon's point
            # with the index <polylineStartIndex> and ends at the polygon's point
            # with the index <polylineEndIndex>
            prevVert = self.bm.verts.new((polygon[polylineStartIndex][0], polygon[polylineStartIndex][1], 0.))
            for i in range(polylineStartIndex+1, polylineEndIndex+1):
                vert = self.bm.verts.new((polygon[i][0], polygon[i][1], 0.))
                self.bm.edges.new((prevVert, vert))
                prevVert = vert
        
        # the final segment
        indices = list( range(connectorStarts[-1]+1, len(polygon)) )
        indices.extend(i for i in range(connectorStarts[0]+1))
        prevVert = self.bm.verts.new((polygon[indices[0]][0], polygon[indices[0]][1], 0.))
        for i in indices:
            vert = self.bm.verts.new((polygon[i][0], polygon[i][1], 0.))
            self.bm.edges.new((prevVert, vert))
            prevVert = vert