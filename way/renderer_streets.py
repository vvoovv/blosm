import os
import bpy

from renderer import Renderer
#from renderer.curve_renderer import CurveRenderer

from util.blender import createMeshObject, createCollection, getBmesh, setBmesh, loadMaterialsFromFile,\
    addGeometryNodesModifier, useAttributeForGnInput, createPolylineMesh

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
        _gnStreet = "blosm_street"
        _gnSidewalk = "blosm_sidewalk"
        _gnSeparator = "blosm_str_separator"
        _gnLamps = "blosm_street_lamps"
        _gnProjectStreets = "blosm_project_streets"
        _gnTerrainPatches = "blosm_terrain_patches"
        _gnProjectOnTerrain = "blosm_project_on_terrain"
        _gnProjectTerrainPatches = "blosm_project_terrain_patches"
        _gnMeshToCurve = "blosm_mesh_to_curve"
        node_groups = bpy.data.node_groups
        if _gnStreet in node_groups:
            self.gnStreet = node_groups[_gnStreet]
            self.gnSidewalk = node_groups[_gnSidewalk]
            self.gnSeparator = node_groups[_gnSeparator]
            self.gnLamps = node_groups[_gnLamps]
            self.gnProjectStreets = node_groups[_gnProjectStreets]
            self.gnTerrainPatches = node_groups[_gnTerrainPatches]
            self.gnProjectOnTerrain = node_groups[_gnProjectOnTerrain]
            self.gnProjectTerrainPatches = node_groups[_gnProjectTerrainPatches]
            self.gnMeshToCurve = node_groups[_gnMeshToCurve]
        else:
            #
            # TEMPORARY CODE
            #
            # load the Geometry Nodes setup with the name "blosm_gn_street_no_terrain" from
            # the file <parent_directory of_<self.app.baseAssetPath>/prochitecture_carriageway_with_sidewalks.blend>
            with bpy.data.libraries.load(os.path.join(os.path.dirname(self.app.baseAssetPath), "prochitecture_streets.blend")) as (_, data_to):
                data_to.node_groups = [
                    _gnStreet, _gnSidewalk, _gnSeparator, _gnLamps,\
                    _gnProjectStreets, _gnTerrainPatches, _gnProjectOnTerrain, _gnProjectTerrainPatches,\
                    _gnMeshToCurve
                ]
            self.gnStreet, self.gnSidewalk, self.gnSeparator, self.gnLamps,\
                self.gnProjectStreets, self.gnTerrainPatches, self.gnProjectOnTerrain,\
                self.gnProjectTerrainPatches, self.gnMeshToCurve = data_to.node_groups
    
    def render(self, manager, data):
        self.terrainRenderer = TerrainPatchesRenderer(self)
        
        # street sections without a cluster
        self.generateStreetSectionsSimple(manager.waySectionLines)
        # street sections clustered
        self.generateStreetSectionsClusters(manager.wayClusters)
        self.renderIntersections(manager)
        
        self.terrainRenderer.processDeadEnds(manager.waySectionLines, manager.wayClusters)
        self.terrainRenderer.addExtent(self.app)
        self.terrainRenderer.setAttributes(manager.waySectionLines, manager.wayClusters)
    
    def generateStreetSection(self, streetSection, waySection, location):
        obj = createMeshObject(
            waySection.tags.get("name", "Street section"),
            collection = self.streetSectionsCollection
        )
        createPolylineMesh(obj, None, streetSection.centerline)
        # project the polyline on the terrain if it's available
        terrainObj = self.projectOnTerrain(obj, self.gnProjectStreets)
        if not terrainObj:
            addGeometryNodesModifier(obj, self.gnMeshToCurve, "Mesh to Curve")
        
        #obj = CurveRenderer.createBlenderObject(
        #    waySection.tags.get("name", "Street section"),
        #    location,
        #    self.streetSectionsCollection,
        #    None
        #)
        #spline = obj.data.splines.new('POLY')
        #spline.points.add(len(centerline)-1)
        #for index,point in enumerate(centerline):
        #    spline.points[index].co = (point[0], point[1], 0., 1.)
        
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
            self.terrainRenderer.processStreetCenterline(streetSection)
    
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
            self.terrainRenderer.processStreetCenterline(streetSection)
    
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
        
        self.projectOnTerrain(self.intersectionAreasObj, self.gnProjectOnTerrain)
    
    def projectOnTerrain(self, obj, gnModifier):
        terrainObj = self.getTerrainObj()
        if terrainObj:
            m = addGeometryNodesModifier(obj, gnModifier, "Project on terrain")
            m["Input_2"] = terrainObj
        return terrainObj
    
    def getTerrainObj(self):
        terrain = self.app.terrain
        if terrain:
            terrain = terrain.terrain
            if terrain:
                terrain.hide_viewport = True
                terrain.hide_render = True
                return terrain
    
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
    
    def __init__(self, streetRenderer):
        self.obj = obj = createMeshObject("Terrain Patches", collection = Renderer.collection)
        obj.data.attributes.new("offset_l", 'FLOAT', 'POINT')
        obj.data.attributes.new("offset_r", 'FLOAT', 'POINT')
        
        m = addGeometryNodesModifier(obj, streetRenderer.gnTerrainPatches, "Terrain patches")
        useAttributeForGnInput(m, "Input_2", "offset_l")
        useAttributeForGnInput(m, "Input_3", "offset_r")
        
        terrainObj = streetRenderer.getTerrainObj()
        if terrainObj:
            m = addGeometryNodesModifier(obj, streetRenderer.gnProjectTerrainPatches, "Project terrain patches")
            m["Input_2"] = terrainObj
        
        self.bm = getBmesh(obj)
    
    def processStreetCenterline(self, streetSection):
        createPolylineMesh(None, self.bm, streetSection.centerline)
    
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
    
    def processDeadEnds(self, waySectionLines, wayClusters):
        
        for streetSection in waySectionLines.values():
            centerline = streetSection.centerline
            directionVector = None
            if not streetSection.startConnected:
                directionVector = _getDirectionVector(centerline[0], centerline[1])
                
                self.offsetEdgePoint(
                    centerline[0],
                    directionVector,
                    0.5*streetSection.width
                )
            
            if not streetSection.endConnected:
                if len(centerline)>2 or not directionVector:
                    directionVector = _getDirectionVector(centerline[-2], centerline[-1])
                self.offsetEdgePoint(
                    centerline[-1],
                    directionVector,
                    0.5*streetSection.width
                )
        
        for streetSection in wayClusters.values():
            centerline = streetSection.centerline
            waySections = streetSection.waySections
            directionVector = None
            
            if not streetSection.startConnected:
                directionVector = _getDirectionVector(centerline[0], centerline[1])
                self.offsetEdgePoint(
                    centerline[0],
                    directionVector,
                    -waySections[0].offset + 0.5*waySections[0].width,
                    waySections[-1].offset + 0.5*waySections[-1].width
                )
            
            if not streetSection.endConnected:
                if len(centerline)>2 or not directionVector:
                    directionVector = _getDirectionVector(centerline[-2], centerline[-1])
                self.offsetEdgePoint(
                    centerline[-1],
                    directionVector,
                    -waySections[0].offset + 0.5*waySections[0].width,
                    waySections[-1].offset + 0.5*waySections[-1].width
                )
    
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
    
    def offsetEdgePoint(self, point, directionVector, offsetLeft, offsetRight=None):
        if offsetRight is None:
            offsetRight = offsetLeft
        
        # point to left from the centerline
        v1 = point - offsetLeft * directionVector
        # point to right from the centerline
        v2 = point + offsetRight * directionVector
        self.bm.edges.new((
            self.bm.verts.new((v1[0], v1[1], 0.)),
            self.bm.verts.new((v2[0], v2[1], 0.))
        ))
    
    def addExtent(self, app):
        # margin in meter
        margin = 2.
        
        minX = app.minX - margin
        minY = app.minY - margin
        maxX = app.maxX + margin
        maxY = app.maxY + margin
        
        # verts
        v1 = self.bm.verts.new((minX, minY, 0.))
        v2 = self.bm.verts.new((maxX, minY, 0.))
        v3 = self.bm.verts.new((maxX, maxY, 0.))
        v4 = self.bm.verts.new((minX, maxY, 0.))
        
        # edges
        self.bm.edges.new((v1, v2))
        self.bm.edges.new((v2, v3))
        self.bm.edges.new((v3, v4))
        self.bm.edges.new((v4, v1))


def _getDirectionVector(point1, point2):
    directionVector = point2 - point1
    directionVector.normalize()
    # vector perpendicular to <directionVector>
    return Vector((directionVector[1], -directionVector[0]))