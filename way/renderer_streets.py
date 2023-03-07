import os
import bpy
from mathutils.geometry import intersect_line_line_2d

from renderer import Renderer
#from renderer.curve_renderer import CurveRenderer

from util.blender import createMeshObject, createCollection, getBmesh, setBmesh, loadMaterialsFromFile,\
    addGeometryNodesModifier, useAttributeForGnInput, createPolylineMesh

from mathutils import Vector


sidewalkWidth = 5.


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
        
        self.intersectionSidewalksObj = createMeshObject(
            "Sidewalks around Intersections",
            collection = Renderer.collection
        )
        self.intersectionSidewalksBm = getBmesh(self.intersectionSidewalksObj)
        
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
            self.setModifierSidewalk(obj, -streetSection.getLeftBorderDistance(), sidewalkWidth)
            
            # sidewalk on the right
            self.setModifierSidewalk(obj, streetSection.getRightBorderDistance(), sidewalkWidth)
            
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
            self.setModifierSidewalk(
                obj,
                -streetSection.getLeftBorderDistance(),
                sidewalkWidth
            )
            
            # sidewalk on the right
            self.setModifierSidewalk(
                obj,
                streetSection.getRightBorderDistance(),
                sidewalkWidth
            )
            
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
        
        for idx,intersectionArea in enumerate(manager.intersectionAreas): # FIXME: remove enumerate
            polygon = intersectionArea.polygon
            bm.faces.new(
                bm.verts.new(Vector((vert[0], vert[1], 0.))) for vert in polygon
            )
            
            self.processIntersectionSidewalks(intersectionArea, manager)
            
            self.terrainRenderer.processIntersection(intersectionArea)
        
        setBmesh(self.intersectionAreasObj, bm)
        
        self.debugIntersectionArea(manager) # FIXME
        
        setBmesh(self.intersectionSidewalksObj, self.intersectionSidewalksBm)
        
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
    
    def processIntersectionSidewalks(self, intersection, manager):
        """
        Process sidewalks around <intersection>
        """
        connectorsInfo = intersection.getConnectorsInfo()
        
        # iterate through all but last connectors in <connectorsInfo>
        for i in range(len(connectorsInfo)-1):
            self.processIntersectionSidewalk(intersection, connectorsInfo[i], connectorsInfo[i+1], manager)
        
        self.processIntersectionSidewalk(intersection, connectorsInfo[-1], connectorsInfo[0], manager)
    
    def processIntersectionSidewalk(self, intersection, connectorInfoL, connectorInfoR, manager):
        """
        Create a mesh for the part of the sidewalk between the street sections attached to the connectors
        described by <connectorInfoL> and <connectorInfoR>.
        
        Important note. The street section attached to the connector described by <connectorInfoR>
        is to the right from the one described by <connectorInfoL>.
        """
        
        # get an instance for the street section attached to the connector described by <connectorInfo1>
        streetSectionL = manager.wayClusters[connectorInfoL[2]]\
            if connectorInfoL[1] else\
            manager.waySectionLines[connectorInfoL[2]]
        streetSectionR = manager.wayClusters[connectorInfoR[2]]\
            if connectorInfoR[1] else\
            manager.waySectionLines[connectorInfoR[2]]
        
        # index of the right point of the left connector
        indexL = (connectorInfoL[0] + 1) % intersection.numPoints
        # index of the left point of the right connector
        indexR = connectorInfoR[0]
        
        offsetToLeft = connectorInfoL[3]=='S'
        offsetDistance = (
            streetSectionL.getLeftBorderDistance()
            if offsetToLeft else
            streetSectionL.getRightBorderDistance()
        ) + sidewalkWidth
        # Offset points for streetSectionL
        point1L = streetSectionL.offsetPoint(
            0 if offsetToLeft else -1,
            offsetToLeft,
            offsetDistance
        )
        point2L = streetSectionL.offsetPoint(
            1 if offsetToLeft else -2,
            offsetToLeft,
            offsetDistance
        )
        
        offsetToLeft = connectorInfoR[3]=='E'
        offsetDistance = (
            streetSectionR.getLeftBorderDistance()
            if offsetToLeft else
            streetSectionR.getRightBorderDistance()
        ) + sidewalkWidth
        point1R = streetSectionR.offsetPoint(
            -1 if offsetToLeft else 0,
            offsetToLeft,
            offsetDistance
        )
        point2R = streetSectionR.offsetPoint(
            -2 if offsetToLeft else 1,
            offsetToLeft,
            offsetDistance
        )
        
        # Check if the segments <point1L>-<point2L> and <point1R>-<point2R> intersect
        if not intersect_line_line_2d(point1L, point2L, point1R, point2R):
            bm = self.intersectionSidewalksBm
            polygon = intersection.polygon
            verts = [
                bm.verts.new((polygon[indexL][0], polygon[indexL][1], 0.)),
                bm.verts.new((point1L[0], point1L[1], 0.)),
                bm.verts.new((point1R[0], point1R[1], 0.)),
                bm.verts.new((polygon[indexR][0], polygon[indexR][1], 0.))
            ]
            if indexL < indexR:
                if indexR - indexL > 1:
                    verts.extend(
                        bm.verts.new((polygon[i][0], polygon[i][1], 0.)) for i in range(indexR-1, indexL, -1)
                    )
            elif not (indexL==intersection.numPoints-1 and indexR==0):
                indices = [] if indexL == intersection.numPoints-1 else\
                    list( range(indexL+1, intersection.numPoints) )
                if indexR == 1:
                    indices.append(0)
                elif indexR > 1:
                    indices.extend(i for i in range(indexR-1))
                verts.extend(
                    bm.verts.new((polygon[i][0], polygon[i][1], 0.)) for i in reversed(indices)
                )
            bm.faces.new(verts)
    
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
    
    def debugIntersectionArea(self, manager):
        self.intersectionAreasObj.data.attributes.new("idx", 'INT', 'FACE')
        for idx,intersection in enumerate(manager.intersectionAreas):
            self.intersectionAreasObj.data.attributes["idx"].data[idx].value = idx
        

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
            offsetL = -streetSection.getLeftBorderDistance()
            offsetR = streetSection.getLeftBorderDistance()
            for _ in streetSection.centerline:
                self.obj.data.attributes["offset_l"].data[index].value = offsetL
                self.obj.data.attributes["offset_r"].data[index].value = offsetR
                index += 1
        
        for streetSection in wayClusters.values():
            offsetL = -streetSection.getLeftBorderDistance()
            offsetR = streetSection.getRightBorderDistance()
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
                    streetSection.getLeftBorderDistance()
                )
            
            if not streetSection.endConnected:
                if len(centerline)>2 or not directionVector:
                    directionVector = _getDirectionVector(centerline[-2], centerline[-1])
                self.offsetEdgePoint(
                    centerline[-1],
                    directionVector,
                    streetSection.getLeftBorderDistance()
                )
        
        for streetSection in wayClusters.values():
            centerline = streetSection.centerline
            directionVector = None
            
            if not streetSection.startConnected:
                directionVector = _getDirectionVector(centerline[0], centerline[1])
                self.offsetEdgePoint(
                    centerline[0],
                    directionVector,
                    streetSection.getLeftBorderDistance(),
                    streetSection.getRightBorderDistance()
                )
            
            if not streetSection.endConnected:
                if len(centerline)>2 or not directionVector:
                    directionVector = _getDirectionVector(centerline[-2], centerline[-1])
                self.offsetEdgePoint(
                    centerline[-1],
                    directionVector,
                    streetSection.getLeftBorderDistance(),
                    streetSection.getRightBorderDistance()
                )
    
    def processIntersection(self, intersection):
        polygon = intersection.polygon
        
        connectorsInfo = intersection.getConnectorsInfo()
        
        #
        # all but the final segments
        #
        for i in range(len(connectorsInfo)-1):
            polylineStartIndex, polylineEndIndex = connectorsInfo[i][0]+1, connectorsInfo[i+1][0]
            # The condition below is used to exclude the case when a connector is directly
            # followed by another connector
            if polylineStartIndex != polylineEndIndex:
                # Create a polyline out of <polygon> that starts at the polygon's point
                # with the index <polylineStartIndex> and ends at the polygon's point
                # with the index <polylineEndIndex>
                prevVert = self.bm.verts.new((polygon[polylineStartIndex][0], polygon[polylineStartIndex][1], 0.))
                for i in range(polylineStartIndex+1, polylineEndIndex+1):
                    vert = self.bm.verts.new((polygon[i][0], polygon[i][1], 0.))
                    self.bm.edges.new((prevVert, vert))
                    prevVert = vert
        
        #
        # the final segment
        #
        # The condition below is used to exclude the case when a connector is directly
        # followed by another connector
        if connectorsInfo[-1][0] != intersection.numPoints-1 or connectorsInfo[0][0]:
            indices = list( range(connectorsInfo[-1][0]+1, intersection.numPoints) )
            indices.extend(i for i in range(connectorsInfo[0][0]+1))
            prevVert = self.bm.verts.new((polygon[indices[0]][0], polygon[indices[0]][1], 0.))
            for i in range(1, len(indices)):
                vert = self.bm.verts.new((
                    polygon[ indices[i] ][0],
                    polygon[ indices[i] ][1],
                    0.
                ))
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