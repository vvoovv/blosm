import os
import bpy
from mathutils.geometry import intersect_line_line_2d

from renderer import Renderer
#from renderer.curve_renderer import CurveRenderer
from .asset_store import AssetStore, AssetType, AssetPart

from util.blender import createMeshObject, createCollection, getBmesh, setBmesh, loadMaterialsFromFile,\
    addGeometryNodesModifier, useAttributeForGnInput, createPolylineMesh

from item_renderer.util import getFilepath

from mathutils import Vector


sidewalkWidth = 5.
pedestrianCrossingWidth = 3.6
stopLineWidth = 1.5
# <p>edestrian <c>rossing and <s>top <l>ine
pcslWidth = pedestrianCrossingWidth + stopLineWidth


class StreetRenderer:
    
    def __init__(self, app):
        self.app = app
        
        self.assetsDir = app.assetsDir
        self.assetPackageDir = app.assetPackageDir
        
        self.streetSectionsCollection = None
        self.streetSectionObjNames = []
        self.assetStore = AssetStore(app.assetInfoFilepathStreet)
    
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
        _gnRoadway = "blosm_roadway"
        _gnSidewalk = "blosm_sidewalk"
        _gnPedestrianCrossing = "blosm_pedestrian_crossing"
        _gnSeparator = "blosm_roadway_separator"
        _gnLamps = "blosm_street_lamps"
        _gnProjectStreets = "blosm_project_streets"
        _gnTerrainPatches = "blosm_terrain_patches"
        _gnProjectOnTerrain = "blosm_project_on_terrain"
        _gnProjectTerrainPatches = "blosm_project_terrain_patches"
        _gnMeshToCurve = "blosm_mesh_to_curve"
        _gnPolygons = "blosm_polygons_uv_material"
        
        node_groups = bpy.data.node_groups
        if _gnRoadway in node_groups:
            self.gnRoadway = node_groups[_gnRoadway]
            self.gnSidewalk = node_groups[_gnSidewalk]
            self.gnPedestrianCrossing = node_groups[_gnPedestrianCrossing]
            self.gnSeparator = node_groups[_gnSeparator]
            self.gnLamps = node_groups[_gnLamps]
            self.gnProjectStreets = node_groups[_gnProjectStreets]
            self.gnTerrainPatches = node_groups[_gnTerrainPatches]
            self.gnProjectOnTerrain = node_groups[_gnProjectOnTerrain]
            self.gnProjectTerrainPatches = node_groups[_gnProjectTerrainPatches]
            self.gnMeshToCurve = node_groups[_gnMeshToCurve]
            self.gnPolygons = node_groups[_gnPolygons]
        else:
            #
            # TEMPORARY CODE
            #
            # load the Geometry Nodes setup
            with bpy.data.libraries.load(os.path.join(os.path.dirname(self.app.baseAssetPath), "prochitecture_streets.blend")) as (_, data_to):
                data_to.node_groups = [
                    _gnRoadway, _gnSidewalk, _gnPedestrianCrossing, _gnSeparator, _gnLamps,\
                    _gnProjectStreets, _gnTerrainPatches, _gnProjectOnTerrain, _gnProjectTerrainPatches,\
                    _gnMeshToCurve, _gnPolygons
                ]
            self.gnRoadway, self.gnSidewalk, self.gnPedestrianCrossing, self.gnSeparator, self.gnLamps,\
                self.gnProjectStreets, self.gnTerrainPatches, self.gnProjectOnTerrain,\
                self.gnProjectTerrainPatches, self.gnMeshToCurve, self.gnPolygons = data_to.node_groups
    
    def render(self, manager, data):
        self.terrainRenderer = TerrainPatchesRenderer(self)
        
        # street sections without a cluster
        self.generateStreetSectionsSimple(manager.waySectionLines)
        # street sections clustered
        self.generateStreetSectionsClusters(manager.wayClusters)
        # intersections
        self.renderIntersections(manager)
        
        self.terrainRenderer.processDeadEnds(manager)
        self.terrainRenderer.addExtent(self.app)
        self.terrainRenderer.setAttributes(manager, self)
    
    def generateStreetSection(self, streetSection, waySection, location):
        obj = createMeshObject(
            waySection.tags.get("name", "Street section"),
            collection = self.streetSectionsCollection
        )
        # create an attribute "offset_weight" for <obj>
        obj.data.attributes.new("offset_weight", 'FLOAT', 'POINT')
        createPolylineMesh(obj, None, streetSection.centerline)
        # Set offset weights. An offset weight is equal to
        # 1/sin(angle/2), where <angle> is the angle between <vec1> and <vec2> (see below the code)
        attributes = obj.data.attributes["offset_weight"].data
        attributes[0].value = attributes[-1].value = 1.
        
        centerline = streetSection.centerline
        numPoints = len(centerline)
        if numPoints > 2:
            vec1 = centerline[0] - centerline[1]
            vec1.normalize()
            for i in range(1, numPoints-1):
                vec2 = centerline[i+1] - centerline[i]
                vec2.normalize()
                vec = vec1 + vec2
                vec.normalize()
                attributes[i].value = abs(1/vec.cross(vec2))
                vec1 = -vec2
        
        # project the polyline on the terrain if it's available
        terrainObj = self.projectOnTerrain(obj, self.gnProjectStreets)
        if not terrainObj:
            addGeometryNodesModifier(obj, self.gnMeshToCurve, "Mesh to Curve")
        
        # remember the index of <streetSection>'s entry in <self.streetSectionObjNames>
        streetSection.index = len(self.streetSectionObjNames)
        # Add the name of <obj> to the list <self.streetSectionObjNames>
        self.streetSectionObjNames.append(obj.name)
        
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
            
            self.setModifierRoadway(
                obj, 
                streetSection,
                pcslWidth,
                pcslWidth
            )
            
            # sidewalk on the left
            streetSection.sidewalkL = self.setModifierSidewalk(
                obj, -streetSection.getLeftBorderDistance(), sidewalkWidth
            )
            
            # sidewalk on the right
            streetSection.sidewalkR = self.setModifierSidewalk(
                obj, streetSection.getRightBorderDistance(), sidewalkWidth
            )
            
            # Use the street centerlines to create terrain patches to the left and to the right
            # from the street centerline
            self.terrainRenderer.processStreetCenterline(streetSection)
    
    def generateStreetSectionsClusters(self, streetSections):
        location = Vector((0., 0., 0.))
        
        for streetSection in streetSections.values():
            obj = self.generateStreetSection(streetSection, streetSection.waySections[0], location)
            
            waySections = streetSection.waySections
            
            # roadways
            for waySection in waySections:
                self.setModifierRoadway(
                    obj,
                    waySection,
                    pcslWidth,
                    pcslWidth
                )

            # sidewalk on the left
            streetSection.sidewalkL = self.setModifierSidewalk(
                obj,
                -streetSection.getLeftBorderDistance(),
                sidewalkWidth
            )
            
            # sidewalk on the right
            streetSection.sidewalkR = self.setModifierSidewalk(
                obj,
                streetSection.getRightBorderDistance(),
                sidewalkWidth
            )
            
            # separators between the roadways
            for i in range(1, len(waySections)):
                self.setModifierSeparator(
                    obj,
                    0.5*(waySections[i-1].offset + 0.5*waySections[i-1].width + waySections[i].offset - 0.5*waySections[i].width),
                    waySections[i].offset - 0.5*waySections[i].width - (waySections[i-1].offset + 0.5*waySections[i-1].width)
                )
            
            # Use the street centerlines to create terrain patches to the left and to the right
            # from the street centerline
            self.terrainRenderer.processStreetCenterline(streetSection)
    
    def setModifierRoadway(self, obj, waySection, trimLengthStart, trimLengthEnd):
        m = addGeometryNodesModifier(obj, self.gnRoadway, "Roadway")
        m["Input_2"] = waySection.offset
        m["Input_3"] = waySection.width
        useAttributeForGnInput(m, "Input_4", "offset_weight")
        # get asset info for the material
        assetInfo = self.assetStore.getAssetInfo(
            AssetType.material, "demo", AssetPart.roadway, self.getClass(waySection)
        )
        # set material
        m["Input_5"] = self.getMaterial(assetInfo)
        # set trim lengths
        m["Input_6"] = trimLengthStart
        m["Input_7"] = trimLengthEnd
    
    def setModifierSeparator(self, obj, offset, width):
        m = addGeometryNodesModifier(obj, self.gnSeparator, "Roadway separator")
        m["Input_2"] = offset
        m["Input_3"] = width
        useAttributeForGnInput(m, "Input_4", "offset_weight")
        # get asset info for the material
        assetInfo = self.assetStore.getAssetInfo(
            AssetType.material, None, AssetPart.ground, "grass"
        )
        # set material
        m["Input_5"] = self.getMaterial(assetInfo)
    
    def setModifierSidewalk(self, obj, offset, width):
        m = addGeometryNodesModifier(obj, self.gnSidewalk, "Sidewalk")
        m["Input_3"] = offset
        m["Input_4"] = width
        useAttributeForGnInput(m, "Input_5", "offset_weight")
        # get asset info for the material
        assetInfo = self.assetStore.getAssetInfo(
            AssetType.material, "demo", AssetPart.sidewalk, None
        )
        # set material
        m["Input_8"] = self.getMaterial(assetInfo)
        return m
    
    def setModifierPedestrianCrossing(self, obj, width):
        m = addGeometryNodesModifier(obj, self.gnPedestrianCrossing, "Pedestrian Crossing")
        m["Input_2"] = width
    
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
        # apply the modifier <self.gnPolygons>
        m = addGeometryNodesModifier(self.intersectionAreasObj, self.gnPolygons, "Intersections")
        # get asset info for the material
        assetInfo = self.assetStore.getAssetInfo(
            AssetType.material, None, AssetPart.pavement, "asphalt"
        )
        # set material
        m["Input_2"] = self.getMaterial(assetInfo)
        
        
        setBmesh(self.intersectionSidewalksObj, self.intersectionSidewalksBm)
        # apply the modifier <self.gnPolygons>
        m = addGeometryNodesModifier(self.intersectionSidewalksObj, self.gnPolygons, "Intersection Sidewalks")
        # get asset info for the material
        assetInfo = self.assetStore.getAssetInfo(
            AssetType.material, None, AssetPart.pavement, None
        )
        # set material
        m["Input_2"] = self.getMaterial(assetInfo)
        
        
        self.debugIntersectionArea(manager) # FIXME
        
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
        
        bm = self.intersectionSidewalksBm
        polygon = intersection.polygon
        verts = None
        
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
        
        offsetToLeftL = not connectorInfoL[3]
        index1L = connectorInfoL[3]
        index2L = -2 if index1L else 1
        offsetDistance = (
            streetSectionL.getLeftBorderDistance()
            if offsetToLeftL else
            streetSectionL.getRightBorderDistance()
        ) + sidewalkWidth
        # Offset points for streetSectionL
        normal1L = streetSectionL.getNormal(index1L, offsetToLeftL)
        point1L = streetSectionL.offsetPoint(
            index1L,
            offsetToLeftL,
            offsetDistance,
            normal1L
        )
        normal2L = streetSectionL.getNormal(index2L, offsetToLeftL)
        point2L = streetSectionL.offsetPoint(
            index2L,
            offsetToLeftL,
            offsetDistance * self.getOffsetWeight(streetSectionL, index2L),
            normal2L
        )
        vectorL = point2L - point1L
        lengthL = vectorL.length
        
        offsetToLeftR = bool(connectorInfoR[3])
        index1R = connectorInfoR[3]
        index2R = -2 if index1R else 1
        offsetDistance = (
            streetSectionR.getLeftBorderDistance()
            if offsetToLeftR else
            streetSectionR.getRightBorderDistance()
        ) + sidewalkWidth
        normal1R = streetSectionR.getNormal(index1R, offsetToLeftR)
        point1R = streetSectionR.offsetPoint(
            index1R,
            offsetToLeftR,
            offsetDistance,
            normal1R
        )
        normal2R = streetSectionR.getNormal(index2R, offsetToLeftR)
        point2R = streetSectionR.offsetPoint(
            index2R,
            offsetToLeftR,
            offsetDistance * self.getOffsetWeight(streetSectionR, index2R),
            normal2R
        )
        vectorR = point2R - point1R
        lengthR = vectorR.length
        
        # Check if the ray with the origin at <point1L> and the direction towards <point2L> intersects
        # with the ray with the origin at <point1R> and the direction towards <point2R>.
        # Actually instead of calculating if the above rays intersect we simply enlarge the segments
        # <point1L>-<point2L> and <point1R>-<point2R> to the length <rayLength> and check if those
        # enlarged segments intersect.
        rayLength = 100.
        intersectionPoint = intersect_line_line_2d(
            point1L,
            point1L + rayLength/lengthL*vectorL,
            point1R,
            point1R + rayLength/lengthR*vectorR
        )
        if intersectionPoint:
            # The rays do intersect.
            
            # Check if the intersection point on the rays belongs to
            # the segment <point1L>-<point2L>
            if (intersectionPoint - point1L).length_squared <= lengthL*lengthL:
                # Calculate the foot of the perpendicular from <intersectionPoint> to
                # the street segment with the indices <index1L> and <index2L> and
                # offset at the border of the roadway in direction defined by
                # normals <normal1L> and <normal2L>
                point1L -= sidewalkWidth*normal1L
                point2L -= sidewalkWidth*normal2L
                # Unit vector along the segment <point1L>-<point2L>
                unitVector = vectorL/lengthL
                # Distance from <point1L> to the foot of the perpendicular
                trimLength = (intersectionPoint - point1L).dot(unitVector)
                if trimLength*trimLength <= (point2L-point1L).length_squared:
                    self.setTrimLength(
                        streetSectionL,
                        trimLength,
                        offsetToLeftL,
                        offsetToLeftL
                    )
                    # Vector to the foot of the perpendicular
                    foot = point1L + trimLength*unitVector
                    verts = [
                        bm.verts.new((polygon[indexL][0], polygon[indexL][1], 0.)),
                        bm.verts.new((foot[0], foot[1], 0.)),
                        bm.verts.new((intersectionPoint[0], intersectionPoint[1], 0.)),
                    ]
            # Check if the intersection point on the rays belongs to
            # the segment <point1R>-<point2R>
            if verts and (intersectionPoint - point1R).length_squared <= lengthR*lengthR:
                # Calculate the foot of the perpendicular from <intersectionPoint> to
                # the street segment with the indices <index1R> and <index2R> and
                # offset at the border of the roadway in direction defined by
                # normals <normal1R> and <normal2R>
                point1R -= sidewalkWidth*normal1R
                point2R -= sidewalkWidth*normal2R
                # Unit vector along the segment <point1R>-<point2R>
                unitVector = vectorR/lengthR
                # Distance from <point1R> to the foot of the perpendicular
                trimLength = (intersectionPoint - point1R).dot(unitVector)
                if trimLength*trimLength <= (point2R-point1R).length_squared:
                    self.setTrimLength(
                        streetSectionR,
                        trimLength,
                        offsetToLeftR,
                        not offsetToLeftR
                    )
                    # Vector to the foot of the perpendicular
                    foot = point1R + trimLength*unitVector
                    verts.append(bm.verts.new((foot[0], foot[1], 0.)))
                
        else:
            verts = [
                bm.verts.new((polygon[indexL][0], polygon[indexL][1], 0.)),
                bm.verts.new((point1L[0], point1L[1], 0.)),
                bm.verts.new((point1R[0], point1R[1], 0.))
            ]
        
        if verts:
            if indexL != indexR:
                verts.append(bm.verts.new((polygon[indexR][0], polygon[indexR][1], 0.)))
                
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
    
    def setAttributeOffsetWeight(self, inputObj, streetSection, startIndex):
        if self.streetSectionObjNames:
            streetSectionObjData = bpy.data.objects[ self.streetSectionObjNames[streetSection.index] ].data.attributes["offset_weight"].data
            inputObjData = inputObj.data.attributes["offset_weight"].data
            # "ss" stands for "street section"
            for inputObjIndex, ssObjIndex in zip(
                    range(startIndex, startIndex+len(streetSection.centerline)),
                    range(len(streetSection.centerline))
                ):
                # Copy the values of the attribute "offset_weight" from the street section object to <inputObj>
                inputObjData[inputObjIndex].value = streetSectionObjData[ssObjIndex].value
    
    def getOffsetWeight(self, streetSection, index):
        return bpy.data.objects[ self.streetSectionObjNames[streetSection.index] ].data.attributes["offset_weight"].data[index].value
    
    def setTrimLength(self, streetSection, trimLength, left, start):
        if left:
            streetSection.sidewalkL["Input_6" if start else "Input_7"] = trimLength
        else:
            streetSection.sidewalkR["Input_6" if start else "Input_7"] = trimLength
    
    def finalize(self):
        return
    
    def cleanup(self):
        return
    
    def getClass(self, waySection):
        numLanes = waySection.nrOfLanes
        if not isinstance(numLanes, int):
            numLanes = numLanes[0] + numLanes[1]
        return str(numLanes) + "_lanes"
    
    def getMaterial(self, assetInfo):
        materialName = assetInfo["material"]
        material = bpy.data.materials.get(materialName)
        
        if not material:
            material = loadMaterialsFromFile(
                getFilepath(self, assetInfo),
                False,
                materialName
            )
            if material:
                material = material[0]
            
        return material
    
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
        obj.data.attributes.new("offset_weight", 'FLOAT', 'POINT')
        
        m = addGeometryNodesModifier(obj, streetRenderer.gnTerrainPatches, "Terrain patches")
        useAttributeForGnInput(m, "Input_2", "offset_l")
        useAttributeForGnInput(m, "Input_3", "offset_r")
        useAttributeForGnInput(m, "Input_4", "offset_weight")
        
        # apply the modifier <self.gnPolygons>
        m = addGeometryNodesModifier(obj, streetRenderer.gnPolygons, "Material for terrain patches")
        # get asset info for the material
        assetInfo = streetRenderer.assetStore.getAssetInfo(
            AssetType.material, None, AssetPart.ground, "grass"
        )
        # set material
        m["Input_2"] = streetRenderer.getMaterial(assetInfo)
        
        terrainObj = streetRenderer.getTerrainObj()
        if terrainObj:
            m = addGeometryNodesModifier(obj, streetRenderer.gnProjectTerrainPatches, "Project terrain patches")
            m["Input_2"] = terrainObj
        
        self.bm = getBmesh(obj)
    
    def processStreetCenterline(self, streetSection):
        createPolylineMesh(None, self.bm, streetSection.centerline)
    
    def setAttributes(self, manager, renderer):
        setBmesh(self.obj, self.bm)
        
        pointIndex = 0
        
        for streetSection in manager.waySectionLines.values():
            offsetL = -streetSection.getLeftBorderDistance()
            offsetR = streetSection.getLeftBorderDistance()
            renderer.setAttributeOffsetWeight(self.obj, streetSection, pointIndex)
            for _ in streetSection.centerline:
                self.obj.data.attributes["offset_l"].data[pointIndex].value = offsetL
                self.obj.data.attributes["offset_r"].data[pointIndex].value = offsetR
                pointIndex += 1
        
        for streetSection in manager.wayClusters.values():
            offsetL = -streetSection.getLeftBorderDistance()
            offsetR = streetSection.getRightBorderDistance()
            renderer.setAttributeOffsetWeight(self.obj, streetSection, pointIndex)
            for _ in streetSection.centerline:
                self.obj.data.attributes["offset_l"].data[pointIndex].value = offsetL
                self.obj.data.attributes["offset_r"].data[pointIndex].value = offsetR
                pointIndex += 1
    
    def processDeadEnds(self, manager):
        
        for streetSection in manager.waySectionLines.values():
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
        
        for streetSection in manager.wayClusters.values():
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