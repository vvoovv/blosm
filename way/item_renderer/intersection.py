from renderer import Renderer
from util.blender import createMeshObject, getBmesh, setBmesh, addGeometryNodesModifier
from ..asset_store import AssetType, AssetPart
from . import ItemRenderer


class Intersection(ItemRenderer):
    
    def __init__(self):
        #
        # The names of the inputs for <self.gnIntersection>
        #
        self.inputCenterlines = (
            None, None, None,
            None, # intersection of 3 streets
            # intersection of 4 streets
            ( ("Socket_2", "Socket_17"), ("Socket_4", "Socket_22"), ("Socket_6", "Socket_27"), ("Socket_8", "Socket_32") )
        )
        
        self.inputWidths = (
            None, None, None,
            None, # intersection of 3 streets
            # intersection of 4 streets
            ( ("Socket_3", "Socket_35"), ("Socket_5", "Socket_23"), ("Socket_7", "Socket_28"), ("Socket_9", "Socket_33") )
        )
        
        self.inputLocations = (
            None, None, None,
            None, # intersection of 3 streets
            # intersection of 4 streets
            ( ("Socket_10", "Socket_19"), ("Socket_11", "Socket_24"), ("Socket_12", "Socket_29"), ("Socket_13", "Socket_34") )
        )
    
    def prepare(self):
        return
    
    def renderItem(self, intersection):
        order = intersection.order
        if order != 4:
            return
        intersection.obj = createMeshObject(
            "Intersections",
            collection = self.r.collectionIntersections
        )
        
        #
        # Generate an intersection polygon
        #
        m = addGeometryNodesModifier(intersection.obj, self.gnIntersection[intersection.order], "Intersection: Generate Polygon")
        i = 0
        connector = intersection.startConnector
        while True:
            street = connector.item
            
            m[ self.inputCenterlines[order][i][0] ] = street.obj
            m[ self.inputWidths[order][i][0] ] = street.head.width if connector.leaving else street.tail.width
            m[ self.inputLocations[order][i][0] ] = connector.leaving
            
            connector = connector.succ
            if connector is intersection.startConnector:
                break
            i += 1
        
        #
        # Set UV-coordinates and material for the intersection polygon
        #
        m = addGeometryNodesModifier(intersection.obj, self.gnPolygon, "Intersection: UV and Material")
        self.setMaterial(m, "Input_2", AssetType.material, None, AssetPart.pavement, "asphalt")
        
        #
        # Project the intersection polygon on the terrain
        #
        if self.r.terrainObj:
            m = addGeometryNodesModifier(intersection.obj, self.gnTerrainArea, "Project on terrain")
            m["Input_2"] = self.r.terrainObj
    
    def finalize(self):
        return
    
    def requestNodeGroups(self, nodeGroupNames):
        nodeGroupNames.add("Blosm Intersection Order 4")
        nodeGroupNames.add("Blosm Polygon UV Material")
        nodeGroupNames.add("blosm_terrain_area")
        
    def setNodeGroups(self, nodeGroups):
        self.gnIntersection = (
            None, None, None,
            None, # intersection of 3 streets
            nodeGroups["Blosm Intersection Order 4"] # intersection of 4 streets
        )
        self.gnPolygon = nodeGroups["Blosm Polygon UV Material"]
        self.gnTerrainArea = nodeGroups["blosm_terrain_area"]