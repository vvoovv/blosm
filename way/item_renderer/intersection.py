from renderer import Renderer
from util.blender import createMeshObject, getBmesh, setBmesh, addGeometryNodesModifier
from ..asset_store import AssetType, AssetPart
from . import ItemRenderer


class Intersection(ItemRenderer):
    
    def prepare(self):
        self.obj = createMeshObject(
            "Intersections",
            collection = Renderer.collection
        )
        
        self.bm = getBmesh(self.obj)
    
    def renderItem(self, intersection):
        self.bm.faces.new(
            self.bm.verts.new((vert[0], vert[1], 0.)) for vert in intersection.area
        )
    
    def finalize(self):
        setBmesh(self.obj, self.bm)
        self.bm = None
        
        # apply the modifier <self.gnPolygons>
        m = addGeometryNodesModifier(self.obj, self.gnPolygons, "Intersections")
        self.setMaterial(m, "Input_2", AssetType.material, None, AssetPart.pavement, "asphalt")
        
        # apply the modifier <self.gnTerrainArea>
        if self.r.terrainObj:
            m = addGeometryNodesModifier(self.obj, self.gnTerrainArea, "Project on terrain")
            m["Input_2"] = self.r.terrainObj
    
    def requestNodeGroups(self, nodeGroupNames):
        nodeGroupNames.add("blosm_polygons_uv_material")
        nodeGroupNames.add("blosm_terrain_area")
        
    def setNodeGroups(self, nodeGroups):
        self.gnPolygons = nodeGroups["blosm_polygons_uv_material"]
        self.gnTerrainArea = nodeGroups["blosm_terrain_area"]