from building.manager import BuildingManager
from building.layer import BuildingLayer


class RealisticBuildingLayer(BuildingLayer):
        
    def __init__(self, layerId, app):
        super().__init__(layerId, app)
        
        # the name for the base UV map used for facade textures
        self.uvLayerNameFacade = "facade"
        # the name for the auxiliary UV map used for claddding textures
        self.uvLayerNameCladding = "cladding"
        # the name for the vertex color layer
        self.vertexColorLayerNameCladding = "cladding_color"
    
    def prepare(self, instance):
        mesh = instance.obj.data
        uv_layers = mesh.uv_layers
        uv_layers.new(name=self.uvLayerNameFacade)
        uv_layers.new(name=self.uvLayerNameCladding)
        
        mesh.vertex_colors.new(name=self.vertexColorLayerNameCladding)
        
        super().prepare(instance)


class RealisticBuildingLayerExport(BuildingLayer):
        
    def __init__(self, layerId, app):
        super().__init__(layerId, app)
        
        # The name for the base UV map used for facade textures
        self.uvLayerNameFacade = "facade"
        # The name for the base UV map used for cladding textures.
        # The same UV-map is used for both the facade and cladding textures
        self.uvLayerNameCladding = "facade"
    
    def prepare(self, instance):
        mesh = instance.obj.data
        uv_layers = mesh.uv_layers
        uv_layers.new(name=self.uvLayerNameFacade)
        
        super().prepare(instance)


class RealisticBuildingManager(BuildingManager):
    
    def __init__(self, osm, buildingParts, layerConstructor=None):
        super().__init__(osm, buildingParts)
        self.layerConstructor = layerConstructor if layerConstructor else RealisticBuildingLayer


class RealisticBuildingManagerExport(RealisticBuildingManager):

    def __init__(self, osm, buildingParts, layerConstructor=None):
        super().__init__(osm, buildingParts, layerConstructor if layerConstructor else RealisticBuildingLayerExport)