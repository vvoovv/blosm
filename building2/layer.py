from building.layer import BuildingLayer

from renderer import Renderer
from util.blender import getBmesh, setBmesh


class RealisticBuildingLayer(BuildingLayer):
    
    def __init__(self, layerId, app):
        super().__init__(layerId, app)
        
        # the name for the base UV map used for facade textures
        self.uvLayerNameFacade = "facade"
        
        # A Blender object for vertex clouds. 3D-modules representing a basic building appearance
        # will be instanced on those vertices.
        self.objGn = None
        # <self.objGnExtra> will be used for the future development
        # Additional details may be added (like, air vents, flowers, interiors).
        # Those 3D-modules for the details will be instanced on vertices that belong to
        # the Blender object(s) <self.objGnExtra>.
        # <self.objGnExtra> could be a single Blender object or a list of Blender objects
        self.objGnExtra = None
    
    def finalize(self):
        super().finalize()
        
        if self.app.preferMesh:
            objGn = self.objGn
            
            setBmesh(objGn, self.bmGn)
            
            # create attributes for the Geometry Nodes
            objGnMesh = self.objGn.data
            # an asset index in the Blender collection
            assetIndex = objGnMesh.attributes.new("asset_index", 'INT', 'POINT').data
            # a unit vector along the related building facade
            unitVector = objGnMesh.attributes.new("vector", 'FLOAT_VECTOR', 'POINT').data
            # 3 float number to scale an instance along its local x, y and z axes
            scale = objGnMesh.attributes.new("scale", 'FLOAT_VECTOR', 'POINT').data
            
            scale = objGn.data.attributes["scale"].data
            for index, (_assetIndex, _unitVector, scaleX, scaleY) in enumerate(self.attributeValuesGn):
                assetIndex[index].value = _assetIndex
                unitVector[index].vector = _unitVector
                scale[index].vector = (scaleX, scaleY, 1.)
            
            self.attributeValuesGn.clear()


class RealisticBuildingLayerBase(RealisticBuildingLayer):
        
    def __init__(self, layerId, app):
        super().__init__(layerId, app)
        
        # the name for the auxiliary UV map used for claddding textures
        self.uvLayerNameCladding = "cladding"
        # the name for the vertex color layer
        self.vertexColorLayerNameCladding = "cladding_color"
    
    def prepare(self):
        mesh = self.obj.data
        uv_layers = mesh.uv_layers
        uv_layers.new(name=self.uvLayerNameFacade)
        uv_layers.new(name=self.uvLayerNameCladding)
        
        mesh.vertex_colors.new(name=self.vertexColorLayerNameCladding)
        
        super().prepare()
        
        if self.app.preferMesh:
            obj = self.obj
            # copy the values from <self.obj>
            self.objGn = Renderer.createBlenderObject(
                obj.name + "_gn",
                obj.location,
                obj.users_collection[0],
                obj.parent
            )
            
            self.attributeValuesGn = []
            
            self.bmGn = getBmesh(self.objGn)


class RealisticBuildingLayerExport(RealisticBuildingLayer):
        
    def __init__(self, layerId, app):
        super().__init__(layerId, app)
        
        # The name for the base UV map used for cladding textures.
        # The same UV-map is used for both the facade and cladding textures
        self.uvLayerNameCladding = "facade"
    
    def prepare(self, instance):
        mesh = instance.obj.data
        uv_layers = mesh.uv_layers
        uv_layers.new(name=self.uvLayerNameFacade)
        
        super().prepare(instance)