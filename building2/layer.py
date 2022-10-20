from operator import attrgetter

from building.layer import BuildingLayer

from renderer import Renderer
from util.blender import getBmesh, setBmesh


class RealisticBuildingLayer(BuildingLayer):
    
    def __init__(self, layerId, app):
        super().__init__(layerId, app)
        
        self.gnMeshAttributes = (
            "asset_index",
            "vector",
            "scale"
        )
        
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
    
    def finalize(self, globalRenderer):
        super().finalize()
        
        if self.app.preferMesh:
            gnMeshAttributes = self.gnMeshAttributes
            objGn = self.objGn
            
            setBmesh(objGn, self.bmGn)
            
            # create attributes for the Geometry Nodes
            objGnMesh = self.objGn.data
            # an asset index in Blender collection <globalRenderer.buildingAssetsCollection>
            assetIndex = objGnMesh.attributes.new(gnMeshAttributes[0], 'INT', 'POINT').data
            # a unit vector along the related building facade
            unitVector = objGnMesh.attributes.new(gnMeshAttributes[1], 'FLOAT_VECTOR', 'POINT').data
            # 3 float number to scale an instance along its local x, y and z axes
            scale = objGnMesh.attributes.new(gnMeshAttributes[2], 'FLOAT_VECTOR', 'POINT').data
            
            # Indices in <assetIndex> refer to Blender objects in Blender collection
            # <globalRenderer.buildingAssetsCollection> sorted by the name of the Blender object.
            # That's way we need the Python dictionary <objNameToIndex> for mapping between
            # the name of the Blender object and its index after the sorting
            objNameToIndex = dict(
                (objName.name,index) for index,objName in enumerate(
                    sorted(globalRenderer.buildingAssetsCollection.objects, key=attrgetter("name"))
                )
            )
            
            for index, (objName, _unitVector, scaleX, scaleZ) in enumerate(self.attributeValuesGn):
                assetIndex[index].value = objNameToIndex[objName]
                unitVector[index].vector = _unitVector
                scale[index].vector = (scaleX, scaleX, scaleZ)
            
            self.attributeValuesGn.clear()
            
            # create a modifier for the Geometry Nodes setup
            m = objGn.modifiers.new("", "NODES")
            m.node_group = globalRenderer.gnBuilding
            # <mAttributes> have the form like: 
            # [
            #     "Input_4", "Input_4_use_attribute", "Input_4_attribute_name",
            #     "Input_3", "Input_3_use_attribute", "Input_3_attribute_name"
            # ]
            mAttributes = list(m.keys())
            for i1,i2 in zip( range(0, len(mAttributes), 3), range(len(mAttributes)//3) ):
                # Set "_use_attribute" to 1 to use geometry attributes instead of
                # using manually entered input values
                m[mAttributes[i1]+"_use_attribute"] = 1
                # set "_attribute_name" to the related mesh attribute of <objGn>
                m[mAttributes[i1]+"_attribute_name"] = gnMeshAttributes[i2]


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