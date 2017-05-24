from app.layer import Layer


class BuildingLayer(Layer):
    
    # the name for the base UV map
    uvName = "UVMap"
    
    # the name for the auxiliary UV map used to keep the size of a BMFace
    uvNameSize = "size"
    
    def prepare(self, instance):
        uv_textures = instance.obj.data.uv_textures
        uv_textures.new(self.uvName)
        uv_textures.new(self.uvNameSize)
        super().prepare(instance)