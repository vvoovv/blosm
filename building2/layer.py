from .. import defs
from ..building.layer import BuildingLayer


class RealisticBuildingLayer(BuildingLayer):
        
    def __init__(self, layerId, app):
        super().__init__(layerId, app)
        
        # the name for the base UV map used for facade textures
        self.uvLayerNameFacade = "facade"

    def prepare(self):
        # The generator owns the BMesh used for Python fallback volumes.
        # Multiple building layers can share this generator.
        uv_layers = self.genVolumes.bm.loops.layers.uv
        if self.uvLayerNameFacade not in uv_layers:
            uv_layers.new(self.uvLayerNameFacade)
        super().prepare()

    def finalize(self, globalRenderer=None):
        # The app also finalizes layers without a renderer. Defer footprint
        # finalization until the building renderer can attach the GN tree.
        if globalRenderer is None:
            return
        super().finalize()
        if self.app.preferableResult == defs.Result.FootprintWithGn:
            globalRenderer.footprintRenderer.finalize(self)


class RealisticBuildingLayerBase(RealisticBuildingLayer):

    def __init__(self, layerId, app):
        super().__init__(layerId, app)
        # the name for the auxiliary UV map used for claddding textures
        self.uvLayerNameCladding = "cladding"
        # the name for the vertex color layer
        self.vertexColorLayerNameCladding = "cladding_color"
    
    def prepare(self):
        bm = self.genVolumes.bm
        uv_layers = bm.loops.layers.uv
        if self.uvLayerNameCladding not in uv_layers:
            uv_layers.new(self.uvLayerNameCladding)
        color_layers = bm.loops.layers.color
        if self.vertexColorLayerNameCladding not in color_layers:
            color_layers.new(self.vertexColorLayerNameCladding)
        super().prepare()


class RealisticBuildingLayerExport(RealisticBuildingLayer):
        
    def __init__(self, layerId, app):
        super().__init__(layerId, app)
        
        # The name for the base UV map used for cladding textures.
        # The same UV-map is used for both the facade and cladding textures
        self.uvLayerNameCladding = self.uvLayerNameFacade
