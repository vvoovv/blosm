

class ItemRenderer:
    
    def init(self, itemRenderers, globalRenderer):
        self.itemRenderers = itemRenderers
        self.r = globalRenderer
        
        self.uvLayer = "data.1"
        self.vertexColorLayer = "Col"

    def requireUvLayer(self, name):
        uv = self.r.bm.loops.layers.uv
        # create a data UV layer
        if not name in uv:
            uv.new(name)
    
    def requireVertexColorLayer(self, name):
        vertex_colors = self.r.bm.loops.layers.color
        # create a vertex color layer for data
        if not name in vertex_colors:
            vertex_colors.new(name)
    
    def preRender(self):
        self.requireUvLayer(self.uvLayer)
        self.requireVertexColorLayer(self.vertexColorLayer)