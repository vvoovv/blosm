import os
import bpy

from util.blender import loadMaterialsFromFile


class ItemRenderer:
    
    def init(self, itemRenderers, globalRenderer):
        self.itemRenderers = itemRenderers
        self.r = globalRenderer
        
        self.vertexColorLayer = "cladding_color"

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
        self.requireVertexColorLayer(self.vertexColorLayer)
    
    def getMaterialTemplate(self, materialTemplateFilename, materialTemplateName):
        materialTemplate = bpy.data.materials.get(materialTemplateName)
        if not materialTemplate:
            bldgMaterialsDirectory = os.path.dirname(self.r.app.bldgMaterialsFilepath)
            materialTemplate = loadMaterialsFromFile(os.path.join(bldgMaterialsDirectory, materialTemplateFilename), True, materialTemplateName)[0]
        return materialTemplate