import os
import bpy

from util.blender import loadMaterialsFromFile


class ItemRenderer:
    
    def __init__(self):
        self.exportMaterials = False
    
    def init(self, itemRenderers, globalRenderer):
        self.itemRenderers = itemRenderers
        self.r = globalRenderer

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
    
    def getMaterialTemplate(self, materialTemplateFilename, materialTemplateName):
        materialTemplate = bpy.data.materials.get(materialTemplateName)
        if not materialTemplate:
            materialTemplate = loadMaterialsFromFile(os.path.join(self.r.bldgMaterialsDirectory, materialTemplateFilename), True, materialTemplateName)[0]
        return materialTemplate