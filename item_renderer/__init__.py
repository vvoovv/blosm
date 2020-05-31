import os
import bpy

from util.blender import loadMaterialsFromFile


class ItemRenderer:
    
    def __init__(self, exportMaterials=False):
        self.exportMaterials = exportMaterials
    
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
            materialTemplate = loadMaterialsFromFile(os.path.join(self.r.assetsDir, materialTemplateFilename), True, materialTemplateName)[0]
        return materialTemplate
    
    def getFacadeMaterialTemplate(self, facadeTextureInfo, claddingTextureInfo, materialTemplateFilename):
        if claddingTextureInfo:
            materialTemplateName = "facade_cladding_color" if self.r.useCladdingColor else "facade_cladding"
        else:
            materialTemplateName = "export"
        return self.getMaterialTemplate(materialTemplateFilename, materialTemplateName)
    
    def renderCladding(self, building, item, face, uvs):
        # <item> could be the current item or its parent item.
        # The latter is the case if there is no style block for the bottom
        
        materialId = ''
        claddingTextureInfo = self.getCladdingTextureInfo(item, building)
        if claddingTextureInfo:
            self.setCladdingUvs(item, face, claddingTextureInfo, uvs)
            materialId = self.getCladdingMaterialId(item, claddingTextureInfo)
            self.createCladdingMaterial(materialId, claddingTextureInfo)
            if not self.exportMaterials:
                self.setVertexColor(item, face)
        self.setMaterial(face, materialId)
        # Return <claddingTextureInfo>, since it may be used by
        # the <renderCladding(..)> of a child class
        return claddingTextureInfo
    
    def setMaterial(self, face, materialId):
        self.r.setMaterial(face, materialId)

    def setCladdingUvs(self, item, face, claddingTextureInfo, uvs):
        textureWidthM = claddingTextureInfo["textureWidthM"]
        textureHeightM = claddingTextureInfo["textureHeightM"]
        self.r.setUvs(
            face,
            # a generator!
            ((uv[0]/textureWidthM, uv[1]/textureHeightM) for uv in uvs),
            self.r.layer.uvLayerNameCladding
        )
    
    def _getCladdingTextureInfo(self, item, building):
        claddingMaterial = item.getCladdingMaterial()
        if claddingMaterial:
            if claddingMaterial in building._cache:
                claddingTextureInfo = building._cache[claddingMaterial]
            else:
                claddingTextureInfo = self.r.claddingTextureStore.getTextureInfo(claddingMaterial)
                building._cache[claddingMaterial] = claddingTextureInfo
        else:
            claddingTextureInfo = None
        return claddingTextureInfo