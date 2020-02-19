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
            materialTemplate = loadMaterialsFromFile(os.path.join(self.r.bldgMaterialsDirectory, materialTemplateFilename), True, materialTemplateName)[0]
        return materialTemplate
    
    def renderCladding(self, building, item, face, uvs):
        # <item> could be the current item or its parent item.
        # The latter is the case if there is no style block for the bottom
        
        materialId = ''
        claddingMaterial = item.getStyleBlockAttrDeep("claddingMaterial")
        claddingTextureInfo = None
        if claddingMaterial:
            claddingTextureInfo = self.getCladdingTextureInfo(claddingMaterial, building)
            if claddingTextureInfo:
                self.setCladdingUvs(item, face, claddingTextureInfo, uvs)
                materialId = self.getCladdingMaterialId(item, claddingTextureInfo)
                self.createCladdingMaterial(materialId, claddingTextureInfo)
                self.setVertexColor(item, face)
        self.r.setMaterial(face, materialId)
        # Return <claddingTextureInfo>, since it may be used by
        # the <renderCladding(..)> of a child class
        return claddingTextureInfo

    def setCladdingUvs(self, item, face, claddingTextureInfo, uvs):
        textureWidthM = claddingTextureInfo["textureWidthM"]
        textureHeightM = claddingTextureInfo["textureHeightM"]
        self.r.setUvs(
            face,
            # a generator!
            ((uv[0]/textureWidthM, uv[1]/textureHeightM) for uv in uvs),
            self.r.layer.uvLayerNameCladding
        )
    
    def getCladdingTextureInfo(self, claddingMaterial, building):
        if claddingMaterial:
            if claddingMaterial in building._cache:
                claddingTextureInfo = building._cache[claddingMaterial]
            else:
                claddingTextureInfo = self.r.claddingTextureStore.getTextureInfo(claddingMaterial)
                building._cache[claddingMaterial] = claddingTextureInfo
        else:
            claddingTextureInfo = None
        return claddingTextureInfo