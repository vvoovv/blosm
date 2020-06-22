import os
import bpy

from util.blender import loadMaterialsFromFile
from util.blender_extra.material import createMaterialFromTemplate, setImage


_materialTemplateFilename = "building_material_templates.blend"


class ItemRenderer:
    
    def __init__(self, exportMaterials=False):
        self.exportMaterials = exportMaterials
        self.materialTemplateFilename = _materialTemplateFilename
    
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
    
    def renderClass(self, item, itemClass, face, uvs):
        building = item.building
        if building.assetInfoBldgIndex is None or building.assetInfoBldgIndex == -1:
            assetInfo = self.r.assetStore.getAssetInfoByClass(
                item.building, item.buildingPart, "texture", None, itemClass
            )
            if assetInfo and building.assetInfoBldgIndex is None:
                building.assetInfoBldgIndex = assetInfo["_bldgIndex"]
        else:
            assetInfo = self.r.assetStore.getAssetInfoByBldgIndexAndClass(
                building.assetInfoBldgIndex, item.buildingPart, "texture", itemClass
            )
            if not assetInfo:
                # never try to use <building.assetInfoBldgIndex> again
                building.assetInfoBldgIndex = -1
                # try to get <assetInfo> without <building.assetInfoBldgIndex>
                assetInfo = self.r.assetStore.getAssetInfoByClass(
                    item.building, item.buildingPart, "texture", None, itemClass
                )
        if assetInfo:
            if item.materialId is None:
                self.setClassMaterialId(item, assetInfo)
            if item.materialId:
                # Сonvert image coordinates in pixels to UV-coordinates between 0. and 1.
                
                # width and height of the whole image:
                imageWidth, imageHeight = bpy.data.materials[item.materialId].node_tree.nodes.get("Image Texture").image.size
                if "offsetXPx" in assetInfo:
                    texUl = assetInfo["offsetXPx"]/imageWidth
                    texUr = texUl + assetInfo["textureWidthPx"]/imageWidth
                else:
                    texUl, texUr = 0., 1.
                if "offsetYPx" in assetInfo:
                    texVt = 1. - assetInfo["offsetYPx"]/imageHeight
                    texVb = texVt - assetInfo["textureHeightPx"]/imageHeight
                else:
                    texVb, texVt = 0., 1.
                self.setClassUvs(item, face, uvs, texUl, texVb, texUr, texVt)
            self.r.setMaterial(face, item.materialId)
        else:
            # no <assetInfo>, so we try to render cladding only
            self.renderCladding(
                building,
                item,
                face,
                item.uvs
            )
    
    def setClassMaterialId(self, item, assetInfo):        
        materialId = self.getClassMaterialId(item, assetInfo)
        if self.createClassMaterial(materialId, assetInfo):
            item.materialId = materialId
        else:
            item.materialId = ""
    
    def getClassMaterialId(self, item, assetInfo):
        return assetInfo["name"]
    
    def createClassMaterial(self, materialName, assetInfo):
        materialTemplate = self.getClassMaterialTemplate(
            assetInfo,
            self.materialTemplateFilename
        )
        if not materialName in bpy.data.materials:
            nodes = createMaterialFromTemplate(materialTemplate, materialName)
            # the overlay texture
            image = setImage(
                assetInfo["name"],
                os.path.join(self.r.assetStore.baseDir, assetInfo["path"]),
                nodes,
                "Image Texture"
            )
            if not image:
                return False
        return True

    def getClassMaterialTemplate(self, assetInfo, materialTemplateFilename):
        materialTemplateName = "class"
        return self.getMaterialTemplate(materialTemplateFilename, materialTemplateName)
    
    def setClassUvs(self, item, face, uvs, texUl, texVb, texUr, texVt):
        self.r.setUvs(
            face,
            item.geometry.getClassUvs(texUl, texVb, texUr, texVt, item.uvs),
            self.r.layer.uvLayerNameFacade
        )
    
    def _setRoofClassUvs(self, face, uvs, texUl, texVb, texUr, texVt):
        minU = min(uv[0] for uv in uvs)
        deltaU = ( max(uv[0] for uv in uvs) - minU ) / (texUr-texUl)
        minV = min(uv[1] for uv in uvs)
        deltaV = ( max(uv[1] for uv in uvs) - minV ) / (texVt-texVb)
        self.r.setUvs(
            face,
            ( ( texUl + (uv[0]-minU)/deltaU, texVb + (uv[1]-minV)/deltaV ) for uv in uvs ),
            self.r.layer.uvLayerNameFacade
        )