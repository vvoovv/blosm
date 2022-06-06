import os
import bpy

from util.blender import loadMaterialsFromFile
from util.blender_extra.material import createMaterialFromTemplate, setImage

_materialTemplateFilename = "building_material_templates.blend"


def _setAssetInfoCache(building, assetInfo, key):
    if assetInfo:
        renderInfo = building.renderInfo
        renderInfo.assetInfoBldgIndex = assetInfo["_bldgIndex"]
        # Save building index from <assetInfo>, so later we can get
        # the buildings index for the given building part and class
        # and get an asset info for sure. We don't save <assetInfo> itself
        # in the cache since there may be several asset infos for the given building and
        # building part and class.
        renderInfo._cache[key] = renderInfo.assetInfoBldgIndex
    

class ItemRenderer:
    
    def __init__(self, exportMaterials):
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
    
    def getMaterialTemplate(self, materialTemplateName):
        materialTemplate = bpy.data.materials.get(materialTemplateName)
        if not materialTemplate:
            materialTemplate = loadMaterialsFromFile(os.path.join(self.r.assetsDir, self.materialTemplateFilename), True, materialTemplateName)[0]
        return materialTemplate
    
    def getFacadeMaterialTemplate(self, facadeTextureInfo, claddingTextureInfo):
        if claddingTextureInfo:
            materialTemplateName = "facade_cladding_color" if self.r.useCladdingColor else "facade_cladding"
        else:
            materialTemplateName = "export"
        return self.getMaterialTemplate(materialTemplateName)
    
    def renderCladding(self, item, face, uvs):
        # <item> could be the current item or its parent item.
        # The latter is the case if there is no style block for the bottom
        
        materialId = ''
        claddingTextureInfo = self.getCladdingTextureInfo(item)
        if claddingTextureInfo:
            materialId = self.getCladdingMaterialId(item, claddingTextureInfo)
            self.createCladdingMaterial(materialId, claddingTextureInfo)
            self.setCladdingUvs(item, face, claddingTextureInfo, uvs)
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
        textureHeightM = textureWidthM * claddingTextureInfo["textureSize"][1] / claddingTextureInfo["textureSize"][0]
        self.r.setUvs(
            face,
            # a generator!
            ((uv[0]/textureWidthM, uv[1]/textureHeightM) for uv in uvs),
            self.r.layer.uvLayerNameCladding
        )
    
    def _getCladdingTextureInfo(self, item):
        claddingMaterial = item.getCladdingMaterial()
        if not claddingMaterial:
            return None
        
        return self.r.assetStore.getAssetInfoCladdingTexture(
            item.building,
            item.getStyleBlockAttrDeep("collection"),
            claddingMaterial,
            item.getStyleBlockAttrDeep("claddingClass")
        )
    
    def renderClass(self, item, itemClass, face, uvs):
        building = item.building
        if building.assetInfoBldgIndex is None:
            mainTextureInfo = self.r.assetStore.getAssetInfoByClass(
                item.building, item.buildingPart, "texture", None, itemClass
            )
            _setAssetInfoCache(
                building,
                mainTextureInfo,
                # here <p> is for part, <c> is for class
                "pc%s%s" % (item.buildingPart, itemClass)
            )
        else:
            key = "pc%s%s" % (item.buildingPart, itemClass)
            # If <key> is available in <building._cache>, that means we'll get <assetInfo> for sure
            mainTextureInfo = self.r.assetStore.getAssetInfoByBldgIndexAndClass(
                building._cache[key] if key in building._cache else building.assetInfoBldgIndex,
                item.buildingPart,
                "texture",
                itemClass
            )
            if not mainTextureInfo:
                # <key> isn't available in <building._cache>, so <building.assetInfoBldgIndex> was used
                # in the call above. No we try to get <assetInfo> without <building.assetInfoBldgIndex>
                mainTextureInfo = self.r.assetStore.getAssetInfoByClass(
                    item.building, item.buildingPart, "texture", None, itemClass
                )
                _setAssetInfoCache(building, mainTextureInfo, key)
        if mainTextureInfo:
            if item.materialId is None:
                claddingTextureInfo = self.getCladdingTextureInfo(item)\
                    if mainTextureInfo.get("cladding") and self.claddingTexture else\
                    None
                self.setClassMaterialId(item, mainTextureInfo, claddingTextureInfo)
            if item.materialId:
                # Сonvert image coordinates in pixels to UV-coordinates between 0. and 1.
                
                # width and height of the whole image:
                imageWidth, imageHeight = mainTextureInfo["textureSize"]
                if "offsetXPx" in mainTextureInfo:
                    texUl = mainTextureInfo["offsetXPx"]/imageWidth
                    texUr = texUl + mainTextureInfo["textureWidthPx"]/imageWidth
                else:
                    texUl, texUr = 0., 1.
                if "offsetYPx" in mainTextureInfo:
                    texVt = 1. - mainTextureInfo["offsetYPx"]/imageHeight
                    texVb = texVt - mainTextureInfo["textureHeightPx"]/imageHeight
                else:
                    texVb, texVt = 0., 1.
                
                self.setClassUvs(item, face, uvs, texUl, texVb, texUr, texVt)
                # uv for cladding and vertex color
                self.renderExtra(item, face, mainTextureInfo, claddingTextureInfo, uvs)
            self.r.setMaterial(face, item.materialId)
        else:
            # no <assetInfo>, so we try to render cladding only
            self.renderCladding(
                item,
                face,
                item.uvs
            )
    
    def setClassMaterialId(self, item, mainTextureInfo, claddingTextureInfo):        
        materialId = self.getFacadeMaterialId(item, mainTextureInfo, claddingTextureInfo)
        
        if self.createFacadeMaterial(item, materialId, mainTextureInfo, claddingTextureInfo, None):
            item.materialId = materialId
        else:
            item.materialId = ""

    def getFacadeMaterialId(self, item, facadeTextureInfo, claddingTextureInfo):
        return "%s_%s" % (facadeTextureInfo["name"], claddingTextureInfo["name"])\
            if claddingTextureInfo\
            else facadeTextureInfo["name"]
    
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
    
    def getAssetInfo(self, item):
        building, collection, part, cl =\
            item.building, item.getStyleBlockAttrDeep("collection"), item.getBuildingPart(), item.getStyleBlockAttrDeep("cl")
        
        assetInfo = None
        if self.r.app.preferMesh:
            assetInfo = self.r.assetStore.getAssetInfoMesh(
                building,
                collection,
                part,
                cl
            )
        
            if assetInfo:
                item.assetInfo = assetInfo

        if not assetInfo:
            # try to get a texture asset
            assetInfo = self.r.assetStore.getAssetInfoTexture(
                building,
                collection,
                part,
                cl
            )
            if assetInfo:
                assetInfo = self.setAttributesForAssetInfoTexture(assetInfo)
                item.assetInfo = assetInfo
        
        return assetInfo
    
    def getTileWidthM(self, item):
        assetInfo = self.getAssetInfo(item)
        return assetInfo["tileWidthM"] if assetInfo else 0.