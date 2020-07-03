import os
import bpy

from util.blender import loadMaterialsFromFile
from util.blender_extra.material import createMaterialFromTemplate, setImage
from .util import setTextureSize, setTextureSize2

_materialTemplateFilename = "building_material_templates.blend"


def _setAssetInfoCache(building, assetInfo, key):
    if assetInfo:
        building.assetInfoBldgIndex = assetInfo["_bldgIndex"]
        # Save building index from <assetInfo>, so later we can get
        # the buildings index for the given building part and class
        # and get an asset info for sure. We don't save <assetInfo> itself
        # in the cache since there may be several asset infos for the given building and
        # building part and class.
        building._cache[key] = building.assetInfoBldgIndex
    

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
    
    def renderCladding(self, item, face, uvs):
        # <item> could be the current item or its parent item.
        # The latter is the case if there is no style block for the bottom
        
        materialId = ''
        claddingTextureInfo = self.getCladdingTextureInfo(item)
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
    
    def _getCladdingTextureInfo(self, item):
        building = item.building
        claddingMaterial = item.getCladdingMaterial()
        if not claddingMaterial:
            return None
        
        # maybe it should be changed to <self.getStyleBlockAttrDeep("claddingClass")>
        claddingClass = item.getStyleBlockAttr("claddingClass")
        
        if building.assetInfoBldgIndex is None:
            if claddingClass:
                claddingTextureInfo = self.r.assetStore.getCladTexInfoByClass(
                    building, claddingMaterial, "texture", claddingClass
                )
                _setAssetInfoCache(
                    building,
                    claddingTextureInfo,
                    # here the first <c> is for cladding, the second <c> is for class
                    "cc%s" % claddingMaterial
                )
            else:
                claddingTextureInfo = self.r.assetStore.getCladTexInfo(
                    building, claddingMaterial, "texture"
                )
                _setAssetInfoCache(
                    building,
                    claddingTextureInfo,
                    # here <c> is for cladding
                    "c%s" % claddingMaterial
                )
        else:
            if claddingClass:
                key = "cc%s" % claddingMaterial
                # If <key> is available in <building._cache>, that means we'll get <claddingTextureInfo> for sure
                claddingTextureInfo = self.r.assetStore.getCladTexInfoByBldgIndexAndClass(
                    building._cache[key] if key in building._cache else building.assetInfoBldgIndex,
                    claddingMaterial,
                    "texture",
                    claddingClass
                )
                if not claddingTextureInfo:
                    # <key> isn't available in <building._cache>, so <building.assetInfoBldgIndex> was used
                    # in the call above. No we try to get <claddingTextureInfo> without <building.assetInfoBldgIndex>
                    claddingTextureInfo = self.r.assetStore.getCladTexInfoByClass(
                        building, claddingMaterial, "texture", claddingClass
                    )
                    _setAssetInfoCache(building, claddingTextureInfo, key)
            else:
                key = "c%s" % claddingMaterial
                # If <key> is available in <building._cache>, that means we'll get <claddingTextureInfo> for sure
                claddingTextureInfo = self.r.assetStore.getCladTexInfoByBldgIndex(
                    building._cache[key] if key in building._cache else building.assetInfoBldgIndex,
                    claddingMaterial,
                    "texture"
                )
                if not claddingTextureInfo:
                    # <key> isn't available in <building._cache>, so <building.assetInfoBldgIndex> was used
                    # in the call above. No we try to get <claddingTextureInfo> without <building.assetInfoBldgIndex>
                    claddingTextureInfo = self.r.assetStore.getCladTexInfo(
                        building, claddingMaterial, "texture"
                    )
                    _setAssetInfoCache(building, claddingTextureInfo, key)
        return claddingTextureInfo
    
    def renderClass(self, item, itemClass, face, uvs):
        building = item.building
        if building.assetInfoBldgIndex is None:
            assetInfo = self.r.assetStore.getAssetInfoByClass(
                item.building, item.buildingPart, "texture", None, itemClass
            )
            _setAssetInfoCache(
                building,
                assetInfo,
                # here <p> is for part, <c> is for class
                "pc%s%s" % (item.buildingPart, itemClass)
            )
        else:
            key = "pc%s%s" % (item.buildingPart, itemClass)
            # If <key> is available in <building._cache>, that means we'll get <assetInfo> for sure
            assetInfo = self.r.assetStore.getAssetInfoByBldgIndexAndClass(
                building._cache[key] if key in building._cache else building.assetInfoBldgIndex,
                item.buildingPart,
                "texture",
                itemClass
            )
            if not assetInfo:
                # <key> isn't available in <building._cache>, so <building.assetInfoBldgIndex> was used
                # in the call above. No we try to get <assetInfo> without <building.assetInfoBldgIndex>
                assetInfo = self.r.assetStore.getAssetInfoByClass(
                    item.building, item.buildingPart, "texture", None, itemClass
                )
                _setAssetInfoCache(building, assetInfo, key)
        if assetInfo:
            if item.materialId is None:
                self.setClassMaterialId(item, assetInfo)
            if item.materialId:
                # Сonvert image coordinates in pixels to UV-coordinates between 0. and 1.
                
                # width and height of the whole image:
                imageWidth, imageHeight = assetInfo["textureSize"]
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
            setTextureSize(assetInfo, image)
        
        setTextureSize2(assetInfo, materialName, "Image Texture")
        
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