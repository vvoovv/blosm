import os
import bpy

from util.blender import loadMaterialsFromFile, linkObjectFromFile
from .util import getFilepath
from util.blender_extra.material import createMaterialFromTemplate, setImage

_materialTemplateFilename = "building_material_templates.blend"


class ItemRenderer:
    
    def __init__(self, exportMaterials):
        self.exportMaterials = exportMaterials
        self.materialTemplateFilename = _materialTemplateFilename
    
    def init(self, itemRenderers, globalRenderer):
        self.itemRenderers = itemRenderers
        self.r = globalRenderer
        self.app = globalRenderer.app

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
        elif facadeTextureInfo.get("specularMapName"):
            materialTemplateName = "facade_specular_color" if self.r.useCladdingColor else "facade_specular"
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
        self.setMaterial(item, face, materialId)
        # Return <claddingTextureInfo>, since it may be used by
        # the <renderCladding(..)> of a child class
        return claddingTextureInfo
    
    def setMaterial(self, item, face, materialId):
        self.r.setMaterial(item.building.element.l, face, materialId)

    def setCladdingUvs(self, item, face, claddingTextureInfo, uvs):
        textureWidthM = claddingTextureInfo["textureWidthM"]
        textureHeightM = textureWidthM * claddingTextureInfo["textureSize"][1] / claddingTextureInfo["textureSize"][0]
        self.r.setUvs(
            face,
            # a generator!
            ((uv[0]/textureWidthM, uv[1]/textureHeightM) for uv in uvs),
            item.building.element.l,
            item.building.element.l.uvLayerNameCladding
        )
    
    def _getCladdingTextureInfo(self, item):
        claddingMaterial = item.getCladdingMaterial()
        if not claddingMaterial:
            return None
        
        return self.app.assetStore.getAssetInfoCladdingTexture(
            item.building,
            item.getStyleBlockAttrDeep("group"),
            claddingMaterial,
            item.getStyleBlockAttrDeep("claddingClass")
        )
    
    def renderWithoutRepeat(self, item):
        
        # asset info could have been set in the call to item.getWidth(..)
        assetInfo = item.assetInfo
        # if <assetInfo=0>, then it was already queried in the asset store and nothing was found
        if assetInfo is None:
            assetInfo = self.getAssetInfo(item)
        
        face = self.r.createFace(item.footprint, item.indices)
        
        if assetInfo:
            if assetInfo["type"] == "texture":
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
                uvs = item.geometry.getClassUvs(texUl, texVb, texUr, texVt, item.uvs)
                
                if item.materialId is None:
                    self.setMaterialId(
                        item,
                        assetInfo,
                        uvs
                    )
                if item.materialId:
                    facadeTextureInfo, claddingTextureInfo = item.materialData
                    layer = item.building.element.l
                    self.r.setUvs(
                        face,
                        uvs,
                        layer,
                        layer.uvLayerNameFacade
                    )
                    self.renderExtra(item, face, facadeTextureInfo, claddingTextureInfo, item.uvs)
                    self.r.setMaterial(layer, face, item.materialId)
                else:
                    self.renderCladding(item, face, uvs)
            else:
                # Mesh assets are not considired at the moment
                assetInfo = None
                
        if not assetInfo:
            self.renderCladding(
                item,
                face,
                item.uvs
            )
            item.materialId = ""

    def getFacadeMaterialId(self, item, facadeTextureInfo, claddingTextureInfo):
        return facadeTextureInfo["name"] + "_" +claddingTextureInfo["name"]\
            if claddingTextureInfo\
            else facadeTextureInfo["name"]
    
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
        building, group, part, cl =\
            item.building, item.getStyleBlockAttrDeep("group"), item.getBuildingPart(), item.getClass()
        
        assetInfo = None
        if self.app.preferMesh:
            assetInfo = self.app.assetStore.getAssetInfo(
                True,
                building,
                group,
                part,
                cl
            )
        
            if assetInfo:
                item.assetInfo = assetInfo

        if not assetInfo:
            # try to get a texture asset
            assetInfo = self.app.assetStore.getAssetInfo(
                False,
                building,
                group,
                part,
                cl
            )
            if assetInfo:
                assetInfo = self.setAttributesForAssetInfoTexture(assetInfo)
                item.assetInfo = assetInfo
            else:
                # <0> prevents from subsequent querying <self.app.assetStore>
                item.assetInfo = 0
        
        return assetInfo
    
    def getTileWidthM(self, item):
        assetInfo = self.getAssetInfo(item)
        return assetInfo["tileWidthM"] if assetInfo else 0.

    def processModuleObject(self, objName, assetInfo):
        """
        Get and process a module object if it wasn't processed before.
        
        Returns:
            <False> if no module object with the name <objName> can be found,
            <True> otherwise.
        """
        
        # If <objectName> isn't available in <meshAssets>, that also means
        # that <objectName> isn't available in <self.r.buildingAssetsCollection.objects>
        if not objName in self.r.meshAssets:
            obj = linkObjectFromFile(getFilepath(self.r, assetInfo), None, objName)
            if not obj:
                return False
            self.processAssetMeshObject(obj, objName)
        return True