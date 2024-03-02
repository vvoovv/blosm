import bpy

from util.blender import loadMaterialsFromFile
from item_renderer.util import getFilepath


class ItemRenderer:
    
    def init(self, globalRenderer):
        self.r = globalRenderer
        self.assetStore = globalRenderer.assetStore
    
    def prepare(self):
        return
    
    def finalize(self):
        return
    
    def setMaterial(self, modifier, modifierAttr, assetType, group, streetPart, cl):
        # get asset info for the material
        assetInfo = self.assetStore.getAssetInfo(
            assetType, group, streetPart, cl
        )
        if assetInfo:
            # set material
            material = self.getMaterial(assetInfo)
            if material:
                modifier[modifierAttr] = material
    
    def getMaterial(self, assetInfo):
        materialName = assetInfo["material"]
        material = bpy.data.materials.get(materialName)
        
        if not material:
            material = loadMaterialsFromFile(
                getFilepath(self.r, assetInfo),
                False,
                materialName
            )
            material = material[0] if material else None
            
        return material