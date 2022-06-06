from .. import ItemRenderer
from util.blender import loadImage
from ..util import getPath


class ItemRendererTexture(ItemRenderer):
    
    def __init__(self, exportMaterials=False):
        super().__init__(exportMaterials)
    
    def getAssetType(self):
        return "texture"
    
    def setAttributesForAssetInfoTexture(self, assetInfo):
        image = loadImage(
            assetInfo["name"],
            getPath(self.r, assetInfo["path"])
        )
        if image:
            textureSize = assetInfo["textureSize"] = tuple(image.size)
            assetInfo["tileWidthM"] = textureSize[0]/\
                (assetInfo["featureRpx"] - assetInfo["featureLpx"])*\
                assetInfo["featureWidthM"] / assetInfo["numTilesU"]
        return assetInfo
            