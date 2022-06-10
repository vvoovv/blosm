import os
from util.blender_extra.material import setImage
from .container import Container
from ..entrance import Entrance as EntranceBase
from ...util import getPath


_entranceFaceWidthPx = 1028


class Entrance(EntranceBase, Container):
        
    def __init__(self):
        # a reference to the Container class used in the parent classes
        self.Container = Container
        Container.__init__(self, exportMaterials=True)
        EntranceBase.__init__(self)
    
    def getFacadeMaterialId(self, item, facadeTextureInfo, claddingTextureInfo):
        color = self.getCladdingColorHex(item)
        return "entrance_%s_%s_%s" % (claddingTextureInfo["name"], color, facadeTextureInfo["name"])\
            if claddingTextureInfo and color else\
            ("%s_%s" % (color, facadeTextureInfo["name"]) if color else facadeTextureInfo["name"])

    def renderLevelGroup(self, parentItem, levelGroup, indices, uvs):
        face = self.r.createFace(parentItem.footprint, indices)
        item = levelGroup.item
        if item.materialId is None:
            self.setMaterialId(
                item,
                parentItem.building,
                # building part
                "entrance",
                uvs
            )
        if item.materialId:
            self.r.setUvs(
                face,
                # we assume that the face is a rectangle
                (
                    (0., 0.), (1., 0.), (1., 1.), (0., 1.)
                ),
                self.r.layer.uvLayerNameFacade
            )
        self.r.setMaterial(face, item.materialId)
    
    def makeTexture(self, item, textureFilename, textureDir, textureFilepath, textColor, entranceTextureInfo, claddingTextureInfo, uvs):
        textureExporter = self.r.textureExporter
        scene = textureExporter.getTemplateScene("compositing_entrance_cladding_color")
        nodes = textureExporter.makeCommonPreparations(
            scene,
            textureFilename,
            textureDir
        )
        faceWidthM = uvs[1][0] - uvs[0][0]
        faceHeightM = uvs[2][1] - uvs[1][1]
        faceWidthPx = _entranceFaceWidthPx
        faceHeightPx = faceHeightM / faceWidthM * faceWidthPx
        # the size of the empty image
        image = nodes["empty_image"].image
        image.generated_width = faceWidthPx
        image.generated_height = faceHeightPx
        # entrance texture
        textureExporter.setImage(
            entranceTextureInfo["name"],
            getPath(self.r, entranceTextureInfo["path"]),
            nodes,
            "entrance_texture"
        )
        # scale for the entrance texture
        scaleY = entranceTextureInfo["textureHeightM"]/entranceTextureInfo["textureHeightPx"]*faceHeightPx/faceHeightM
        textureExporter.setScaleNode(
            nodes,
            "entrance_scale",
            entranceTextureInfo["textureWidthM"]/entranceTextureInfo["textureWidthPx"]*faceWidthPx/faceWidthM,
            scaleY
        )
        # translate for the entrance texture
        textureExporter.setTranslateNode(
            nodes,
            "entrance_translate",
            0,
            (scaleY*entranceTextureInfo["textureHeightPx"] - faceHeightPx)/2
        )
        # cladding texture
        textureExporter.setImage(
            claddingTextureInfo["name"],
            getPath(self.r, claddingTextureInfo["path"]),
            nodes,
            "cladding_texture"
        )
        # scale for the cladding texture
        scaleFactor = claddingTextureInfo["textureWidthM"]/claddingTextureInfo["textureWidthPx"]*\
            faceWidthPx/faceWidthM
        textureExporter.setScaleNode(
            nodes,
            "cladding_scale",
            scaleFactor,
            scaleFactor
        )
        # cladding color
        textureExporter.setColor(textColor, nodes, "cladding_color")
        # render the resulting texture
        textureExporter.renderTexture(scene, textureFilepath)