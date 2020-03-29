import os
from util.blender_extra.material import setImage
from .container import Container
from ..door import Door as DoorBase


_doorFaceWidthPx = 1028


class Door(DoorBase, Container):
        
    def __init__(self):
        # a reference to the Container class used in the parent classes
        self.Container = Container
        Container.__init__(self, exportMaterials=True)
        DoorBase.__init__(self)
    
    def getFacadeMaterialId(self, item, facadeTextureInfo, claddingTextureInfo):
        color = self.getCladdingColor(item)
        return "door_%s_%s_%s" % (claddingTextureInfo["material"], color, facadeTextureInfo["name"])\
            if claddingTextureInfo and color\
            else facadeTextureInfo["name"]

    def renderLevelGroup(self, parentItem, levelGroup, indices, uvs):
        face = self.r.createFace(parentItem.building, indices)
        item = levelGroup.item
        if item.materialId is None:
            self.setMaterialId(
                item,
                parentItem.building,
                # building part
                "door",
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
    
    def makeTexture(self, textureFilename, textureDir, textColor, doorTextureInfo, claddingTextureInfo, uvs):
        textureExporter = self.r.textureExporter
        scene = textureExporter.getTemplateScene("compositing_door_cladding_color")
        nodes = textureExporter.makeCommonPreparations(
            scene,
            textureFilename,
            textureDir
        )
        faceWidthM = uvs[1][0] - uvs[0][0]
        faceHeightM = uvs[2][1] - uvs[1][1]
        faceWidthPx = _doorFaceWidthPx
        faceHeightPx = faceHeightM / faceWidthM * faceWidthPx
        # the size of the empty image
        image = nodes["empty_image"].image
        image.generated_width = faceWidthPx
        image.generated_height = faceHeightPx
        # door texture
        setImage(
            doorTextureInfo["name"],
            os.path.join(textureExporter.bldgMaterialsDirectory, doorTextureInfo["path"]),
            nodes,
            "door_texture"
        )
        # scale for the door texture
        scaleY = doorTextureInfo["textureHeightM"]/doorTextureInfo["textureHeightPx"]*faceHeightPx/faceHeightM
        self.setScaleNode(
            nodes,
            "door_scale",
            doorTextureInfo["textureWidthM"]/doorTextureInfo["textureWidthPx"]*faceWidthPx/faceWidthM,
            scaleY
        )
        # translate for the door texture
        self.setTranslateNode(
            nodes,
            "door_translate",
            0,
            (scaleY*doorTextureInfo["textureHeightPx"] - faceHeightPx)/2
        )
        # cladding texture
        setImage(
            claddingTextureInfo["name"],
            os.path.join(textureExporter.bldgMaterialsDirectory, claddingTextureInfo["path"]),
            nodes,
            "cladding_texture"
        )
        # scale for the cladding texture
        scaleFactor = claddingTextureInfo["textureWidthM"]/claddingTextureInfo["textureWidthPx"]*\
            faceWidthPx/faceWidthM
        self.setScaleNode(
            nodes,
            "cladding_scale",
            scaleFactor,
            scaleFactor
        )
        # cladding color
        self.setColor(textColor, nodes, "cladding_color")
        # render the resulting texture
        textureExporter.renderTexture(scene, textureFilename, textureDir)