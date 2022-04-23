import bpy
from .container import Container
from ..entrance import Entrance as EntranceBase
from util.blender_extra.material import createMaterialFromTemplate, setImage
from ...util import getPath


class Entrance(EntranceBase, Container):
    
    def __init__(self):
        # a reference to the Container class used in the parent classes
        self.Container = Container
        Container.__init__(self, exportMaterials=False)
        EntranceBase.__init__(self)
    
    def renderLevelGroup(self, parentItem, levelGroup, indices, uvs):
        face = self.r.createFace(parentItem.building, indices)
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
            facadeTextureInfo, claddingTextureInfo = item.materialData
            faceWidth = uvs[1][0] - uvs[0][0]
            faceHeight = uvs[2][1] - uvs[1][1]
            entranceWidth = facadeTextureInfo["textureWidthM"]
            entranceHeight = facadeTextureInfo["textureHeightM"]
            u1 = 0.5 - 0.5*faceWidth/entranceWidth
            u2 = 1. - u1
            v = faceHeight/entranceHeight
            self.r.setUvs(
                face,
                # we assume that the face is a rectangle
                (
                    (u1, 0.), (u2, 0.), (u2, v), (u1, v)
                ),
                self.r.layer.uvLayerNameFacade
            )
            # set UV-coordinates for the cladding texture
            self.setCladdingUvs(item, face, claddingTextureInfo, uvs)
            self.setVertexColor(item, face)
        self.r.setMaterial(face, item.materialId)
    
    def createFacadeMaterial(self, item, materialName, facadeTextureInfo, claddingTextureInfo, uvs):
        if not materialName in bpy.data.materials:
            materialTemplate = self.getFacadeMaterialTemplate(
                facadeTextureInfo,
                claddingTextureInfo
            )
            nodes = createMaterialFromTemplate(materialTemplate, materialName)
            # the overlay texture
            setImage(
                facadeTextureInfo["name"],
                getPath(self.r, facadeTextureInfo["path"]),
                nodes,
                "Main"
            )
            if claddingTextureInfo:
                # The wall material (i.e. background) texture,
                # set it just in case
                setImage(
                    claddingTextureInfo["name"],
                    getPath(self.r, claddingTextureInfo["path"]),
                    nodes,
                    "Cladding"
                )
        return True

    def getFacadeMaterialTemplate(self, facadeTextureInfo, claddingTextureInfo):
        if claddingTextureInfo:
            materialTemplateName = "entrance_cladding_color" if self.r.useCladdingColor else "entrance_cladding"
        else:
            materialTemplateName = "export"
        return self.getMaterialTemplate(materialTemplateName)