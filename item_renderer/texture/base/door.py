import os
import bpy
from .container import Container
from ..door import Door as DoorBase
from util.blender_extra.material import createMaterialFromTemplate, setImage


class Door(DoorBase, Container):
    
    def __init__(self):
        # a reference to the Container class used in the parent classes
        self.Container = Container
        Container.__init__(self)
        DoorBase.__init__(self)
    
    def render(self, building, levelGroup, parentItem, indices, uvs):
        face = self.r.createFace(building, indices)
        item = levelGroup.item
        if item.materialId is None:
            self.setMaterialId(
                item,
                building,
                # building part
                "door",
                uvs,
                # item renderer
                self
            )
        if item.materialId:
            facadeTextureInfo, claddingTextureInfo = item.materialData
            faceWidth = uvs[1][0] - uvs[0][0]
            faceHeight = uvs[2][1] - uvs[1][1]
            doorWidth = facadeTextureInfo["textureWidthM"]
            doorHeight = facadeTextureInfo["textureHeightM"]
            u1 = 0.5 - 0.5*faceWidth/doorWidth
            u2 = 1. - u1
            v = faceHeight/doorHeight
            self.r.setUvs(
                face,
                # we assume that the face is a rectangle
                (
                    (u1, 0.), (u2, 0.), (u2, v), (u1, v)
                ),
                self.r.layer.uvLayerNameFacade
            )
            # set UV-coordinates for the cladding texture
            self.setCladdingUvs(face, uvs, claddingTextureInfo)
            self.setVertexColor(item, face)
        self.r.setMaterial(face, item.materialId)
    
    def createFacadeMaterial(self, materialName, facadeTextureInfo, claddingTextureInfo, uvs):
        materialTemplate = self.getMaterialTemplate(
            self.facadeMaterialTemplateFilename,
            self.facadeMaterialTemplateName
        )
        if not materialName in bpy.data.materials:
            nodes = createMaterialFromTemplate(materialTemplate, materialName)
            # the overlay texture
            setImage(
                facadeTextureInfo["name"],
                os.path.join(self.r.bldgMaterialsDirectory, facadeTextureInfo["path"]),
                nodes,
                "Overlay"
            )
            if claddingTextureInfo:
                # The wall material (i.e. background) texture,
                # set it just in case
                setImage(
                    claddingTextureInfo["name"],
                    os.path.join(self.r.bldgMaterialsDirectory, claddingTextureInfo["path"]),
                    nodes,
                    "Wall Material"
                )
                nodes["Mapping"].inputs[3].default_value[0] = 1./claddingTextureInfo["textureWidthM"]
                nodes["Mapping"].inputs[3].default_value[1] = 1./claddingTextureInfo["textureHeightM"]
        return True