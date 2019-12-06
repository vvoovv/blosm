import os
import bpy
from util.blender_extra.material import createMaterialFromTemplate, setImage


_materialTemplateFilename = "building_material_templates.blend"
_materialTemplateName = "door_overlay_template"


class Door:
    """
    The Door renderer is the special case of the <item_renderer.level.Level> when
    a door in the only element in the level markup
    
    A mixin class for Door texture based item renderers
    """
    
    def __init__(self):
        self.facadeMaterialTemplateFilename = _materialTemplateFilename
        self.facadeMaterialTemplateName = _materialTemplateName
        # do we need to initialize <self.facadePatternInfo>
        self.initFacadePatternInfo = False
        self.facadePatternInfo = dict(Door=1)
    
    def render(self, building, levelGroup, parentItem, indices, uvs, texOffsetU, texOffsetV):
        face = self.r.createFace(building, indices)
        # set UV-coordinates for the cladding textures
        if not self.exportMaterials:
            self.setCladdingUvs(face, uvs)
        item = levelGroup.item
        if item.materialId is None:
            self.setMaterialId(
                item,
                building,
                # building part
                "door",
                # item renderer
                self
            )
        if item.materialId:
            facadeTextureInfo = item.materialData
            faceWidth = uvs[1][0] - texOffsetU
            faceHeight = uvs[2][1] - texOffsetV
            doorWidth = facadeTextureInfo["textureWidthM"]
            doorHeight = facadeTextureInfo["textureHeightM"]
            u1 = 0.5 - 0.5*faceWidth/doorWidth
            u2 = 1. - u1
            v = faceHeight/doorHeight
            self.r.setUvs(
                face,
                (
                    (u1, 0.), (u2, 0.), (u2, v), (u1, v)
                ),
                self.r.layer.uvLayerNameFacade
            )
            self.setVertexColor(item, face)
        self.r.setMaterial(face, item.materialId)
    
    def createFacadeMaterial(self, materialName, facadeTextureInfo, claddingTextureInfo):
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