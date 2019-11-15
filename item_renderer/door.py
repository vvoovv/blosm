import os
import bpy
from manager import Manager
from .container import Container
from util.blender import loadMaterialsFromFile
from util.blender_extra.material import createMaterialFromTemplate, setImage


_materialTemplateFilename = "building_material_templates.blend"
_materialTemplateName = "door_overlay_template"


class Door(Container):
    """
    The Door renderer is the special case of the <item_renderer.level.Level> when
    a door in the only element in the level markup
    """
    
    def __init__(self):
        self.facadeMaterialTemplateFilename = _materialTemplateFilename
        self.facadeMaterialTemplateName = _materialTemplateName
        # do we need to initialize <self.facadePatternInfo>
        self.initFacadePatternInfo = False
        self.facadePatternInfo = dict(Door=1)
    
    def render(self, building, levelGroup, parentItem, indices, uvs, texOffsetU, texOffsetV):
        face = self.r.createFace(building, indices, uvs)
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
            self.setData(
                face,
                self.r.layer.uvNameSize,
                # face width
                parentItem.width
            )
            self.setData(
                face,
                self.uvLayer,
                (
                    # offset for the texture U-coordinate
                    texOffsetU,
                    # offset for the texture V-coordinate
                    texOffsetV
                )
            )
            self.setVertexColor(item, face)
        self.r.setMaterial(face, item.materialId)
    
    def createFacadeMaterial(self, materialName, facadeTextureInfo, claddingTextureInfo):
        materialTemplate = self.getMaterialTemplate(
            self.facadeMaterialTemplateFilename,
            self.facadeMaterialTemplateName
        )
        if not materialName in bpy.data.materials:
            bldgMaterialsDirectory = os.path.dirname(self.r.app.bldgMaterialsFilepath)
            nodes = createMaterialFromTemplate(materialTemplate, materialName)
            # the overlay texture
            setImage(
                facadeTextureInfo["name"],
                os.path.join(bldgMaterialsDirectory, facadeTextureInfo["path"]),
                nodes,
                "Overlay"
            )
            nodes["Door Width"].outputs[0].default_value = facadeTextureInfo["textureWidthM"]
            nodes["Door Height"].outputs[0].default_value = facadeTextureInfo["textureHeightM"]
            if claddingTextureInfo:
                # The wall material (i.e. background) texture,
                # set it just in case
                setImage(
                    claddingTextureInfo["name"],
                    os.path.join(bldgMaterialsDirectory, claddingTextureInfo["path"]),
                    nodes,
                    "Wall Material"
                )
                nodes["Mapping"].scale[0] = 1./claddingTextureInfo["textureWidthM"]
                nodes["Mapping"].scale[1] = 1./claddingTextureInfo["textureHeightM"]
        return True