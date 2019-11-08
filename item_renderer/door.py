import os
import bpy
from .container import Container
from util.blender import loadMaterialsFromFile
from util.blender_extra.material import createMaterialFromTemplate, setImage, setCustomNodeValue


_materialTemplateFilename = "building_material_templates.blend"
_materialTemplateName = "door_overlay_template"


class Door(Container):
    """
    The Door renderer is the special case of the <item_renderer.level.Level> when
    a door in the only element in the level markup
    """
    
    def __init__(self):
        self.materialTemplateFilename = _materialTemplateFilename
        self.materialTemplateName = _materialTemplateName
        # do we need to initialize <self.facadePatternInfo>
        self.initFacadePatternInfo = False
        self.facadePatternInfo = dict(Door=1)
    
    def render(self, building, levelGroup, parentItem, indices, uvs, texOffsetU, texOffsetV):
        item = levelGroup.item
        face = self.r.createFace(item.building, indices, uvs)
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
            self.setColor(face, self.vertexColorLayer, (0.7, 0.3, 0.3, 1.))
        self.r.setMaterial(face, item.materialId)
    
    def createMaterial(self, materialName, textureInfo):
        materialTemplateName = _materialTemplateName
        wallTextureWidthM = 1.5
        wallTextureHeightM = 1.5
        wallTextureFilename = "cc0textures_bricks11_col.jpg"
        wallTexturePath = "textures/cladding/brick"
        
        materialTemplate = bpy.data.materials.get(materialTemplateName)
        if not materialTemplate:
            bldgMaterialsDirectory = os.path.dirname(self.r.app.bldgMaterialsFilepath)
            materialTemplate = loadMaterialsFromFile(os.path.join(bldgMaterialsDirectory, _materialTemplateFilename), True, materialTemplateName)[0]
        if not materialName in bpy.data.materials:
            bldgMaterialsDirectory = os.path.dirname(self.r.app.bldgMaterialsFilepath)
            nodes = createMaterialFromTemplate(materialTemplate, materialName)
            # the overlay texture
            setImage(
                textureInfo["name"],
                os.path.join(bldgMaterialsDirectory, textureInfo["path"]),
                nodes,
                "Overlay"
            )
            # The wall material (i.e. background) texture,
            # set it just in case
            setImage(
                wallTextureFilename,
                os.path.join(bldgMaterialsDirectory, wallTexturePath),
                nodes,
                "Wall Material"
            )
            nodes["Door Width"].outputs[0].default_value = textureInfo["textureWidthM"]
            nodes["Door Height"].outputs[0].default_value = textureInfo["textureHeightM"]
            nodes["Mapping"].scale[0] = 1./wallTextureWidthM
            nodes["Mapping"].scale[1] = 1./wallTextureHeightM
        return True