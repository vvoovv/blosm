import os
import bpy
from . import ItemRenderer
from util.blender import loadMaterialsFromFile
from util.blender_extra.material import createMaterialFromTemplate, setImage, setCustomNodeValue


_materialTemplateFilename = "building_material_templates.blend"
_materialTemplateName = "door_overlay_template"


class Door(ItemRenderer):

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