import os
import bpy
from ..container import Container as ContainerBase
from grammar.arrangement import Horizontal, Vertical
from grammar.symmetry import MiddleOfLast, RightmostOfLast

from util.blender_extra.material import createMaterialFromTemplate, setImage

from util import zAxis


_claddingMaterialTemplateFilename = "building_material_templates.blend"
_claddingMaterialTemplateName = "tiles_color_template"


class Container(ContainerBase):
    """
    The base class for the item renderers Facade, Div, Layer, Basement
    """
    
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
                nodes["Mapping"].inputs[3].default_value[0] = 1./claddingTextureInfo["textureHeightM"]
        return True
    
    def createCladdingMaterial(self, materialName, claddingTextureInfo):
        materialTemplate = self.getMaterialTemplate(
            _claddingMaterialTemplateFilename,
            _claddingMaterialTemplateName
        )
        if not materialName in bpy.data.materials:
            nodes = createMaterialFromTemplate(materialTemplate, materialName)
            # The wall material (i.e. background) texture,
            # set it just in case
            setImage(
                claddingTextureInfo["name"],
                os.path.join(self.r.bldgMaterialsDirectory, claddingTextureInfo["path"]),
                nodes,
                "Cladding Texture"
            )
            nodes["Mapping"].inputs[3].default_value[0] = 1./claddingTextureInfo["textureWidthM"]
            nodes["Mapping"].inputs[3].default_value[1] = 1./claddingTextureInfo["textureHeightM"]
        # return True for consistency with <self.getFacadeMaterialId(..)>
        return True

    def getFacadeMaterialId(self, item, facadeTextureInfo, claddingTextureInfo):
        return "%s_%s" % (facadeTextureInfo["name"], claddingTextureInfo["material"])\
            if claddingTextureInfo\
            else facadeTextureInfo["name"]