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
        self.materialTemplateFilename = _materialTemplateFilename
        self.materialTemplateName = _materialTemplateName
        # do we need to initialize <self.facadePatternInfo>
        self.initFacadePatternInfo = False
        self.facadePatternInfo = dict(Door=1)
    
    def render(self, building, levelGroup, parentItem, indices, uvs, texOffsetU, texOffsetV):
        item = levelGroup.item
        face = self.r.createFace(item.building, indices, uvs)
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
            claddingColor = Manager.normalizeColor(item.getStyleBlockAttrDeep("claddingColor"))
            if claddingColor:
                self.setColor(face, self.vertexColorLayer, Manager.getColor(claddingColor))
        self.r.setMaterial(face, item.materialId)
    
    def createMaterial(self, materialName, facadeTextureInfo, claddingTextureInfo):
        materialTemplate = bpy.data.materials.get(_materialTemplateName)
        if not materialTemplate:
            bldgMaterialsDirectory = os.path.dirname(self.r.app.bldgMaterialsFilepath)
            materialTemplate = loadMaterialsFromFile(os.path.join(bldgMaterialsDirectory, _materialTemplateFilename), True, _materialTemplateName)[0]
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