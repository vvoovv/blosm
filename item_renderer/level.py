import os
import bpy
from .container import Container
from util.blender import loadMaterialsFromFile
from util.blender_extra.material import createMaterialFromTemplate, setImage, setCustomNodeValue


_materialTemplateFilename = "building_material_templates.blend"
_materialTemplateName = "facade_overlay_template"


class Level(Container):
    
    def __init__(self):
        # The following Python dictionary is used to calculated the number of windows and balconies
        # in the Level pattern
        self.facadePatternInfo = dict(Window=0, Balcony=0, Door=0)
    
    def init(self, itemRenderers, globalRenderer):
        super().init(itemRenderers, globalRenderer)
    
    def render(self, building, levelGroup, indices, uvs, texOffsetU, texOffsetV):
        item = levelGroup.item
        face = self.r.createFace(item.building, indices, uvs)
        if item.markup:
            # process the special case: the door which the only item in the markup
            isSingleDoor = len(item.markup) == 1 and item.markup[0].__class__.__name__ == "Door"
            if isSingleDoor:
                self.setMaterialId(
                    item,
                    building,
                    # building part
                    "door",
                    # item renderer
                    self.itemRenderers["Door"]
                )
            else:
                self.setMaterialId(
                    item,
                    building,
                    # getting building part
                    item.buildingPart if item.buildingPart else (
                        "groundlevel" if levelGroup.singleLevel and not levelGroup.index1 else "level"
                    ),
                    self
                )
        if item.materialId:
            if isSingleDoor:
                self.setData(
                    face,
                    self.r.layer.uvNameSize,
                    # face width
                    item.width
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
            else:
                levelHeights = item.footprint.levelHeights
                self.setData(
                    face,
                    self.r.layer.uvNameSize,
                    (
                        # face width
                        item.width,
                        # level height
                        levelHeights.getLevelHeight(levelGroup.index1)\
                        if levelGroup.singleLevel else\
                        levelHeights.getHeight(levelGroup.index1, levelGroup.index2)/(levelGroup.index2-levelGroup.index1)
                    )
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
        textureWidthPx = textureInfo["textureWidthPx"]
        textureHeightPx = textureInfo["textureHeightPx"]
        numberOfTilesU = textureInfo["numTilesU"]
        numberOfTilesV = textureInfo["numTilesV"]
        tileWidthPx = textureWidthPx/numberOfTilesU
        # factor = windowWidthM/windowWidthPx
        factor = textureInfo["windowWidthM"]/(textureInfo["windowRpx"]-textureInfo["windowLpx"])

        textureWidthM = factor*textureWidthPx
        tileSizeUdefaultM = factor*tileWidthPx
        textureUoffsetM = 0.
        
        textureLevelHeightM = factor*textureHeightPx/numberOfTilesV
        textureHeightM = factor*textureHeightPx
        textureVoffsetM = 0.
        
        materialTemplateName = _materialTemplateName
        customNode = "FacadeOverlay"
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
            nodes["Mapping"].scale[0] = 1./wallTextureWidthM
            nodes["Mapping"].scale[1] = 1./wallTextureHeightM
            # the mask for the emission
            #setImage(fileName, directory, nodes, "Emission Mask", "emissive")
            # setting nodes
            n = nodes[customNode]
            setCustomNodeValue(n, "Texture Width", textureWidthM)
            setCustomNodeValue(n, "Number of Tiles U", numberOfTilesU)
            setCustomNodeValue(n, "Tile Size U Default", tileSizeUdefaultM)
            setCustomNodeValue(n, "Texture U-Offset", textureUoffsetM)
            setCustomNodeValue(n, "Number of Tiles V", numberOfTilesV)
            setCustomNodeValue(n, "Texture Level Height", textureLevelHeightM)
            setCustomNodeValue(n, "Texture Height", textureHeightM)
            setCustomNodeValue(n, "Texture V-Offset", textureVoffsetM)
        return True
    
    def getMaterialId(self, textureInfo):
        return textureInfo["name"]