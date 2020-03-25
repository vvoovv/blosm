import os
import bpy
from util.blender_extra.material import createMaterialFromTemplate, setImage


_materialTemplateFilename = "building_material_templates.blend"
_materialTemplateName = "facade_overlay_template"
_curtainWallMaterialTemplateName = "facade_mirrored_glass_template"


class Level:
    
    def __init__(self):
        self.facadeMaterialTemplateFilename = _materialTemplateFilename
        self.facadeMaterialTemplateName = _materialTemplateName
        
        # do we need to initialize <self.facadePatternInfo>
        self.initFacadePatternInfo = True
        # The following Python dictionary is used to calculated the number of windows and balconies
        # in the Level pattern
        self.facadePatternInfo = dict(Window=0, Balcony=0, Door=0)
    
    def init(self, itemRenderers, globalRenderer):
        self.Container.init(self, itemRenderers, globalRenderer)
    
    def getNumLevelsInFace(self, levelGroup):
        return 1 if levelGroup.singleLevel else (levelGroup.index2-levelGroup.index1+1)


class CurtainWall(Level):
    
    def __init__(self):
        super().__init__()
        
        self.noCladdingTexture = True
        
        self.facadeMaterialTemplateName = _curtainWallMaterialTemplateName
        
        self.facadePatternInfo = None
        # do we need to initialize <self.facadePatternInfo>
        self.initFacadePatternInfo = False
    
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
                "Image Texture"
            )
            # specular map
            setImage(
                facadeTextureInfo["specularMapName"],
                os.path.join(self.r.bldgMaterialsDirectory, facadeTextureInfo["path"]),
                nodes,
                "Specular Map"
            )
        return True