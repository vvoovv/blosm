import os
import bpy
from util.blender_extra.material import createMaterialFromTemplate, setImage


_materialTemplateFilename = "building_material_templates.blend"


class Level:
    
    def __init__(self):
        self.facadeMaterialTemplateFilename = _materialTemplateFilename
        
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
        
        self.facadePatternInfo = None
        # do we need to initialize <self.facadePatternInfo>
        self.initFacadePatternInfo = False
    
    def createFacadeMaterial(self, materialName, facadeTextureInfo, claddingTextureInfo, uvs):
        materialTemplate = self.getFacadeMaterialTemplate(
            facadeTextureInfo,
            None,
            self.facadeMaterialTemplateFilename
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
            if facadeTextureInfo.get("specularMapName"):
                setImage(
                    facadeTextureInfo["specularMapName"],
                    os.path.join(self.r.bldgMaterialsDirectory, facadeTextureInfo["path"]),
                    nodes,
                    "Specular Map"
                )
        return True
    
    def getFacadeMaterialTemplate(self, facadeTextureInfo, claddingTextureInfo, materialTemplateFilename):
        useMixinColor = self.r.useMixinColor and not facadeTextureInfo["noMixinColor"]
        useSpecularMap = facadeTextureInfo.get("specularMapName")
        
        if useSpecularMap and useMixinColor:
            materialTemplateName = "facade_specular_color"
        elif useSpecularMap:
            materialTemplateName = "facade_specular"
        elif useMixinColor:
            materialTemplateName = "facade_color"
        else:
            materialTemplateName = "export"
        
        return self.getMaterialTemplate(materialTemplateFilename, materialTemplateName)