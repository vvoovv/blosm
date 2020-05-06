

_materialTemplateFilename = "building_material_templates.blend"


class Level:
    
    def __init__(self):
        self.facadeMaterialTemplateFilename = _materialTemplateFilename
    
    def init(self, itemRenderers, globalRenderer):
        self.Container.init(self, itemRenderers, globalRenderer)
    
    def getNumLevelsInFace(self, levelGroup):
        return 1 if levelGroup.singleLevel else (levelGroup.index2-levelGroup.index1+1)


class CurtainWall(Level):
    
    def __init__(self):
        super().__init__()
        
        self.noCladdingTexture = True