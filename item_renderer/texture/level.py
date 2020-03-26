

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