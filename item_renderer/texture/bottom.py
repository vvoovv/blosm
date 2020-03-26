_materialTemplateFilename = "building_material_templates.blend"


class Bottom:
    
    def __init__(self):
        self.facadeMaterialTemplateFilename = _materialTemplateFilename
        
        # do we need to initialize <self.facadePatternInfo>
        self.initFacadePatternInfo = True
        # The following Python dictionary is used to calculated the number of windows and doors
        # in the Bottom pattern
        self.facadePatternInfo = dict(Window=0, Door=0)
    
    def getNumLevelsInFace(self, levelGroup):
        return 1