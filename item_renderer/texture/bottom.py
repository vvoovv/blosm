_materialTemplateFilename = "building_material_templates.blend"


class Bottom:
    
    def __init__(self):
        self.facadeMaterialTemplateFilename = _materialTemplateFilename
    
    def getNumLevelsInFace(self, levelGroup):
        return 1