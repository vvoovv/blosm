from .container import Container


_materialTemplateFilename = "building_material_templates.blend"
_materialTemplateName = "facade_overlay_template"


class Basement(Container):
    
    def __init__(self):
        self.facadeMaterialTemplateFilename = _materialTemplateFilename
        self.facadeMaterialTemplateName = _materialTemplateName
        
        # do we need to initialize <self.facadePatternInfo>
        self.initFacadePatternInfo = True
        # The following Python dictionary is used to calculated the number of windows and doors
        # in the Basement pattern
        self.facadePatternInfo = dict(Window=0, Door=0)
    
    def getHeightForMaterial(self, levelGroup):
        return levelGroup.item.footprint.levelHeights.basementHeight
    
    def getNumLevelsInFace(self, levelGroup):
        return 1