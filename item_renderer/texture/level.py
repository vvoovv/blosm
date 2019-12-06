_materialTemplateFilename = "building_material_templates.blend"
_materialTemplateName = "facade_overlay_template"


class Level:
    """
    A mixin class for Level texture based item renderers
    """
    
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
        self.doorRenderer = itemRenderers["Door"]
    
    def getNumLevelsInFace(self, levelGroup):
        return 1 if levelGroup.singleLevel else (levelGroup.index2-levelGroup.index1+1)
    
    def getRenderer(self, levelGroup):
        item = levelGroup.item
        # here is the special case: the door which the only item in the markup
        return self.doorRenderer\
            if len(item.markup) == 1 and item.markup[0].__class__.__name__ == "Door"\
            else self