from .container import Container


_materialTemplateFilename = "building_material_templates.blend"
_materialTemplateName = "facade_overlay_template"


class Level(Container):
    
    def __init__(self):
        self.materialTemplateFilename = _materialTemplateFilename
        self.materialTemplateName = _materialTemplateName
        
        # do we need to initialize <self.facadePatternInfo>
        self.initFacadePatternInfo = True
        # The following Python dictionary is used to calculated the number of windows and balconies
        # in the Level pattern
        self.facadePatternInfo = dict(Window=0, Balcony=0, Door=0)
    
    def init(self, itemRenderers, globalRenderer):
        super().init(itemRenderers, globalRenderer)
        self.doorRenderer = itemRenderers["Door"]
    
    def getHeightForMaterial(self, levelGroup):
        levelHeights = levelGroup.item.footprint.levelHeights
        # return level height
        return levelHeights.getLevelHeight(levelGroup.index1)\
            if levelGroup.singleLevel else\
            levelHeights.getHeight(levelGroup.index1, levelGroup.index2)/(levelGroup.index2-levelGroup.index1)
    
    def getRenderer(self, levelGroup):
        item = levelGroup.item
        # here is the special case: the door which the only item in the markup
        return self.doorRenderer\
            if len(item.markup) == 1 and item.markup[0].__class__.__name__ == "Door"\
            else self