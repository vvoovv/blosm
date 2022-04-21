from .container import Container


class Level(Container):
    
    def __init__(self, parent, styleBlock):
        super().__init__(parent, parent.footprint, styleBlock)
        self.buildingPart = "level"
    
    def getLevelRenderer(self, levelGroup, itemRenderers):
        """
        Get a renderer for the <levelGroup> representing the item.
        """
        # here is the special case: the door which the only item in the markup
        return itemRenderers["Door"]\
            if len(self.markup) == 1 and self.markup[0].__class__.__name__ == "Door"\
            else itemRenderers["Level"]


class CurtainWall(Level):
        
    width = 0.
    
    def __init__(self, parent, styleBlock):
        super().__init__(parent, styleBlock)
        self.buildingPart = "curtain_wall"
        # It doesn't need the <facadePatternInfo>
        self.hasFacadePatternInfo = False
    
    def getLevelRenderer(self, levelGroup, itemRenderers):
        """
        Get a renderer for the <levelGroup> representing the item.
        """
        return itemRenderers["CurtainWall"]
    
    def getWidth(self):
        width = self.getStyleBlockAttr("width")
        if width is None:
            width = CurtainWall.width
        self.width = width
        return width