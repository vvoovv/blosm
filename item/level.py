from .container import Container


class Level(Container):
    
    def __init__(self):
        super().__init__()
        self.buildingPart = "level"
    
    def getLevelRenderer(self, levelGroup, itemRenderers):
        """
        Get a renderer for the <levelGroup> representing the item.
        """
        # here is the special case: the door which the only item in the markup
        return itemRenderers["Door"]\
            if len(self.markup) == 1 and self.markup[0].__class__.__name__ == "Door"\
            else itemRenderers["Level"]
        
    @classmethod
    def getItem(cls, itemFactory, parent, styleBlock):
        item = itemFactory.getItem(cls)
        item.init()
        item.parent = parent
        item.footprint = parent.footprint
        item.building = parent.building
        item.styleBlock = styleBlock
        return item


class CurtainWall(Level):
        
    width = 0.
    
    def __init__(self):
        super().__init__()
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
    
    @classmethod
    def getItem(cls, itemFactory, parent, styleBlock):
        item = itemFactory.getItem(cls)
        item.init()
        item.parent = parent
        item.footprint = parent.footprint
        item.building = parent.building
        item.styleBlock = styleBlock
        return item