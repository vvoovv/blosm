from .container import Container


class Level(Container):
    
    def __init__(self, parent, styleBlock):
        super().__init__(parent, parent.footprint, styleBlock)
        self.isContainer = True
        self.buildingPart = "level"
    
    def getLevelRenderer(self, levelGroup, itemRenderers):
        """
        Get a renderer for the <levelGroup> representing the item.
        """
        # here is the special case: the entrance which the only item in the markup
        return itemRenderers["Entrance"]\
            if len(self.markup) == 1 and self.markup[0].__class__.__name__ == "Entrance"\
            else itemRenderers["Level"]