from .container import Container


class Top(Container):
    
    def __init__(self, parent, styleBlock):
        super().__init__(parent, parent, styleBlock)
        self.buildingPart = "top"
    
    def getLevelRenderer(self, levelGroup, itemRenderers):
        """
        Get a renderer for the <levelGroup> representing the item.
        """
        return itemRenderers["Top"]