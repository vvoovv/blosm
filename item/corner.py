from . import Item


class Corner(Item):
    
    def __init__(self, parent, footprint, styleBlock):
        super().__init__(parent, footprint, parent.facade, styleBlock)
        self.buildingPart = "corner"
    
    def getClass(self):
        # First we check, if the corner item has the "class" attribute.
        # Then we perform a deep search for the attribute "cornerClass"
        return self.getStyleBlockAttr("cl") or self.getStyleBlockAttrDeep("cornerClass")