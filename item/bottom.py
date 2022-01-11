from .container import Container


class Bottom(Container):
    
    def __init__(self, parent, styleBlock):
        super().__init__(parent, parent, styleBlock)
        self.buildingPart = "bottom"