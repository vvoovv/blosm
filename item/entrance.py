from . import Item


class Entrance(Item):
    
    def __init__(self, parent, footprint, styleBlock):
        super().__init__(parent, footprint, parent.facade, styleBlock)
        self.buildingPart = "entrance"