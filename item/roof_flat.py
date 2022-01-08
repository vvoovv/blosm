from .roof_item import RoofItem


class RoofFlat(RoofItem):
    
    def __init__(self, parent, firstVertIndex):
        super().__init__(parent)
        self.firstVertIndex = firstVertIndex