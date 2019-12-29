from .. import ItemRenderer


class RoofFlat(ItemRenderer):
    
    def __init__(self):
        self.exportMaterials = False
    
    def render(self, roofItem):
        face = self.r.createFace(roofItem.building, roofItem.indices)