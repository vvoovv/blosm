from .roof_flat import RoofFlat
from item.roof_hipped import RoofHipped as ItemRoofHipped


class RoofHipped(RoofFlat):
    
    def __init__(self, data, itemStore, itemFactory, roofRenderer):
        super().__init__(data, itemStore, itemFactory, roofRenderer)
        self.extrudeTillRoof = True
    
    def extrude(self, footprint):
        if footprint.noWalls:
            return
        super().extrude(footprint)
    
    def getRoofItem(self, footprint, firstVertIndex):
        return ItemRoofHipped.getItem(
            self.itemFactory,
            footprint,
            firstVertIndex
        )