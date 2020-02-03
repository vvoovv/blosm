from .roof_flat import RoofFlat
from item.roof_generatrix import RoofGeneratrix as ItemRoofGeneratrix


class RoofGeneratrix(RoofFlat):
    
    def __init__(self, data, itemStore, itemFactory, roofRenderer):
        super().__init__(data, itemStore, itemFactory, roofRenderer)
        self.hasRoofLevels = False
        self.extrudeTillRoof = True
    
    def extrude(self, footprint):
        if footprint.noWalls:
            return
        super().extrude(footprint)
    
    def getRoofItem(self, footprint, firstVertIndex):
        return ItemRoofGeneratrix.getItem(
            self.itemFactory,
            footprint,
            firstVertIndex
        )