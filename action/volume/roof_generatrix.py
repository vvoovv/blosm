from .roof_flat import RoofLeveled
from item.roof_generatrix import RoofGeneratrix as ItemRoofGeneratrix


class RoofGeneratrix(RoofLeveled):
    
    height = 4.
    
    def __init__(self, data, itemStore, itemFactory, facadeRenderer, roofRenderer):
        super().__init__(data, itemStore, itemFactory, facadeRenderer, roofRenderer)
        self.hasRoofLevels = False
        self.extrudeTillRoof = True
    
    def getRoofItem(self, footprint, firstVertIndex):
        return ItemRoofGeneratrix.getItem(
            self.itemFactory,
            footprint,
            firstVertIndex
        )