from .roof_flat import RoofLeveled
from item.roof_generatrix import RoofGeneratrix as ItemRoofGeneratrix


class RoofGeneratrix(RoofLeveled):
    
    height = 4.
    
    def __init__(self, roofRendererId, data, volumeAction, itemRenderers):
        super().__init__(roofRendererId, data, volumeAction, itemRenderers)
        self.hasRoofLevels = False
        self.extrudeTillRoof = True
    
    def getRoofItem(self, footprint):
        return ItemRoofGeneratrix.getItem(
            self.itemFactory,
            footprint,
            self.getRoofFirstVertIndex(footprint)
        )