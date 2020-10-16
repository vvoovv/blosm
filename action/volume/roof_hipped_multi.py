from .roof_flat_multi import RoofMulti
from .roof_hipped import RoofHipped
from item.roof_hipped_multi import RoofHippedMulti as ItemRoofHippedMulti


class RoofHippedMulti(RoofMulti, RoofHipped):
    
    def getRoofItem(self, footprint):
        return ItemRoofHippedMulti.getItem(self.itemFactory, footprint)
    
    def render(self, footprint, roofItem):
        return
        