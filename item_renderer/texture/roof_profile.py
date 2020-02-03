from .. import ItemRenderer
from grammar import smoothness


class RoofProfile(ItemRenderer):
        
    def render(self, roofItem):
        building = roofItem.building
        smoothFaces = roofItem.getStyleBlockAttr("faces") is smoothness.Smooth
        
        for roofSide in roofItem.roofSides:
            face = self.r.createFace(
                building,
                roofSide.indices
            )
            if smoothFaces:
                face.smooth = True
            #self.renderCladding(building, roofItem, face, None)