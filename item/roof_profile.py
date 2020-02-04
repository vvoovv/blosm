from .roof_item import RoofItem
from .roof_side import RoofSide


class RoofProfile(RoofItem):
    
    def __init__(self):
        super().__init__()
        # a Python list of instances of item.roof_side.RoofSide
        self.roofSides = []
    
    def init(self):
        super().init()
        self.roofSides.clear()
    
    def addRoofSide(self, roofSideIndices, slotIndex, itemFactory):
        self.roofSides.append(
            RoofSide.getItem(itemFactory, self, roofSideIndices, slotIndex)
        )
    
    @classmethod
    def getItem(cls, itemFactory, parent, roofVertexData):
        item = itemFactory.getItem(cls)
        item.init()
        item.parent = parent
        item.footprint = parent
        item.setStyleBlock()
        item.building = parent.building
        item.roofVertexData = roofVertexData
        return item