from .roof_item import RoofWithSidesItem


class RoofProfile(RoofWithSidesItem):
    
    @classmethod
    def getItem(cls, itemFactory, parent):
        item = itemFactory.getItem(cls)
        item.init()
        item.parent = parent
        item.footprint = parent
        item.setStyleBlock()
        item.building = parent.building
        return item