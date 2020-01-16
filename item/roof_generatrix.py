from .roof_item import RoofItem


class RoofGeneratrix(RoofItem):
    
    @classmethod
    def getItem(cls, itemFactory, parent, firstVertIndex):
        item = itemFactory.getItem(cls)
        item.init()
        item.parent = parent
        item.footprint = parent
        item.setStyleBlock()
        item.building = parent.building
        item.firstVertIndex = firstVertIndex
        return item