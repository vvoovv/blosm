from . import Item


class RoofPyramidal(Item):
    
    @classmethod
    def getItem(cls, itemFactory, parent, indices):
        item = itemFactory.getItem(cls)
        item.init()
        item.parent = parent
        item.footprint = parent
        item.footprint = parent
        item.building = parent.building
        item.indices = indices
        return item