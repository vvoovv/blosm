from .container import Container


class Basement(Container):
    
    def __init__(self):
        super().__init__()
        self.buildingPart = "basement"

    @classmethod
    def getItem(cls, itemFactory, parent, styleBlock):
        item = itemFactory.getItem(cls)
        item.init()
        item.parent = parent
        item.footprint = parent.footprint
        item.building = parent.building
        item.styleBlock = styleBlock
        return item