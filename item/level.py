from .container import Container


class Level(Container):
    
    def __init__(self):
        super().__init__()

    @classmethod
    def getItem(cls, itemFactory, parent, styleBlock):
        item = itemFactory.getItem(cls)
        item.init()
        item.parent = parent
        item.building = parent.building
        item.styleBlock = styleBlock
        return item