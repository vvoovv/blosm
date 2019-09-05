from .container import Container
from .level_groups import LevelGroups


class Div(Container):
    
    def __init__(self):
        super().__init__()
        self.levelGroups = LevelGroups(self)
        
    def init(self):
        super().init()
        self.levelGroups.clear()
    
    @classmethod
    def getItem(cls, itemFactory, parent, styleBlock):
        item = itemFactory.getItem(cls)
        item.init()
        item.parent = parent
        item.footprint = parent.footprint
        item.building = parent.building
        item.styleBlock = styleBlock
        return item