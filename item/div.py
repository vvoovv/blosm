from .container import Container
from .level_styles import LevelStyles


class Div(Container):
    
    def __init__(self):
        super().__init__()
        self.levelStyles = LevelStyles()
    
    @classmethod
    def getItem(cls, itemFactory, parent, styleBlock):
        item = itemFactory.getItem(cls)
        item.init()
        item.parent = parent
        item.styleBlock = styleBlock
        return item