from .container import Container
from .level_groups import LevelGroups


class Div(Container):
    
    def __init__(self, parent, footprint, styleBlock):
        super().__init__(parent, footprint, styleBlock)
        self.levelGroups = LevelGroups(self)
        self.minHeight = self.footprint.minHeight
        self.minLevel = self.footprint.minLevel
        self.highEnoughForLevel = True