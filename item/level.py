from .container import Container
from grammar.arrangement import Vertical


class Level(Container):
    
    def __init__(self):
        super().__init__()
        # the arrangement of markup items is always vertical
        self.arrangement = Vertical

    @classmethod
    def getItem(cls, itemFactory, parent, styleBlock):
        item = itemFactory.getItem(cls)
        item.init()
        item.parent = parent
        item.styleBlock = styleBlock
        return item