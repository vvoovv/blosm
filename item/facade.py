from .container import Container
from .level_groups import LevelGroups


class Facade(Container):
    """
    Represents a building facade.
    It's typically composed of one or more faces (in the most cases rectangular ones)
    """
    
    def __init__(self):
        super().__init__()
        self.levelGroups = LevelGroups(self)
    
    def init(self):
        self.levelGroups.clear()

    @classmethod
    def getItem(cls, volumeGenerator, parent, indices):
        item = volumeGenerator.itemFactory.getItem(cls)
        item.init()
        item.parent = parent
        item.footprint = parent
        item.building = parent.building
        item.indices = indices
        # <volumeGenerator> knows which geometry the facade items= has and how to map UV-coordinates
        volumeGenerator.initFacadeItem(item)
        return item
    
    @property
    def front(self):
        return True
    
    @property
    def back(self):
        return True