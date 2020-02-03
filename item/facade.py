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
    def getItem(cls, itemFactory, parent, geometry, indices, uvs):
        item = itemFactory.getItem(cls)
        item.init()
        item.parent = parent
        item.footprint = parent
        item.building = parent.building
        item.indices = indices
        item.width = geometry.getWidth(uvs)
        item.geometry = geometry
        # assign uv-coordinates (i.e. surface coordinates on the facade plane)
        item.uvs = uvs
        return item
    
    @property
    def front(self):
        return True
    
    @property
    def back(self):
        return True