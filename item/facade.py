from .container import Container
from .level_groups import LevelGroups


class Facade(Container):
    """
    Represents a building facade.
    It's typically composed of one or more faces (in the most cases rectangular ones)
    """
    
    def __init__(self):
        super().__init__()
        self.valid = True
        self.faces = []
        self.levelGroups = LevelGroups(self)
    
    def init(self):
        self.valid = True
        self.faces.clear()
        self.levelGroups.clear()
        
        self.normal = None
        self.type = ("front", "back", "side")
        self.neighborL = None
        self.neighborR = None

    @classmethod
    def getItem(cls, itemFactory, parent, indices, width, heightLeft, heightRightOffset):
        item = itemFactory.getItem(cls)
        item.init()
        item.parent = parent
        item.footprint = parent
        item.footprint = parent
        item.building = parent.building
        item.indices = indices
        item.width = width
        # assign uv-coordinates
        item.uvs = ( (0., 0.), (width, heightRightOffset), (width, heightLeft), (0., heightLeft) )
        return item
    
    @property
    def front(self):
        return True
    
    @property
    def back(self):
        return True