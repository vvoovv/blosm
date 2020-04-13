from .div import Div


class Facade(Div):
    """
    Represents a building facade.
    It's typically composed of one or more faces (in the most cases rectangular ones)
    """
    
    def __init__(self):
        super().__init__()
        self.outer = True
    
    def init(self):
        super().init()
        if not self.outer:
            self.outer = True

    @classmethod
    def getItem(cls, volumeGenerator, parent, indices):
        item = volumeGenerator.itemFactory.getItem(cls)
        item.init()
        item.parent = parent
        item.footprint = parent
        item.building = parent.building
        item.indices = indices
        # <volumeGenerator> knows which geometry the facade items have and how to map UV-coordinates
        volumeGenerator.initFacadeItem(item)
        return item
    
    @property
    def front(self):
        return True
    
    @property
    def back(self):
        return True