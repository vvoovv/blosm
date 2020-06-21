from .div import Div


class Facade(Div):
    """
    Represents a building facade.
    It's typically composed of one or more faces (in the most cases rectangular ones)
    """
    
    def __init__(self):
        super().__init__()
        
        self.buildingPart = "facade"
        
        self.outer = True
        
        self.front = False
        self.back = False
        self.side = False
        
        self.numEntrances = 0
    
    def init(self):
        super().init()
        
        if not self.outer:
            self.outer = True
        
        if self.front:
            self.front = False
        elif self.back:
            self.back = False
        elif self.side:
            self.side = False
        
        if self.numEntrances:
            self.numEntrances = 0

    @classmethod
    def getItem(cls, volumeGenerator, parent, indices, edgeIndex):
        item = volumeGenerator.itemFactory.getItem(cls)
        item.init()
        item.parent = parent
        item.footprint = parent
        item.building = parent.building
        item.indices = indices
        item.edgeIndex = edgeIndex
        # <volumeGenerator> knows which geometry the facade items have and how to map UV-coordinates
        volumeGenerator.initFacadeItem(item)
        return item