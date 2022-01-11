from .div import Div


class Facade(Div):
    """
    Represents a building facade.
    It's typically composed of one or more faces (in the most cases rectangular ones)
    """
    
    def __init__(self, parent, indices, edgeIndex, volumeGenerator):
        super().__init__(parent, parent, None)
        
        self.indices = indices
        self.edgeIndex = edgeIndex
        
        self.buildingPart = "facade"
        
        self.outer = True
        
        self.front = False
        self.back = False
        self.side = False
        
        self.numEntrances = 0
        
        # <volumeGenerator> knows which geometry the facade items have and how to map UV-coordinates
        volumeGenerator.initFacadeItem(self)