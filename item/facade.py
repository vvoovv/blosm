from .div import Div
from defs.facade_classification import FacadeClass


class Facade(Div):
    """
    Represents a building facade.
    It's typically composed of one or more faces (in the most cases rectangular ones)
    """
    
    def __init__(self, parent, indices, vector, volumeGenerator):
        super().__init__(parent, parent, None)
        
        self.indices = indices
        
        self.buildingPart = "facade"
        
        self.outer = True
        
        # set facade class
        self.front = False
        self.side = False
        self.back = False
        self.shared = False
        facadeClass = vector.edge.cl
        if facadeClass:
            if facadeClass == FacadeClass.front:
                self.front = True
            elif facadeClass == FacadeClass.side:
                self.side = True
            elif facadeClass == FacadeClass.shared:
                self.shared = True
            else:
                self.back = True
        
        # <volumeGenerator> knows which geometry the facade items have and how to map UV-coordinates
        volumeGenerator.initFacadeItem(self)