from .div import Div
from defs.facade_classification import FacadeClass


class Facade(Div):
    """
    Represents a building facade.
    It's typically composed of one or more faces (in the most cases rectangular ones)
    """
    
    def __init__(self, parent, indices, vector, volumeGenerator):
        super().__init__(parent, parent, self, None)
        
        self.cornerL = True
        self.cornerR = True
        
        # We store in the variables below information about corner items. The variables are set to
        # a Python set. Z-coordinate of a corner item is used as the key. The variables below prevent
        # from creating a corner item twice for each of the adjacent facades.
        self.cornerInfoL = None
        self.cornerInfoR = None
        
        self.indices = indices
        
        self.buildingPart = "facade"
        
        # Is facade visible or hidden by other facades that belong to other buildings or building parts?
        self.visible = True
        
        self.vector = vector
        vector.facade = self
        
        self.outer = True
        
        # is facade processed in the action <FacadeBoolean>?
        self.processed = False
        
        # set facade class
        self.cl = None
        self.front = False
        self.side = False
        self.back = False
        self.shared = False
        # If <vector> lies on the building footprint, its edge may hold the facade class
        cl = vector.edge.cl
        if cl:
            self.setClass(cl)
        
        # <volumeGenerator> knows which geometry the facade items have and how to map UV-coordinates
        volumeGenerator.initFacadeItem(self)
    
    def setClass(self, cl):
        self.cl = cl
        if cl == FacadeClass.front:
            self.front = True
        elif cl == FacadeClass.side:
            self.side = True
        elif cl == FacadeClass.back:
            self.back = True
        elif cl == FacadeClass.shared:
            self.shared = True
    
    @property
    def left(self):
        """
        The neighbor of the facade from the left
        """
        return self.vector.prev.facade
    
    @property
    def right(self):
        """
        The neighbor of the facade from the right
        """
        return self.vector.next.facade
    
    @property
    def gable(self):
        # just a simple check of the number of the vertices
        return len(self.indices) > 4 or self.footprint.noWalls