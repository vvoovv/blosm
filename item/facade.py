from . import Item


class Facade(Item):
    """
    Represents a building facade.
    It's typically composed of one or more faces (in the most cases rectangular ones)
    """
    
    def __init__(self):
        pass
    
    def init(self):
        self.faces = []
        self.normal = None
        self.type = ("front", "back", "side")
        self.neighborL = None
        self.neighborR = None
        self.neighborT = None
        self.neighborB = None