

class Building:
    """
    A wrapper for a building
    """
    def __init__(self, element):
        self.outline = element
        self.parts = []


class Footprint:
    
    def __init__(self, element):
        self.element = element