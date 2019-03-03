

class Item:
    
    def __init__(self):
        # For example, a parent for a facade is a footprint
        self.parent = None
    
    def clone(self):
        item = self.__class__()
        return item