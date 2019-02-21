

class Item:
    
    def __init__(self):
        # For example, a parent for facade is a footprint
        self.parent = None
    
    def clone(self, attr):
        item = self.__class__()
        item.init(attr)
        return item