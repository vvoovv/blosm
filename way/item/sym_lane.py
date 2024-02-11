from .item import Item


class SymLane(Item):
    
    ID = 0
    
    def __init__(self, location):
        super().__init__()
        self.id = SymLane.ID
        SymLane.ID += 1
        self._location = location

    @property
    def location(self):
        return self._location
