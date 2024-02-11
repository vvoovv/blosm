from .item import Item


class Crosswalk(Item):
    
    ID = 0
    
    def __init__(self, location):
        super().__init__()
        self.id = Crosswalk.ID
        Crosswalk.ID += 1
        self._location = location

    @property
    def location(self):
        return self._location
