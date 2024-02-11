from .item import Item


class Corner(Item):
    
    ID = 0
    
    def __init__(self, location):
        super().__init__()
        self.id = Corner.ID
        Corner.ID += 1
        self._location = location

    @property
    def location(self):
        return self._location