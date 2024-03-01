from .item import Item


class PtStop(Item):
    
    ID = 0
    
    def __init__(self, element):
        super().__init__()
        self.id = PtStop.ID
        PtStop.ID += 1

        self.street = None

        
        self.element = element
        #self._location = location

    @property
    def location(self):
        return self._location
