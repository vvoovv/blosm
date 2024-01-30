

class PtStop:
    
    ID = 0
    
    def __init__(self, element):
        self.id = PtStop.ID
        PtStop.ID += 1
        
        self.element = element
        #self._location = location

    @property
    def location(self):
        return self._location
