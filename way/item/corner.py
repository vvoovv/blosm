class Corner(object):
    ID = 0
    def __init__(self, location):
        self.id = Corner.ID
        Corner.ID += 1
        self._location = location

    @property
    def location(self):
        return self._location