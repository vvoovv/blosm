class Intersection(object):
    ID = 0
    def __init__(self, location):
        self.id = Intersection.ID
        Intersection.ID += 1
        self._location = location

    @property
    def location(self):
        return self._location