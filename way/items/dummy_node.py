from mathutils import Vector

class DummyNode(object):
    ID = 0
    def __init__(self):
        self.id = DummyNode.ID
        DummyNode.ID += 1
        self._location = Vector((0,0))

    @property
    def location(self):
        return self._location
