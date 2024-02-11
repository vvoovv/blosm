from mathutils import Vector
from .item import Item


class DummyNode(Item):
    
    ID = 0
    
    def __init__(self):
        super().__init__()
        self.id = DummyNode.ID
        DummyNode.ID += 1
        self._location = Vector((0,0))

    @property
    def location(self):
        return self._location
