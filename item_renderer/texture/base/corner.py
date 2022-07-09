from .container import Container
from ..corner import Corner as CornerBase


class Corner(CornerBase, Container):
    
    def __init__(self):
        # a reference to the Container class used in the parent classes
        self.Container = Container
        Container.__init__(self, exportMaterials=False)
        CornerBase.__init__(self)