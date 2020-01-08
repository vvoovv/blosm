from .item_renderer import ItemRenderer
from .container import Container
from ..facade import Facade as FacadeBase
from ..div import Div as DivBase
from ..level import Level as LevelBase
from ..basement import Basement as BasementBase
from .door import Door

from ..roof_flat import RoofFlat as RoofFlatBase
from ..roof_pyramidal import RoofPyramidal as RoofPyramidalBase


class Facade(FacadeBase, Container):
    
    def __init__(self):
        # a reference to the Container class used in the parent classes
        self.Container = Container
        Container.__init__(self)


class Div(DivBase, Container):
    
    def __init__(self):
        # a reference to the Container class used in the parent classes
        self.Container = Container
        Container.__init__(self)


class Level(LevelBase, Container):
    
    def __init__(self):
        # a reference to the Container class used in the parent classes
        self.Container = Container
        Container.__init__(self)
        LevelBase.__init__(self)


class Basement(BasementBase, Container):
    
    def __init__(self):
        # a reference to the Container class used in the parent classes
        self.Container = Container
        Container.__init__(self)
        BasementBase.__init__(self)


class RoofFlat(RoofFlatBase, ItemRenderer):
    pass


class RoofPyramidal(RoofPyramidalBase):
    pass