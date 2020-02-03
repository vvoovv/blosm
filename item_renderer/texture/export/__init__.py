from .item_renderer import ItemRenderer
from .container import Container
from ..facade import Facade as FacadeBase
from ..div import Div as DivBase
from ..level import Level as LevelBase
from ..basement import Basement as BasementBase
from .door import Door

from ..roof_flat import RoofFlat as RoofFlatBase
from ..roof_generatrix import RoofGeneratrix as RoofGeneratrixBase
from ..roof_pyramidal import RoofPyramidal as RoofPyramidalBase
from ..roof_profile import RoofProfile as RoofProfileBase


class Facade(FacadeBase, Container):
    
    def __init__(self):
        # a reference to the Container class used in the parent classes
        self.Container = Container
        Container.__init__(self, exportMaterials=True)


class Div(DivBase, Container):
    
    def __init__(self):
        # a reference to the Container class used in the parent classes
        self.Container = Container
        Container.__init__(self, exportMaterials=True)


class Level(LevelBase, Container):
    
    def __init__(self):
        # a reference to the Container class used in the parent classes
        self.Container = Container
        Container.__init__(self, exportMaterials=True)
        LevelBase.__init__(self)


class Basement(BasementBase, Container):
    
    def __init__(self):
        # a reference to the Container class used in the parent classes
        self.Container = Container
        Container.__init__(self, exportMaterials=True)
        BasementBase.__init__(self)


class RoofFlat(RoofFlatBase, ItemRenderer):
    
    def __init__(self):
        super().__init__(exportMaterials=True)


class RoofGeneratrix(RoofGeneratrixBase, ItemRenderer):
    
    def __init__(self, generatrix):
        super().__init__(generatrix, exportMaterials=True)


class RoofPyramidal(RoofPyramidalBase):
    pass


class RoofProfile(RoofProfileBase, ItemRenderer):
    
    def __init__(self, generatrix):
        super().__init__(exportMaterials=True)