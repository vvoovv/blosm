from .item_renderer import ItemRendererMixin
from .container import Container
from ..facade import Facade as FacadeBase
from ..div import Div as DivBase
from ..level import Level as LevelBase
from ..bottom import Bottom as BottomBase
from .door import Door
from .level import CurtainWall

from ..roof_flat import RoofFlat as RoofFlatBase
from .roof_flat_multi import RoofFlatMulti
from ..roof_generatrix import RoofGeneratrix as RoofGeneratrixBase
from ..roof_pyramidal import RoofPyramidal as RoofPyramidalBase
from ..roof_profile import RoofProfile as RoofProfileBase


class Facade(FacadeBase, Container):
    
    def __init__(self):
        # a reference to the Container class used in the parent classes
        self.Container = Container
        Container.__init__(self, exportMaterials=False)


class Div(DivBase, Container):
    
    def __init__(self):
        # a reference to the Container class used in the parent classes
        self.Container = Container
        Container.__init__(self, exportMaterials=False)


class Level(LevelBase, Container):
    
    def __init__(self):
        # a reference to the Container class used in the parent classes
        self.Container = Container
        Container.__init__(self, exportMaterials=False)
        LevelBase.__init__(self)


class Bottom(BottomBase, Container):
    
    def __init__(self):
        # a reference to the Container class used in the parent classes
        self.Container = Container
        Container.__init__(self, exportMaterials=False)
        BottomBase.__init__(self)


class RoofFlat(RoofFlatBase, ItemRendererMixin):
    pass


class RoofGeneratrix(RoofGeneratrixBase, ItemRendererMixin):
    pass


class RoofPyramidal(RoofPyramidalBase, ItemRendererMixin):
    pass


class RoofProfile(RoofProfileBase, ItemRendererMixin):
    pass