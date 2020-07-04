from .item_renderer import ItemRendererMixin
from .container import Container
from ..facade import Facade as FacadeBase
from ..div import Div as DivBase
from ..level import Level as LevelBase
from ..bottom import Bottom as BottomBase
from .door import Door
from .level import CurtainWall

from ..roof_flat import RoofFlat as RoofFlatBase
from ..roof_flat_multi import RoofFlatMulti as RoofFlatMultiBase
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


class Bottom(BottomBase, Container):
    
    def __init__(self):
        # a reference to the Container class used in the parent classes
        self.Container = Container
        Container.__init__(self, exportMaterials=True)
        BottomBase.__init__(self)


class RoofFlat(RoofFlatBase, ItemRendererMixin):
    
    def __init__(self):
        super().__init__(exportMaterials=True)


class RoofFlatMulti(RoofFlatMultiBase, ItemRendererMixin):
    
    def __init__(self):
        super().__init__(exportMaterials=True)


class RoofGeneratrix(RoofGeneratrixBase, ItemRendererMixin):
    
    def __init__(self, generatrix, basePointPosition):
        super().__init__(generatrix, basePointPosition, exportMaterials=True)


class RoofPyramidal(RoofPyramidalBase, ItemRendererMixin):
    
    def __init__(self):
        super().__init__(exportMaterials=True)


class RoofProfile(RoofProfileBase, ItemRendererMixin):
    
    def __init__(self):
        super().__init__(exportMaterials=True)