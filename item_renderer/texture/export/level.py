import os
import bpy
from util.blender_extra.material import createMaterialFromTemplate, setImage

from .container import Container
from ..level import CurtainWall as CurtainWallBase


class CurtainWall(CurtainWallBase, Container):
    
    def __init__(self):
        # a reference to the Container class used in the parent classes
        self.Container = Container
        Container.__init__(self, exportMaterials=True)
        CurtainWallBase.__init__(self)