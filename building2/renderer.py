from building.renderer import Renderer
from .item_factory import ItemFactory

from item.footprint import Footprint
from item.facade import Facade
from item.level import Level
from item.div import Div
from item.roof import Roof
from item.roof_side import RoofSide
from item.window import Window
from item.door import Door
from item.balcony import Balcony
from item.chimney import Chimney


def _createReferenceItems():
    return (
        Building(),
        Footprint(),
        Facade(),
        Level(),
        Div(),
        Roof(),
        RoofSide(),
        Window(),
        Door(),
        Balcony(),
        Chimney()
    )


class Building:
    """
    A class representing the building for the renderer
    """
    
    def __init__(self):
        self.verts = []
        self.faces = []
        self.uvs = []


class BuildingRendererNew(Renderer):
    
    def __init__(self, app):
        self.itemFactory = ItemFactory(_createReferenceItems(self))
    
    def render(self, building, data):
        