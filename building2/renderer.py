from building.renderer import Renderer
from .item_store import ItemStore
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
        (Building(), 5),
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
        referenceItems = _createReferenceItems()
        self.itemStore = ItemStore(referenceItems)
        self.itemFactory = ItemFactory(referenceItems)
    
    def render(self, buildingP, data):
        parts = buildingP.parts
        itemFactory = self.itemFactory
        # <buildingP> means "building from the parser"
        building = itemFactory.getItem(Building)
        partTag = buildingP.outline.tags.get("building:part")
        if not parts or (partTag and partTag != "no"):
            # the building has no parts
            footprint = itemFactory.getItem(Footprint, building)
        if parts:
            for part in parts:
                self.renderElement(part, building, osm)