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

from action.terrain import Terrain


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
    
    def init(self, outline):
        self.outline = outline
        self.offsetZ = None
    
    def clone(self):
        building = Building()
        return building


class BuildingRendererNew(Renderer):
    
    def __init__(self, app, styleStore, getStyle=None, actions=None):
        self.app = app
        self.styleStore = styleStore
        self.getStyle = getStyle
        referenceItems = _createReferenceItems()
        self.itemStore = ItemStore(referenceItems)
        self.itemFactory = ItemFactory(referenceItems)
        if actions:
            self.actions = actions
        else:
            terrainAction = Terrain(app, self.itemStore, self.itemFactory)
            self.actions = [terrainAction]
    
    def render(self, buildingP, data):
        parts = buildingP.parts
        itemFactory = self.itemFactory
        itemStore = self.itemStore
        
        # get the style of the building
        style = self.styleStore.get(self.getStyle(buildingP, self.app))
        
        # <buildingP> means "building from the parser"
        outline = buildingP.outline
        building = itemFactory.getItem(Building, outline)
        partTag = outline.tags.get("building:part")
        if not parts or (partTag and partTag != "no"):
            # the building has no parts
            footprint = itemFactory.getItem(Footprint, outline, building)
            itemStore.add(footprint)
        if parts:
            itemStore.add(itemFactory.getItem(Footprint, part, building) for part in parts)