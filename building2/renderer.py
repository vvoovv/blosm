from building.renderer import Renderer
from .item_store import ItemStore
from .item_factory import ItemFactory

from item.footprint import Footprint
from item.facade import Facade
from item.level import Level
from item.div import Div
from item.basement import Basement
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
        Basement(),
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
        # counterparts for <self.verts> in the BMesh
        self.bmVerts = []
        self._styleBlockCache = {}
    
    def init(self, outline):
        self.verts.clear()
        self.bmVerts.clear()
        # <outline> is an instance of the class as defined by the data model (e.g. parse.osm.way.Way) 
        self.outline = outline
        self.offsetZ = None
        # Instance of item.footprint.Footprint, it's only used if the building definition
        # in the data model doesn't contain building parts, i.e. the building is defined completely
        # by its outline
        self.footprint = None
        self._styleBlockCache.clear()
    
    def clone(self):
        building = Building()
        return building
    
    @classmethod
    def getItem(cls, itemFactory, outline):
        item = itemFactory.getItem(cls)
        item.init(outline)
        return item


class BuildingRendererNew(Renderer):
    
    def __init__(self, app, styleStore, getStyle=None):
        self.app = app
        self.styleStore = styleStore
        self.getStyle = getStyle
        referenceItems = _createReferenceItems()
        self.itemStore = ItemStore(referenceItems)
        self.itemFactory = ItemFactory(referenceItems)
    
    def render(self, buildingP, data):
        parts = buildingP.parts
        itemFactory = self.itemFactory
        itemStore = self.itemStore
        
        # get the style of the building
        style = self.styleStore.get(self.getStyle(buildingP, self.app))
        
        # <buildingP> means "building from the parser"
        outline = buildingP.outline
        
        self.preRender(outline)
        
        building = Building.getItem(itemFactory, outline)
        partTag = outline.tags.get("building:part")
        if not parts or (partTag and partTag != "no"):
            # the building has no parts
            footprint = Footprint.getItem(itemFactory, outline, building)
            # this attribute <footprint> below may be used in <action.terrain.Terrain>
            building.footprint = footprint
            itemStore.add(footprint)
        if parts:
            itemStore.add((Footprint.getItem(itemFactory, part, building) for part in parts), Footprint, len(parts))
        
        for itemClass in (Building, Footprint):
            for action in itemClass.actions:
                action.do(building, itemClass, style)
                if itemStore.skip:
                    break
            if itemStore.skip:
                itemStore.skip = False
                break
        itemStore.clear()
        
        self.postRender(outline)
    
    def createFace(self, building, indices, uvs):
        bm = self.bm
        verts = building.verts
        bmVerts = building.bmVerts
        
        # extend <bmVerts> to have the same number of vertices as in <verts>
        bmVerts.extend(None for _ in range(len(verts)-len(bmVerts)))
        
        # check if we have BMVerts for for all <indices>
        for index in indices:
            if not bmVerts[index]:
                bmVerts[index] = bm.verts.new( verts[index] )
        
        face = bm.faces.new(bmVerts[index] for index in indices)
        if uvs:
            # assign uv coordinates
            uvLayer = bm.loops.layers.uv[0]
            loops = face.loops
            for loop,uv in zip(loops, uvs):
                loop[uvLayer].uv = uv