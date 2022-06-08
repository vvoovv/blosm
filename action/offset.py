from mathutils import Vector
from . import Action
import parse


class Offset(Action):
    """
    Calculates offset for a building if the option "Import as a single object" is NOT activated.
    Also creates a Blender object for the building in question and positions it appropriately.
    """
    
    def preprocess(self, buildingsP):
        # <buildingsP> means "buildings from the parser"
        return

    def do(self, building, style, globalRenderer):
        element = building.element
        offset = next(
            element.getOuterData(self.data) if element.t is parse.multipolygon else element.getData(self.data)
        )
        offset = Vector( (offset[0], offset[1], 0.) )
        
        layer = element.l
        layer.obj = globalRenderer.createBlenderObject(
            globalRenderer.getName(element),
            offset+building.renderInfo.offset if building.renderInfo.offset else offset,
            collection = layer.getCollection(globalRenderer.collection),
            parent = layer.getParent( layer.getCollection(globalRenderer.collection) )
        )
        layer.prepare()
        
        building.renderInfo.offset = -offset