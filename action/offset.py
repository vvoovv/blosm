from mathutils import Vector
from . import Action
from renderer import Renderer


class Offset(Action):
    """
    Calculates offset for a building if the option "Import as a single object" is NOT activated.
    Also creates a Blender object for the building in question and positions it appropriately.
    """

    def do(self, building, itemClass, style, globalRenderer):
        outline = building.outline
        offset = Vector(
            next(
                outline.getOuterData(self.data) if outline.t is Renderer.multipolygon else outline.getData(self.data)
            )
        )
        
        layer = outline.l
        globalRenderer.obj = globalRenderer.createBlenderObject(
            globalRenderer.getName(outline),
            offset+building.offset if building.offset else offset,
            collection = layer.getCollection(globalRenderer.collection),
            parent = layer.getParent( layer.getCollection(globalRenderer.collection) )
        )
        layer.prepare(globalRenderer)
        
        building.offset = -offset