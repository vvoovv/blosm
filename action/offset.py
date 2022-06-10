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
        renderInfo = building.renderInfo
        
        offset = next(
            element.getOuterData(self.data) if element.t is parse.multipolygon else element.getData(self.data)
        )
        # <renderInfo.offsetVertex> could have been set in <action.terrain>
        renderInfo.offsetBlenderObject = Vector( (offset[0], offset[1], renderInfo.offsetVertex[2]) ) if renderInfo.offsetVertex else Vector( (offset[0], offset[1], 0.) )
        
        renderInfo.offsetVertex = Vector( (-offset[0], -offset[1], 0.) )