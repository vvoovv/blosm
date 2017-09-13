from . import RoofRealistic
from building.roof.hipped import RoofHipped
from .flat import RoofFlatRealistic
from .profile import RoofProfileRealistic


class RoofHippedRealistic(RoofRealistic, RoofHipped):
    
    def renderRoofTextured(self):
        if self.makeFlat:
            return RoofFlatRealistic.renderRoofTextured(self)
        else:
            super().renderRoofTextured()
    
    def renderWalls(self):
        if self.makeFlat:
            RoofFlatRealistic.renderWalls(self)
        else:
            RoofProfileRealistic.renderWalls(self)