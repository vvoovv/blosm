from . import RoofRealistic
from building.roof.hipped import RoofHipped
from .flat import RoofFlatRealistic, RoofFlat
from .profile import RoofProfileRealistic, RoofProfile


class RoofHippedRealistic(RoofRealistic, RoofHipped):
    
    def renderRoofTextured(self):
        if self.makeFlat:
            return RoofFlatRealistic.renderRoofTextured(self)
        else:
            super().renderRoofTextured()
    
    def renderWalls(self):
        if self.makeFlat:
            if self.mrw:
                RoofFlatRealistic.renderWalls(self)
            else:
                RoofFlat.renderWalls(self)
        else:
            if self.mrw:
                RoofProfileRealistic.renderWalls(self)
            else:
                RoofProfile.renderWalls(self)