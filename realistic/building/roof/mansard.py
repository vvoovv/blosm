from . import RoofRealistic
from building.roof.mansard import RoofMansard
from .flat import RoofFlatRealistic


class RoofMansardRealistic(RoofRealistic, RoofMansard):
    
    def renderRoofTextured(self):
        if self.makeFlat:
            return RoofFlatRealistic.renderRoofTextured(self)
        else:
            super().renderRoofTextured()