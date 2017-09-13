from . import RoofRealistic
from building.roof.half_hipped import RoofHalfHipped
from .profile import RoofProfileRealistic


class RoofHalfHippedRealistic(RoofRealistic, RoofHalfHipped):
    
    def renderWalls(self):
        RoofProfileRealistic.renderWalls(self)