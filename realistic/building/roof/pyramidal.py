from . import RoofRealistic
from building.roof.pyramidal import RoofPyramidal
from .flat import RoofFlatRealistic


class RoofPyramidalRealistic(RoofRealistic, RoofPyramidal):
    
    def renderWalls(self):
        RoofFlatRealistic.renderWalls(self)