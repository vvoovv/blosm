from building.renderer import BuildingRenderer
from realistic.building.layer import BuildingLayer

from .roof.flat import RoofFlatRealistic


class RealisticBuildingRenderer(BuildingRenderer):
    
    def __init__(self, app, layerId):
        super().__init__(app, layerId, BuildingLayer)
    
    def initRoofs(self):
        super().initRoofs()
        self.roofs['flat'] = RoofFlatRealistic()