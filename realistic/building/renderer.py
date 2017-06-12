from building.renderer import BuildingRenderer
from realistic.building.layer import BuildingLayer

from .roof.flat import RoofFlatRealistic


class RealisticBuildingRenderer(BuildingRenderer):
    
    def __init__(self, app, layerId):
        super().__init__(app, layerId, BuildingLayer)
        
        self.roofs['flat'] = RoofFlatRealistic()
        # set renderer for each roof
        for r in self.roofs:
            self.roofs[r].r = self