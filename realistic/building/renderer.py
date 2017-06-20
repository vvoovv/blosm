from building.renderer import BuildingRenderer
from realistic.building.layer import BuildingLayer

from .roof.flat import RoofFlatRealistic


class RealisticBuildingRenderer(BuildingRenderer):
    
    def __init__(self, app, layerId, **kwargs):
        self.bldgPreRender = None
        super().__init__(app, layerId, BuildingLayer)
        for k in kwargs:
            setattr(self, k, kwargs[k])
        # create a dictionary for material managers
        self.materialManagers = {}
    
    def getMaterialManager(self, constructor):
        name = constructor.__name__
        mm = self.materialManagers.get(name)
        if not mm:
            mm = constructor(self)
            self.materialManagers[name] = mm
        return mm
    
    def initRoofs(self):
        super().initRoofs()
        self.roofs['flat'] = RoofFlatRealistic()