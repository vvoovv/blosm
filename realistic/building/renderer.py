from building.renderer import BuildingRenderer
from realistic.building.layer import BuildingLayer

from .roof.flat import RoofFlatRealistic


class RealisticBuildingRenderer(BuildingRenderer):
    
    def __init__(self, app, layerId, **kwargs):
        self.bldgPreRender = None
        super().__init__(app, layerId, BuildingLayer)
        for k in kwargs:
            setattr(self, k, kwargs[k])
        # a Python dictionary of material renderers
        self.materialRenderers = {}
        # a Python set of names of processed material groups;
        # it maybe used by several material renderers
        self.materialGroups = set()
    
    def getMaterialRenderer(self, constructor):
        name = constructor.__name__
        # <mr> stands for "material renderer"
        mr = self.materialRenderers.get(name)
        if not mr:
            mr = constructor(self)
            self.materialRenderers[name] = mr
        return mr
    
    def initRoofs(self):
        super().initRoofs()
        self.roofs['flat'] = RoofFlatRealistic()