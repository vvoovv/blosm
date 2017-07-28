from building.renderer import *
from realistic.building.layer import BuildingLayer

from .roof.flat import RoofFlatRealistic
from .roof.profile import RoofProfileRealistic


class RealisticBuildingRenderer(BuildingRenderer):
    
    def __init__(self, app, layerId, **kwargs):
        self.bldgPreRender = None
        super().__init__(app, layerId, BuildingLayer)
        for k in kwargs:
            setattr(self, k, kwargs[k])
        # A Python dictionary for mapping between material names and material renderers;
        # it's normally set via <kwargs>
        if not self.materials:
            self.materials = {}
        # a Python dictionary of material renderers
        self.materialRenderers = {}
        # a Python set of names of processed material groups;
        # it maybe used by several material renderers
        self.materialGroups = set()
    
    def getMaterialRenderer(self, name):
        # <mr> stands for "material renderer"
        mr = self.materialRenderers.get(name)
        if not mr:
            constructor = self.materials.get(name)
            if constructor:
                mr = constructor(self, name)
                self.materialRenderers[name] = mr
        return mr
    
    def initRoofs(self):
        super().initRoofs()
        self.roofs['flat'] = RoofFlatRealistic()
        self.roofs['gabled'] = RoofProfileRealistic(gabledRoof)
        self.roofs['round'] = RoofProfileRealistic(roundRoof)
        self.roofs['gambrel'] = RoofProfileRealistic(gambrelRoof)
        self.roofs['saltbox'] = RoofProfileRealistic(saltboxRoof)
        self.roofs['mansard'] = RoofProfileRealistic(gabledRoof)