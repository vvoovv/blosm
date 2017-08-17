from building.renderer import *
from realistic.building.layer import BuildingLayer

from .roof.flat import RoofFlatRealistic
from .roof.profile import RoofProfileRealistic
from .roof.pyramidal import RoofPyramidalRealistic
from .roof.skillion import RoofSkillionRealistic
from .roof.hipped import RoofHippedRealistic
from .roof.half_hipped import RoofHalfHippedRealistic
from .roof.mansard import RoofMansardRealistic
from .roof.mesh import RoofMeshRealistic


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
        self.roofs['pyramidal'] = RoofPyramidalRealistic()
        self.roofs['skillion'] = RoofSkillionRealistic()
        self.roofs['hipped'] = RoofHippedRealistic(gabledRoof)
        self.roofs['round'] = RoofProfileRealistic(roundRoof)
        self.roofs['half-hipped'] = RoofHalfHippedRealistic()
        self.roofs['gambrel'] = RoofProfileRealistic(gambrelRoof)
        self.roofs['saltbox'] = RoofProfileRealistic(saltboxRoof)
        self.roofs['mansard'] = RoofMansardRealistic(gabledRoof)
        self.roofs['dome'] = RoofMeshRealistic("roof_dome")
        self.roofs['onion'] = RoofMeshRealistic("roof_onion")