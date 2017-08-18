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
        """
        The override of the parent class method
        """
        self.flatRoofMulti = RoofFlatMulti()
        self.roofs = {
            'flat': RoofFlatRealistic(),
            'gabled': RoofProfileRealistic(gabledRoof),
            'pyramidal': RoofPyramidalRealistic(),
            'skillion': RoofSkillionRealistic(),
            'hipped': RoofHippedRealistic(),
            'dome': RoofMeshRealistic("roof_dome"),
            'onion': RoofMeshRealistic("roof_onion"),
            'round': RoofProfileRealistic(roundRoof),
            'half-hipped': RoofHalfHippedRealistic(),
            'gambrel': RoofProfileRealistic(gambrelRoof),
            'saltbox': RoofProfileRealistic(saltboxRoof),
            'mansard': RoofMansardRealistic()
        }