from . import RoofRealistic
from building.roof.mesh import RoofMesh
from .flat import RoofFlatRealistic


class RoofMeshRealistic(RoofRealistic, RoofMesh):
    
    def __init__(self, mesh):
        super().__init__(mesh)

    def setMaterial(self, obj, slot):
        mrr = self.mrr
        if mrr:
            mrr.renderForObject(obj, slot)
        else:
            super().setMaterial(obj, slot)

    def renderWalls(self):
        RoofFlatRealistic.renderWalls(self)