from building.manager import BuildingManager
from realistic.building.layer import RealisticBuildingLayer


class RealisticBuildingManager(BuildingManager):
    
    def __init__(self, osm, buildingParts):
        super().__init__(osm, buildingParts)
        self.layerConstructor = RealisticBuildingLayer