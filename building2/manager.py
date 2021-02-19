from building.manager import BuildingManager
from .renderer import Building


class RealisticBuildingManager(BuildingManager):
    
    def process(self):
        super().process()
        for action in Building.actions:
            action.preprocess(self.buildings)