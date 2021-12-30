from building.manager import BuildingManager
from item.building import Building


class RealisticBuildingManager(BuildingManager):
    
    def process(self):
        super().process()
        for action in Building.actions:
            action.preprocess(self.buildings)