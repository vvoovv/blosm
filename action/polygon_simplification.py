

class PolygonSimplification:
    
    def do(self, manager):

        for building in manager.buildings:
            polygon = building.polygon
            