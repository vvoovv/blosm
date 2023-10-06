from building.manager import BuildingManager


class RealisticBuildingManager(BuildingManager):
    
    def process(self):
        super().process()
        for action in self.renderer.actions:
            action.preprocess(self.buildings)
    
    def render(self):
        renderer = self.renderer
        
        for building in self.buildings:
            renderer.render(building, self.data)
        
        if not self.app.renderAfterExtrude:
            for building in self.buildings:
                renderer.renderExtrudedVolumes(building, self.data)
    
    def parseRelation(self, element, elementId):
        # FIXME temporarily skip multipolygons
        return