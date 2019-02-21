import math
from . import Action
from renderer import Renderer


class Terrain(Action):

    def do(self, building, data):
        if self.app.terrain:
            self.projectSingleVertex(building, data)
    
    def projectSingleVertex(self, building, data):
        outline = building.outline
        # take the first vertex of the outline as the offset
        offset = self.app.terrain.project(
            next( outline.getOuterData(data) if outline.t is Renderer.multipolygon else outline.getData(data) )
        )
        if offset:
            building.offsetZ = offset
        else:
            self.itemStore.clear()
    
    def projectAllVertices(self, building, data):
        outline = building.outline
        coords = outline.getOuterData(data) if outline.t is Renderer.multipolygon else outline.getData(data)
        basementMin = math.inf
        basementMax = -math.inf
        for coord in coords:
            z = self.app.terrain.project(coord)[2]
            if z < basementMin:
                basementMin = z
            if z > basementMax:
                basementMax = z
        building.basementMin = basementMin
        building.basementMax = basementMax