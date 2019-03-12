import math
from . import Action
from renderer import Renderer

#while itemStore.hasItems(itemClass):
#    item = itemStore.getItem(itemClass)

class Terrain(Action):

    def do(self, building, itemClass):
        if self.app.terrain:
            self.projectSingleVertex(building)
    
    def projectSingleVertex(self, building):
        outline = building.outline
        # take the first vertex of the outline as the offset
        offset = self.app.terrain.project(
            next( outline.getOuterData(self.data) if outline.t is Renderer.multipolygon else outline.getData(self.data) )
        )
        if offset:
            building.offsetZ = offset
        else:
            # the building is outside the terrain so skip the whole building
            self.itemStore.skip = True
    
    def projectAllVertices(self, building):
        outline = building.outline
        coords = outline.getOuterData(self.data) if outline.t is Renderer.multipolygon else outline.getData(self.data)
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