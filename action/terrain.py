from . import Action
import parse

from util import zAxis


class Terrain(Action):
    
    def preprocess(self, buildingsP):
        # <buildingsP> means "buildings from the parser"
        self.app.terrain.initProjectionProxy(buildingsP, self.data)
    
    def do(self, building, style, globalRenderer):
        #self.projectSingleVertex(building)
        self.projectAllVertices(building)
    
    def projectAllVertices(self, building):
        element = building.element
        
        maxZ = max(
            (
                self.app.terrain.project2(vert)\
                for vert in\
                (element.getOuterData(self.data) if element.t is parse.multipolygon else element.getData(self.data))
            ),
            key = lambda vert: vert[2]
        )[2]
        
        if maxZ == self.app.terrain.projectLocation:
            # the building is outside the terrain so skip the whole building
            self.skipBuilding()
            return
        
        # we use the lowest z-coordinate among the footprint vertices projected on the terrain as <offsetZ>
        offsetZ = min(
            (
                self.app.terrain.project2(vert)\
                for vert in\
                (element.getOuterData(self.data) if element.t is parse.multipolygon else element.getData(self.data))
            ),
            key = lambda vert: vert[2]
        )
        building.renderInfo.offsetVertex = offsetZ[2] * zAxis
        # we also need to store the altitude difference for the building footprint
        building.renderInfo.altitudeDifference = maxZ - offsetZ[2]
    
    def projectSingleVertex(self, building):
        element = building.element
        # take the first vertex of the outline as the offset
        offsetZ = self.app.terrain.project(
            next( element.getOuterData(self.data) if element.t is parse.multipolygon else element.getData(self.data) )
        )
        if offsetZ:
            building.offset = offsetZ[2] * zAxis
        else:
            # the building is outside the terrain so skip the whole building
            self.skipBuilding()
    
    def skipBuilding(self):
        self.itemStore.skip = True
    
    def cleanup(self):
        self.app.terrain.cleanupProjectionProxy()