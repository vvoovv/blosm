import math
from . import Action
from renderer import Renderer

from util import zAxis


class Terrain(Action):
    
    def preprocess(self, buildingsP):
        # <buildingsP> means "buildings from the parser"
        self.app.terrain.initProjectionProxy(buildingsP, self.data)
    
    def do(self, building, itemClass, style, globalRenderer):
        #self.projectSingleVertex(building)
        #self.projectAllVertices(building)
        self.testProject(building)
    
    def testProject(self, building):
        outline = building.outline
        # take the first vertex of the outline as the offset
        offsetZ = max(
            (
                self.app.terrain.project2(vert)\
                for vert in\
                (outline.getOuterData(self.data) if outline.t is Renderer.multipolygon else outline.getData(self.data))
            ),
            key = lambda vert: vert[2]
        )
        if offsetZ:
            building.offset = offsetZ[2] * zAxis
        else:
            # the building is outside the terrain so skip the whole building
            self.itemStore.skip = True
    
    def projectSingleVertex(self, building):
        outline = building.outline
        # take the first vertex of the outline as the offset
        offsetZ = self.app.terrain.project(
            next( outline.getOuterData(self.data) if outline.t is Renderer.multipolygon else outline.getData(self.data) )
        )
        if offsetZ:
            building.offset = offsetZ[2] * zAxis
        else:
            # the building is outside the terrain so skip the whole building
            self.itemStore.skip = True
    
    def projectAllVertices(self, building):
        outline = building.outline
        if outline.t is Renderer.multipolygon:
            coords = outline.getOuterData(self.data)
        else:
            coords = outline.getData(self.data)
            if building.footprint:
                # building definition has no building parts
                polygon = building.footprint.polygon
                polygon.init(coords)
                coords = polygon.verts
        basementMin = math.inf
        basementMax = -math.inf
        skip = True
        for coord in coords:
            coord = self.app.terrain.project(coord)
            # If at least one point is on the terrain,
            # keep the whole building for consideration
            if coord:
                if skip:
                    skip = False
                z = coord[2]
                if z < basementMin:
                    basementMin = z
                if z > basementMax:
                    basementMax = z
        if skip:
            # the building is outside the terrain so skip the whole building
            self.itemStore.skip = True
        else:
            building.basementMin = basementMin
            building.basementMax = basementMax