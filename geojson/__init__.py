from manager import Manager as _Manager
from building.manager import BuildingManager as _BuildingManager


class Manager(_Manager):
    
    def parsePolygon(self, feature, featureId):
        # render it in <self.render(..)>
        feature.r = True
    
    def parseMultipolygon(self, feature, featureId):
        # render it in <self.render(..)>
        feature.r = True

    def render(self):
        data = self.data
        
        for polygon in data.polygons:
            if polygon.valid and polygon.r:
                renderer = polygon.rr or self.renderer
                renderer.preRender(polygon)
                renderer.renderPolygon(polygon, data)
                renderer.postRender(polygon)

        for multipolygon in data.multipolygons:
            if multipolygon.valid and multipolygon.r:
                renderer = multipolygon.rr or self.renderer
                renderer.preRender(multipolygon)
                renderer.renderMultiPolygon(multipolygon, data)
                renderer.postRender(multipolygon)
        
        for node in data.nodes:
            renderer = node.rr or self.nodeRenderer
            #renderer.preRender(node)
            renderer.renderNode(node, data)
            #renderer.postRender(node)


class Building:
    """
    A wrapper for a GeoJson building
    """
    def __init__(self, element):
        self.outline = element
        self.parts = []
    
    def addPart(self, part):
        self.parts.append(part)


class BuildingManager(_BuildingManager):
    
    def process(self):
        pass
    
    def parsePolygon(self, feature, featureId):
        # create a wrapper for the GeoJson <feature>
        building = Building(feature)
        # store the related wrapper in the attribute <b>
        feature.b = building
        self.buildings.append(building)
    
    def parseMultipolygon(self, feature, featureId):
        self.parsePolygon(feature, featureId)