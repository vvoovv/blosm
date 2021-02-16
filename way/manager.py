from . import RealWay


_allWays = (
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    "service",
    "pedestrian",
    "track",
    "footway",
    "steps",
    "cycleway",
    "bridleway",
    "other"
)


class RealWayManager:
    
    def __init__(self, data, app):
        self.id = "ways"
        self.data = data
        
        self.layers = dict((layerId, []) for layerId in _allWays)
        self.createLayerMapping()
        
        self.app.managers.append(self)
    
    def createLayerMapping(self):
        """
        Create mapping between user or GUI layer names and the layer names from <_allWays>
        """
        self.layerMapping = dict(
        
        )

    def parseWay(self, element, elementId):
        self.createRealWay(element)
    
    def parseRelation(self, element, elementId):
        self.createRealWay(element)
    
    def createRealWay(self, element):
        # create a wrapper for the OSM way <element>
        self.layers[element.l.appId].append( RealWay(element) )
    
    def getAllWays(self):
        return (way in way for layerId in _allWays for way in self.layers[layerId])
    
    def process(self):
        pass
    
    def render(self):
        for way in self.getAllWays():
            self.renderer.render(way, self.osm)