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


_facadeVisibilityWays = (
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    #"service",
    "pedestrian",
    "track",
    #"footway",
    #"steps",
    #"cycleway",
    #"bridleway",
    #"other"
)


class RealWayManager:
    
    def __init__(self, data, app):
        self.id = "ways"
        self.data = data
        self.app = app
        
        # use the default layer class in the <app>
        self.layerClass = None
        
        # don't accept broken multipolygons
        self.acceptBroken = False
        
        self.layers = dict((layerId, []) for layerId in _allWays)
        
        self.actions = []
        
        app.addManager(self)

    def parseWay(self, element, elementId):
        self.createRealWay(element)
    
    def parseRelation(self, element, elementId):
        return
    
    def createRealWay(self, element):
        # create a wrapper for the OSM way <element>
        self.layers[element.l.mlId].append( RealWay(element) )
    
    def getAllWays(self):
        return (way for layerId in _allWays for way in self.layers[layerId])
    
    def getFacadeVisibilityWays(self):
        return (way for layerId in _facadeVisibilityWays for way in self.layers[layerId])
    
    def process(self):
        for action in self.actions:
            action.do(self)
    
    def setRenderer(self, renderer, app):
        self.renderer = renderer
        app.addRenderer(renderer)
    
    def render(self):
        for way in self.getAllWays():
            self.renderer.render(way, self.data)
    
    def addAction(self, action):
        action.app = self.app
        self.actions.append(action)