from . import RealWay, allWayCategories


facadeVisibilityWayCategories = set((
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    "service",
    "pedestrian",
    "track"
    #"footway",
    #"steps",
    #"cycleway"
))


class RealWayManager:
    
    def __init__(self, data, app):
        self.id = "ways"
        self.data = data
        self.app = app
        
        # use the default layer class in the <app>
        self.layerClass = None
        
        # don't accept broken multipolygons
        self.acceptBroken = False
        
        self.layers = dict((category, []) for category in allWayCategories)
        
        self.actions = []
        
        app.addManager(self)

    def parseWay(self, element, elementId):
        self.createRealWay(element)
    
    def parseRelation(self, element, elementId):
        return
    
    def createRealWay(self, element):
        # create a wrapper for the OSM way <element>
        way = RealWay(element)
        self.layers[way.category].append(way)
    
    def getAllWays(self):
        return (way for category in allWayCategories for way in self.layers[category])
    
    def getFacadeVisibilityWays(self):
        return (way for category in facadeVisibilityWayCategories for way in self.layers[category])
    
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