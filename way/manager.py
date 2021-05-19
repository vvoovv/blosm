from . import Way
from defs.way import allWayCategories, facadeVisibilityWayCategories


class WayManager:
    
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
        self.createWay(element)
    
    def parseRelation(self, element, elementId):
        return
    
    def createWay(self, element):
        # create a wrapper for the OSM way <element>
        way = Way(element, self)
        self.layers[way.category].append(way)
    
    def getAllWays(self):
        return (way for category in allWayCategories for way in self.layers[category])
    
    def getFacadeVisibilityWays(self):
        return (
            way for category in facadeVisibilityWayCategories for way in self.layers[category] \
            if not way.bridge and not way.tunnel
        )
    
    def process(self):
        for way in self.getAllWays():
            # <osm.projection> may not be availabe in the constructor of a <Way>, but it's needed to
            # create way-segments. That's why an additional <init(..)> method is needed.
            way.init(self)
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