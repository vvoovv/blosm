from . import Way, Railway
from defs.base.polyline import Polyline
from defs.way import allWayCategories, facadeVisibilityWayCategories, wayIntersectionCategories


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
        
        self.renderers = []
        
        self.actions = []
        
        # <self.networkGraph>, <self.waySectionGraph> and <self.junctions> are set in an action,
        # for example <action.way_clustering.Way>
        self.networkGraph = self.waySectionGraph = self.junctions = None
        
        app.addManager(self)

    def parseWay(self, element, elementId):
        # create a wrapper for the OSM way <element>
        way = Way(element, self)
        self.layers[way.category].append(way)
    
    def parseRelation(self, element, elementId):
        return
    
    def getAllWays(self):
        return (way for category in allWayCategories for way in self.layers[category])
    
    def getAllIntersectionWays(self):
        return (way for category in wayIntersectionCategories for way in self.layers[category])
    
    def getFacadeVisibilityWays(self):
        return (
            way for category in facadeVisibilityWayCategories for way in self.layers[category] \
            if not way.bridge and not way.tunnel
        )

    def getSelectedWays(self, categories):
        return ( way for category in categories for way in self.layers[category] )
    
    def process(self):
        for way in self.getAllWays():
            # <osm.projection> may not be availabe in the constructor of a <Way>, but it's needed to
            # create way-segments. That's why an additional <init(..)> method is needed.
            way.init(self)
        for action in self.actions:
            action.do(self)
    
    def addRenderer(self, renderer):
        self.renderers.append(renderer)
        self.app.addRenderer(renderer)
    
    def render(self):
        for renderer in self.renderers:
            renderer.render(self, self.data)
    
    def renderExtra(self):
        return
    
    def addAction(self, action):
        action.app = self.app
        self.actions.append(action)
    
    def getRailwayManager(self):
        return RailwayManager(self)

class RailwayManager:
    """
    An auxiliary manager to process railways
    """
    
    def __init__(self, wayManager):
        self.layers = wayManager.layers
        # use the default layer class in the <app>
        self.layerClass = None
    
    def parseWay(self, element, elementId):
        # create a wrapper for the OSM way <element>
        way = Railway(element, self)
        self.layers[way.category].append(way)
    
    def parseRelation(self, element, elementId):
        return


class RoadPolygonsManager:
    
    def __init__(self, data, app):
        self.id = "road_polygons"
        self.data = data
        self.app = app
        
        self.polylines = []
        self.buildings = None
        #self.connectedManagers = []
        self.actions = []
        
        # don't accept broken multipolygons
        self.acceptBroken = False
        
        app.addManager(self)
    
    def addAction(self, action):
        action.app = self.app
        self.actions.append(action)
    
    def process(self):
        # get <self.buildings>
        self.buildings = self.app.managersById.get("buildings") and self.app.managersById.get("buildings").buildings
        
        for polyline in self.polylines:
            polyline.init(self)
            
        #for connectedManager in self.connectedManagers:
        #    self.polylines.extend(connectedManager.getPolylines())
        
        for action in self.actions:
            action.do(self)

    def parseWay(self, element, elementId):
        self.polylines.append(Polyline(element))
    
    def parseRelation(self, element, elementId):
        return
    
    def render(self):
        """
        A temporary function for rendering the results in matplotlib
        """
        ax = self.mpl.ax
        for polyline in self.polylines:
            lineStyle, areaColor = self.getStyles(polyline.element)
            if lineStyle:
                for edge in polyline.edges:
                    ax.plot(
                        (edge.v1[0], edge.v2[0]),
                        (edge.v1[1], edge.v2[1]),
                        **lineStyle
                    )
            if areaColor:
                ax.fill(
                    [ edge.v1[0] for edge in polyline.edges ],
                    [ edge.v1[1] for edge in polyline.edges ],
                    areaColor
                )
    
    def getStyles(self, element):
        """
        A temporary function for rendering the results in matplotlib
        """
        tags = element.tags
        if "building" in tags:
            return ( dict(color="#c2b5aa", linewidth=1.), "#d9d0c9" )
        elif "landuse" in tags:
            landuse = tags["landuse"]
            if landuse == "construction":
                return (None, "#c7c7b4")
            elif landuse == "industrial":
                return (None, "#e6d1e3")
            elif landuse == "forest":
                return (None, "#9dca8a")
            elif landuse == "cemetery":
                return (None, "#aacbaf")
            elif landuse in ("grass", "village_green"):
                return (None, "#c5ec94")
            else:
                return (None, "#a7a87e")
        elif "natural" in tags:
            natural = tags["natural"]
            if natural == "water":
                return (None, "#a6c6c6")
            elif natural == "tree_row":
                return ( dict(color="#a9cea1", linewidth=2.5), None )
            elif natural == "wood":
                return (None, "#9dca8a")
            else:
                return (None, "#d6d99f")
        elif "barrier" in tags:
            return ( dict(color="black", linewidth=0.5), None )