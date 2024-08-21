from . import Way, Railway
from defs.way import allWayCategories, facadeVisibilityWayCategories, wayIntersectionCategories, vehicleRoadsCategories
from style import StyleStore
from way.waymap.waymap import WayMap
from way.item.street import Street


class WayManager:
    
    def __init__(self, data, app, getStyle):
        self.id = "ways"
        self.data = data
        self.app = app
        self.styleStore = StyleStore(app.pmlFilepathStreet, app.assetsDir, styles=None)
        self.getStyle = getStyle
        
        # no layers for this manager
        self.layerClass = None
        
        # don't accept broken multipolygons
        self.acceptBroken = False
        
        self.layers = dict((category, []) for category in allWayCategories)
        
        self.renderers = []
        
        self.actions = []
        
        # <self.intersections>, <self.wayClusters> and <self.waySectionLines>
        # are used for realistic rendering of streets
        self.majorIntersections  = []
        self.minorIntersections  = []
        self.transitionSymLanes = []
        self.transitionSideLanes = []
        self.streets = []
        # street sections and clusters
        self.waymap = WayMap()
        self.waySectionLines = dict()
        self.wayClusters = dict()
        
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

    def getAllVehicleWays(self):
        return (
            way for category in vehicleRoadsCategories for way in self.layers[category] \
            if not way.bridge and not way.tunnel
        )
    
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
    
    def iterStreets(self):
        return iter(self.streets)

    def iterStreetsFromWaymap(self):
        def findMinorNodes(street):
            srcIsect = self.waymap.getMinorNode(street.src)
            if srcIsect:
                srcIsect = srcIsect if srcIsect.leaving == street or srcIsect.arriving == street else None
            dstIsect = self.waymap.getMinorNode(street.dst)
            if dstIsect:
                dstIsect = dstIsect if dstIsect.leaving == street or dstIsect.arriving == street else None
            return srcIsect, dstIsect
        
        processedStreets = set()
        for src, dst, key, street in self.waymap.edges(data='object',keys=True):
            if street in processedStreets:
                continue

            srcIsectInit, dstIsectInit = findMinorNodes(street)
            hasMinors = srcIsectInit or dstIsectInit

            if not hasMinors:
                processedStreets.add(street)
                streetStyle = self.styleStore.get( self.getStyle(street) )
                street.style = streetStyle
                street.setStyleBlockFromTop(streetStyle)
                yield street
            else:
                # Create a new Street
                longStreet = Street(street.src,street.dst)
                longStreet.insertStreetEnd(street)

                if dstIsectInit:    # Minor intersection at the end of this street
                    dstIsectInit.street = longStreet
                    longStreet.insertEnd(dstIsectInit)   # insert minor intersection object
                    dstIsectCurr = dstIsectInit
                    while True:
                        nextStreet = dstIsectCurr.leaving
                        if nextStreet in processedStreets:
                            break
                        longStreet.insertStreetEnd(nextStreet)
                        processedStreets.add(nextStreet)
                        _, dstIsectCurr = findMinorNodes(nextStreet)
                        if not dstIsectCurr:
                            break
                        dstIsectCurr.street = longStreet
                        longStreet.insertEnd(dstIsectCurr) 
 
                if srcIsectInit:        # Minor intersection at the front of this street
                    srcIsectInit.street = longStreet
                    longStreet.insertFront(srcIsectInit)   # insert minor intersection object
                    srcIsectCurr = srcIsectInit
                    while True:
                        prevStreet = srcIsectCurr.arriving
                        if prevStreet in processedStreets:
                            break
                        longStreet.insertStreetFront(prevStreet)
                        processedStreets.add(prevStreet)
                        srcIsectCurr, _ = findMinorNodes(prevStreet)
                        if not srcIsectCurr:
                            break
                        srcIsectCurr.street = longStreet
                        longStreet.insertFront(srcIsectCurr)   # insert minor intersection object

                streetStyle = self.styleStore.get( self.getStyle(longStreet) )
                longStreet.style = streetStyle
                longStreet.setStyleBlockFromTop(streetStyle)

                yield longStreet

class RailwayManager:
    """
    An auxiliary manager to process railways
    """
    
    def __init__(self, wayManager):
        self.layers = wayManager.layers
        # no layers for this manager
        self.layerClass = None
    
    def parseWay(self, element, elementId):
        # create a wrapper for the OSM way <element>
        way = Railway(element, self)
        self.layers[way.category].append(way)
    
    def parseRelation(self, element, elementId):
        return