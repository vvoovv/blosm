from itertools import tee,islice, cycle

from .item import Item
from lib.CompGeom.PolyLine import PolyLine
# from lib.CompGeom.offset_intersection import offsetPolylineIntersection
from lib.CompGeom.centerline import pointInPolygon
# from defs.way_cluster_params import transitionSlope
from way.item.section import Section
from way.item.connectors import IntConnector

# helper functions -----------------------------------------------
def pairs(iterable):
    # s -> (s0,s1), (s1,s2), (s2, s3), ...
    p1, p2 = tee(iterable)
    next(p2, None)
    return zip(p1,p2)

def cyclePair(lst):
    prevs, nexts = tee(lst)
    prevs = islice(cycle(prevs), len(lst) - 1, None)
    return zip(prevs,nexts)

def cycleTriples(iterable):
    # iterable -> (pn-1,pn,p0), (pn,p0,p1), (p0,p1,p2), (p1,p2,p3), (p2,p3,p4), ... 
    p1, p2, p3 = tee(iterable,3)
    p1 = islice(cycle(p1), len(iterable) - 2, None)
    p2 = islice(cycle(p2), len(iterable) - 1, None)
    return zip(p1,p2,p3)
# end helper functions -------------------------------------------

class LeavingWay():
    # This class holds the data of a way that leaves an intersection. The direction
    # of its polyline, the sides 'left' and 'right', and the start and target
    # trim positions are redirected correctly to the instance <section> of Section.

    def __init__(self, street, section, leaving):
        # street:   Instance of Street, connected to the intersection
        # section:  Instance of a Section, that holds the way-section in its
        #           original direction.
        # leaving:  Direction as outgoing section. True, if same as original.
        self.street = street
        self.section = section
        self.leaving = leaving
        self.offset = self.section.offset if leaving else -self.section.offset
        self.polyline = PolyLine(section.polyline[:]) if leaving else PolyLine(section.polyline[::-1])
        self.widthL = self.section.width/2 - self.offset
        self.widthR = -self.section.width/2 - self.offset
        self.polygon = self.polyline.buffer(abs(self.widthL),abs(self.widthR))
        self.isLoop = self.section.isLoop

    def updateOffset(self):
        if self.section.offset != 0.0:
            self.offset = self.section.offset if self.leaving else -self.section.offset
            self.widthL = self.section.width/2 - self.offset
            self.widthR = -self.section.width/2 - self.offset

class Intersection(Item):
    
    ID = 0
    
    def __init__(self, location):
        super().__init__()
        self.id = Intersection.ID
        Intersection.ID += 1
        self._location = location
        self.leaveWays = []

        # Reference to first connector of circular doubly-linked list of IntConnectors.
        self.startConnector = None

    @property
    def location(self):
        return self._location
    
    @property
    def order(self):
        return len(self.leaveWays)

    def update(self, inStreets, outStreets):
        for street in inStreets:
            item = street.tail
            if isinstance(item,Section):
                self.leaveWays.append( LeavingWay(street, item, False) )
        for street in outStreets:
            item = street.head
            if isinstance(item,Section):
                self.leaveWays.append( LeavingWay(street, item, True) )
        self.sortSections()

    def sortSections(self):
    # Sort the leaving ways in <self.leaveWays> by the angle in counter-clockwise order
    # of the first segment around their start positions (the node at <self.location>)
        def _pseudoangle(d):
            p = d[0]/(abs(d[0])+abs(d[1])) # -1 .. 1 increasing with x
            return 3 + p if d[1] < 0 else 1 - p 
        
        self.leaveWays = sorted(self.leaveWays, key=lambda x: _pseudoangle(x.polyline[1]-self.location))

    def insertConnector(self, connector):
        # Inserts the instance <connector> of IntConnector into the circular doubly-linked list,
        # attached to self.startConnector. It is inserted "after", which is in counter-clockwise direction.
        if self.startConnector is None:
            connector.succ = connector.pred = connector
            self.startConnector = connector
        else:
            last = self.startConnector.pred
            connector.succ = self.startConnector
            self.startConnector.pred = connector
            connector.pred = last
            last.succ = connector

    def processIntersection(self):
        if len(self.leaveWays) < 3:
            return
        
        for way in self.leaveWays:
            way.updateOffset()
        
        self.connectors_old = dict()
        # self.area = []

        for way in self.leaveWays:
            connector = IntConnector(self)
            connector.item = way.street
            connector.leaving = way.leaving
            if connector.leaving:
                way.street.pred = connector
            else:
                way.street.succ = connector
            self.insertConnector(connector)
