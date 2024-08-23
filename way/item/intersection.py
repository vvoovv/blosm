from itertools import tee,islice, cycle

from .item import Item
from lib.CompGeom.PolyLine import PolyLine
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

        # Attributes for minor intersection
        self.isMinor = False

        self.leftHead = None
        self.leftTail = None
        self.rightHead = None
        self.rightTail = None

        self.leaving = None     # leaving major street
        self.arriving = None    # arriving major street

        self._pred = None       # will be set when linked into a Street
        self._succ = None       # will be set when linked into a Street

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

# For minor intersections -----------------------------------------------------
    @staticmethod
    def isMinorCategory(section):
        if not section.valid: return False
        return  section.category in  ['footway', 'cycleway','service'] or \
                ('service' in section.tags and \
                section.tags['service']=='driveway')

    def isMinorIntersection(self):
        minorCount = 0
        majorCount = 0
        majorCategories = set()
        majorIndices = []
        for indx, leaveWay in enumerate(self.leaveWays):
            if Intersection.isMinorCategory(leaveWay.section):
                minorCount += 1
            else:
                majorCount += 1
                majorCategories.add(leaveWay.section.category)
                majorIndices.append(indx)
        return (majorCount == 2 and minorCount>0 and len(majorCategories)==1)

    def insertLeftConnector(self, conn):
        # Inserts the instance <connector> of IntConnector at the end of the linear doubly-linked list,
        # attached to self.leftHead. It is inserted "after", which is in counter-clockwise direction.
        connector = conn.copy()
        if self.leftHead is None:
            connector.pred = None
            connector.succ = None
            self.leftHead = connector
            self.leftTail = connector
        else:
            self.leftTail.succ = connector
            connector.succ = None
            connector.pred = self.leftTail
            self.leftTail = connector

    def insertRightConnector(self, conn):
        # Inserts the instance <connector> of IntConnector at the end of the linear doubly-linked list,
        # attached to self.rightHead. It is inserted "after", which is in counter-clockwise direction.
        connector = conn.copy()
        if self.rightHead is None:
            connector.pred = None
            connector.succ = None
            self.rightHead = connector
            self.rightTail = connector
        else:
            self.rightTail.succ = connector
            connector.succ = None
            connector.pred = self.leftTail
            self.rightTail = connector

    def transformToMinor(self):
        self.isMinor = True

        # Find a leaving major street (by connector <conn>)
        for conn in IntConnector.iterate_from(self.startConnector):
            if conn.leaving and not Intersection.isMinorCategory(conn.item.head):
                break
        self.leaving = conn.item

        # The circular list of connectors of this intersection is
        # ordered counter-clockwise. When we start with a leaving section,
        # the first minor sections are to the left.
        for conn in IntConnector.iterate_from(conn.succ):
            section = conn.item.head if conn.leaving else conn.item.tail
            if Intersection.isMinorCategory(section):
                self.insertLeftConnector(conn)
            else:
                break # We found the next major section

        self.arriving = conn.item

        # Then, the minor sections to the right are collected
        for conn in IntConnector.iterate_from(conn.succ):
            section = conn.item.head if conn.leaving else conn.item.tail
            if Intersection.isMinorCategory(section):
                self.insertRightConnector(conn)
            else:
                break # this is again the first major section

        # Check direction of major streets
        # see https://github.com/prochitecture/blosm/issues/106#issuecomment-2305297075
        if self.leaving.dst == self.location and self.arriving.src == self.location:
            self.leaving, self.arriving = self.arriving, self.leaving
        
        if self.leaving.src != self.location or self.arriving.dst != self.location:
            # One of the streets runs the wrong way
            # Do not accept as minor intersection
            self.isMinor = False

    @staticmethod
    def iterate_from(conn_item):
        while conn_item is not None:
            yield conn_item
            conn_item = conn_item.succ
