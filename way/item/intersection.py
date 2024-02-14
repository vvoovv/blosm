from itertools import tee,islice, cycle

from .item import Item
from lib.CompGeom.PolyLine import PolyLine
from lib.CompGeom.offset_intersection import offsetPolylineIntersection
from lib.CompGeom.centerline import pointInPolygon
from lib.CompGeom.LinePolygonClipper import LinePolygonClipper
from defs.way_cluster_params import transitionSlope
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


class Intersection(Item):
    
    ID = 0
    
    def __init__(self, location):
        super().__init__()
        self.id = Intersection.ID
        Intersection.ID += 1
        self._location = location
        self.leaveWays = []

        self.area = []
        # self.connectors_old = None
        self.startConnector = None

    @property
    def location(self):
        return self._location
    
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
        # attached to self.connectors. It is inserted "after", which is in counter-clockwise direction.
        if self.startConnector is None:
            connector.succ = connector.pred = connector
            self.startConnector = connector
        else:
            last = self.startConnector.pred
            connector.succ = self.startConnector
            self.startConnector.pred = connector
            connector.pred = last
            last.succ = connector

    def cleanShortWays(self,debug=False):
        # A way is defined as short way, when at least one of its left or right ends of its 
        # border is within the area of its neighbor way. Such ways are removed from the
        # intersection, but returned for eventual further use.
        shortWays = []

        # If outways of intersection form loops, don' clean anything
        if any(way.isLoop for way in self.leaveWays):
            return []

        if len(self.leaveWays) < 3:
            return []

        for left,centre,right in cycleTriples(self.leaveWays):
            endR = centre.polyline.offsetPointAt(len(centre.polyline)-1.,centre.widthR)
            endL = centre.polyline.offsetPointAt(len(centre.polyline)-1.,centre.widthL)
            # if debug:
            #     from debug import plt,plotPolygon,plotEnd
            #     plotPolygon(right.polygon,False,'r')
            #     plotPolygon(left.polygon,False,'g')
            #     plotPolygon(centre.polygon,False,'k:')
            #     plt.plot(endR[0],endR[1],'ro')
            #     plt.plot(endL[0],endL[1],'go')
            #     plotEnd()
            if pointInPolygon(right.polygon, endR) in ('IN','ON'):
                shortWays.append(centre)
            elif pointInPolygon(right.polygon, endL) in ('IN','ON'):
                # If the centerline leaves the neighbor and returns, then it's not a short way.
                border = centre.polyline.parallelOffset(centre.widthL)
                doesLeave = any( (pointInPolygon(right.polygon,p) == 'OUT') for p in border )
                if not doesLeave:
                    shortWays.append(centre)
            if pointInPolygon(left.polygon, endR) in ('IN','ON'):
                # If the centerline leaves the neighbor and returns, then it's not a short way.
                border = centre.polyline.parallelOffset(centre.widthR)
                doesLeave = any( (pointInPolygon(left.polygon,p) == 'OUT') for p in border )
                if not doesLeave:
                    shortWays.append(centre)
            elif pointInPolygon(left.polygon, endL) in ('IN','ON'):
                shortWays.append(centre)

        # Remove short ways from Intersection
        for way in shortWays:
            if way in self.leaveWays:
                self.leaveWays.remove(way)

        return shortWays
    
    def processIntersection(self):
        if len(self.leaveWays) < 3:
            return
        
        self.connectors_old = dict()
        self.area = []
        
        for rightWay, centerWay, leftWay in cycleTriples(self.leaveWays):
            # Embed the connector of centerWay
            connector = IntConnector(self)
            connector.item = centerWay.street
            connector.leaving = centerWay.leaving
            if connector.leaving:
                centerWay.street.pred = connector
            else:
                centerWay.street.succ = connector

            # Find the boundary intersection p1 at the right side of center-way
            p1, type = offsetPolylineIntersection(rightWay.polyline,centerWay.polyline,rightWay.widthL,-centerWay.widthR,True,0.1)
            if type == 'valid':
                _,tP1 = centerWay.polyline.orthoProj(p1)
                if tP1 < 0.:
                    tP1 = 0.
                    p1 = centerWay.polyline.offsetPointAt(tP1,centerWay.widthR)
                elif isinstance(tP1,int):   # inter buffer point, reproject
                    p1 = centerWay.polyline.offsetPointAt(tP1,centerWay.widthR)
            elif type == 'parallel':
                transWidth = min(centerWay.polyline.length()*0.5,max(1.,abs(rightWay.widthL+centerWay.widthR)/transitionSlope) )
                tP1 = centerWay.polyline.d2t(transWidth)
                p1 = centerWay.polyline.offsetPointAt(tP1,centerWay.widthR)
            else: # out
                print('out')
                # from debug import plt
                # p = self.location
                # plt.plot(p[0],p[1],'ro',markersize=12,zorder=999,markeredgecolor='orange', markerfacecolor='none')
                continue

            # Find the boundary intersection p3 at the left side of center-way
            p3, type = offsetPolylineIntersection(centerWay.polyline,leftWay.polyline,centerWay.widthL,-leftWay.widthR,True,0.1)
            if type == 'valid':
                _,tP3 = centerWay.polyline.orthoProj(p3)
                if tP3 < 0.:
                    tP3 = 0.
                    p3 = centerWay.polyline.offsetPointAt(tP3,centerWay.widthL)
                elif isinstance(tP3,int):   # inter buffer point, reproject
                    p3 = centerWay.polyline.offsetPointAt(tP3,centerWay.widthL)
            elif type == 'parallel':
                transWidth = min(centerWay.polyline.length()*0.5,max(1.,abs(centerWay.widthL+leftWay.widthR)/transitionSlope) )
                tP3 = centerWay.polyline.d2t(transWidth)
                p3 = centerWay.polyline.offsetPointAt(tP3,centerWay.widthL)
            else: # out
                print('out')
                # from debug import plt
                # p = self.location
                # plt.plot(p[0],p[1],'ro',markersize=12,zorder=999,markeredgecolor='orange', markerfacecolor='none')
                continue

            # Project p1 and p3 onto the centerline of the center-way and create intermediate
            # polygon point <p2>.
            Id = centerWay.section.id  if centerWay.leaving else -centerWay.section.id
            t0 = 0.
            if tP3 > tP1:
                p2 = centerWay.polyline.offsetPointAt(tP3,centerWay.widthR)
                t0 = tP3
                self.connectors_old[Id] = len(self.area)+1
                connector.index = len(self.area)+1
            else:
                p2 = centerWay.polyline.offsetPointAt(tP1,centerWay.widthL)
                t0 = tP1
                self.connectors_old[Id] = len(self.area)
                connector.index = len(self.area)

            if centerWay.leaving:
                centerWay.section.trimS = max(centerWay.section.trimS, t0)
            else:
                t = len(centerWay.section.polyline)-1 - t0
                centerWay.section.trimT = min(centerWay.section.trimT, t)            

            # last vertex == first vertex ?
            if self.area and (p1-self.area[-1]).length < 0.01:
                self.area = self.area[:-1]
                self.connectors_old[Id] -= 1
                connector.index -= 1

            if p1==p2 or p2==p3: # way is perpendicular
                self.area.extend([p1,p3])
            else:
                self.area.extend([p1,p2,p3])

            self.insertConnector(connector)

        if self.area:
            self.area = self.area[:-1] if (self.area[0]-self.area[-1]).length < 0.001 else self.area
