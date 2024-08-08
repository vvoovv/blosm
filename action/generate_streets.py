from collections import defaultdict
from itertools import tee, islice, cycle, permutations, accumulate
from statistics import median
from mathutils import Vector
import re

from app import AppType

from way.way_network import WayNetwork, NetSection
from way.way_algorithms import createSectionNetwork
from way.way_section import WaySection
from way.way_properties import lanePattern
from way.way_cluster import createLeftTransition, createClippedEndArea, createRightTransition, \
                           createLeftIntersection, createRightIntersection, createShortClusterIntersection
# from way.way_intersections import Intersection
from way.intersection_cluster import IntersectionCluster
from way.overlap_handler import EndWayHandler
from defs.road_polygons import ExcludedWayTags
from defs.way_cluster_params import minTemplateLength, minNeighborLength, searchDist,\
                                    canPair, dbScanDist, transitionSlope

from way.item import Intersection, Corner, Section, Street, SideLane, SymLane

from lib.SweepIntersectorLib.SweepIntersector import SweepIntersector
from lib.CompGeom.StaticSpatialIndex import StaticSpatialIndex, BBox
from lib.CompGeom.algorithms import SCClipper, orderAsPolygon
from lib.CompGeom.BoolPolyOps import boolPolyOp
from lib.CompGeom.GraphBasedAlgos import DisjointSets
from lib.CompGeom.PolyLine import PolyLine, LinearInterpolator
from lib.CompGeom.LinePolygonClipper import LinePolygonClipper
from lib.CompGeom.centerline import centerlineOf, pointInPolygon
from lib.CompGeom.dbscan import dbClusterScan
from lib.CompGeom.clipParallelPart import clipParallelPart
from lib.CompGeom.simplifyers import simplifyEnds,simplifyRDP


class BaseWaySection:
    
    def getNormal(self, index, left):
        centerline = self.centerline
        numPoints = len(centerline)
        
        if index == 0 or (index==-2 and numPoints==2):
            vector = centerline[1] - centerline[0]
        elif index == -1 or (index==1 and numPoints==2):
            vector = centerline[-1] - centerline[-2]
        else:
            # vector along <centerline> after the point with <index>
            vectorAfter = centerline[index+1] - centerline[index]
            vectorAfter.normalize()
            # vector along <centerline> before the point with <index>
            vectorBefore = centerline[index-1] - centerline[index]
            vectorBefore.normalize()
            # check if <vectorAfter> and <vectorBefore> are collinear and of the opposite direction
            if abs(-1-vectorAfter.dot(vectorBefore)) < 0.001:
                vector = vectorAfter
            else:
                vector = vectorBefore + vectorAfter
                vector.normalize()
                # check if <vector> is to the right or to the left relative to <vectorAfter>
                crossProductPositive = vector[0]*vectorAfter[1] - vector[1]*vectorAfter[0] > 0.
                return vector\
                    if (left and not crossProductPositive) or (not left and crossProductPositive) else\
                    -vector
        
        vector.normalize()
        
        return Vector((-vector[1], vector[0])) if left else Vector((vector[1], -vector[0]))
    
    def offsetPoint(self, index, left, distance, normal=None):
        return self.centerline[index] + distance * (normal or self.getNormal(index, left))


class StreetSection(BaseWaySection):
    
    def __init__(self):
        # The centerline of the street in forward direction (the direction of the centerline
        # is the direction it was originally drawn by the OSM volunteer operator). When it
        # starts or ends at an intersection, it is shortened to fit there.
        # A Python list of vertices of type mathutils.Vector.
        self.centerline = None
        
        # The width of the way.
        self.width = None
        
        # The category of the way-section.
        self.category = None

        # The OSM tags of the way-section.
        self.tags = None

        # The number of lanes, seen relative to the forward direction
        # of the centerline. For one-way streets, only one <lanesForward>
        # is used.
        self.forwardLanes = 0

        # The number of lanes, seen relative to the inverse forward direction
        # of the centerline. For one-way streets, only one <lanesForward>
        # is used.
        self.backwardLanes = 0

        # The number of lanes, tagged as 'both_ways'.
        # of the centerline. For one-way streets, only one <lanesForward>
        # is used.
        self.bothLanes  = 0

        # self.forwardLanes + self.backwardLanes + self.bothLanes
        self.totalLanes  = 0

        # The offset from the centerline at the wider part of the street, when it has turn
        # or a merge lanes. Zero for all other streets.
        self.offset = 0.

        # True, if there is a transition lane at left (laneL) or right (LaneR)
        self.laneR = False
        self.laneL = False

        # Although the centerline of the street is already shortened, these attributes give
        # the trim values for "virtual" ways in clusters (constant distance between them).
        self.trimStart = 0.
        self.trimEnd = 0.
        
        # Instance of either TransitionSideLane, TransitionSymLane or IntersectionArea,
        # if connected to one of those at start.
        self.start = None

        # Instance of either TransitionSideLane, TransitionSymLane or IntersectionArea,
        # if connected to one of those at end.
        self.end = None

        # Number of vertices along the street's polyline.
        self.numPoints = 0

        self.rendered = False
    
    def getLeftBorderDistance(self):
        return 0.5*self.width
    
    def getRightBorderDistance(self):
        return 0.5*self.width

# class TransitionSideLane():
#     def __init__(self):
#         # A tuple of way IDs of the way-sections, that connect to this transition. The ID is
#         # positive, when the start of the way-section connects to the transition, and
#         # negative, when the way-section ends here.
#         self.ways = None

#         # True, if there is a turn lane to the left from the narrower way-section.
#         self.laneL = False

#         # True, if there is a turn lane to the right from the narrower way-section.
#         self.laneR = False

#         # Instance of incoming StreetSection.
#         self.incoming = None

#         # Instance of outgoing StreetSection.
#         self.outgoing = None

#         # self.outgoing.totalLanes > self.incoming.totalLanes
#         self.totalLanesIncreased = 0

# class TransitionSymLane():
#     def __init__(self):
#         # The vertices of the polygon in counter-clockwise order. A Python
#         # list of vertices of type mathutils.Vector.
#         self.polygon = None
        
#         # The connectors of this polygon to the way-sections.
#         # A dictionary of indices, where the key is the corresponding
#         # way-section key in the dictionary of TrimmedWaySection. The key is
#         # positive, when the start of the way-section connects to the area, and
#         # negative, when the way-section ends here. The value <index> in the
#         # dictionary value is the the first index in the intersection polygon
#         # for this connector. The way connects between <index> and <index>+1. 
#         self.connectors = dict()
                
#         self.numPoints = 0
        
#         self.connectorsInfo = None

    def getConnectorsInfo(self):
        """
        Form a Python list of Python tuples with the following 3 elements:
        * index in <self.polygon> of the starting point of the connector
            to the adjacent way section or way cluster
        * <True> if it's a connector to a street segment, <False> otherwise
        * <segmentId>
        * 0 if a street segment start at the connector -1 otherwise
        """
        if not self.connectorsInfo:
            self.numPoints = len(self.polygon)
            
            connectorsInfo = self.connectorsInfo = []
            if self.connectors:
                connectorsInfo.extend(
                    # <False> means that it isn't a cluster of street segments
                    (polyIndex, False, abs(segmentId), 0 if segmentId>0 else -1) for segmentId, polyIndex in self.connectors.items()
                )
        
            connectorsInfo.sort()
        
        return self.connectorsInfo


class WayCluster(BaseWaySection):
    
    def __init__(self):
        # The centerline of the cluster in forward direction. (first vertex is start.
        # and last is end). A Python list of vertices of type <mathutils.Vector>.
        # It is the centerline between the centerlines of the outermost ways in the cluster.
        self.centerline = []
        
        # The distance from the cluster centerline to the centerline of the leftmost way,
        # seen relative to the direction of the cluster centerline.
        self.distToLeft = 0.
        
        # A Python list of way descriptors represented by instances of the class <TrimmedWaySection>.
        # The way descriptors in this list are ordered from
        # left to right relative to the centerline of the cluster.
        self.waySections = []

        # True, if start of cluster is connected to other ways or clusters, else dead-end
        self.startConnected = None

        # True, if end of way is connected to other ways or clusters, else dead-end
        self.endConnected = None
    
    def getLeftBorderDistance(self):
        return -self.waySections[0].offset + 0.5*self.waySections[0].width
    
    def getRightBorderDistance(self):
        return self.waySections[-1].offset + 0.5*self.waySections[-1].width


class IntersectionArea():
    
    def __init__(self):
        # The vertices of the polygon in counter-clockwise order. A Python
        # list of vertices of type mathutils.Vector.
        self.polygon = None
        
        # The connectors of this polygon to the way-sections.
        # A dictionary of indices, where the key is the corresponding
        # way-section key in the dictionary of TrimmedWaySection. The key is
        # positive, when the start of the way-section connects to the area, and
        # negative, when the way-section ends here. The value <index> in the
        # dictionary value is the the first index in the intersection polygon
        # for this connector. The way connects between <index> and <index>+1. 
        self.connectors = dict()
        
        # The connectors of this polygon to the way-clusters.
        # A dictionary of indices, where the key is the corresponding way-cluster
        # in the dictionary <wayClusters> of The manager. The key is positive, when
        # the start of the way-cluster connects to the area, and negative, when the
        # way-cluster ends here. The value <index> in the dictionary value is the
        # first index in the intersection polygon for this connector. The way-cluster
        # connects between <index> and <index>+1.
        self.clusterConns = dict()
        
        self.numPoints = 0
        
        self.connectorsInfo = None
    
    def getConnectorsInfo(self):
        """
        Form a Python list of Python tuples with the following 3 elements:
        * index in <self.polygon> of the starting point of the connector
            to the adjacent way section or way cluster
        * <True> if it's a connector to a street segment, <False> otherwise
        * <segmentId>
        * 0 if a street segment start at the connector -1 otherwise
        """
        if not self.connectorsInfo:
            self.numPoints = len(self.polygon)
            
            connectorsInfo = self.connectorsInfo = []
            if self.connectors:
                connectorsInfo.extend(
                    # <False> means that it isn't a cluster of street segments
                    (polyIndex, False, abs(segmentId), 0 if segmentId>0 else -1) for segmentId, polyIndex in self.connectors.items()
                )
            if self.clusterConns:
                connectorsInfo.extend(
                    # <True> means that it is a cluster of street segments
                    (polyIndex, True, abs(segmentId),  0 if segmentId>0 else -1) for segmentId, polyIndex in self.clusterConns.items()
                )
        
            connectorsInfo.sort()
        
        return self.connectorsInfo


# helper functions -----------------------------------------------
def pairs(iterable):
    # iterable -> (p0,p1), (p1,p2), (p2, p3), ...
    p1, p2 = tee(iterable)
    next(p2, None)
    return zip(p1,p2)

def cyclePair(iterable):
    # iterable -> (p0,p1), (p1,p2), (p2, p3), ..., (pn, p0)
    prevs, nexts = tee(iterable)
    prevs = islice(cycle(prevs), len(iterable) - 1, None)
    return zip(prevs,nexts)

def triples(iterable):
    # iterable -> (p0,p1,p2), (p1,p2,p3), (p2,p3,p4), ...
    p1, p2, p3 = tee(iterable,3)
    next(p2, None)
    next(p3, None)
    next(p3, None)
    return zip(p1,p2,p3)

def isEdgy(polyline):
    vu = polyline.unitVectors()
    for v1,v2 in pairs(vu):
        if abs(v1.cross(v2)) > 0.6:
            return True
    return False

def mad(x):
    # median absolute deviation
    # https://en.wikipedia.org/wiki/Median_absolute_deviation
    m = median(x)
    return median( abs(xi-m) for xi in x)

def createStreetSection(offset, width, nrOfLanes, category, tags):
    waySection = StreetSection()
    
    waySection.offset, waySection.width, waySection.nrOfLanes, waySection.category, waySection.tags =\
        offset, width, nrOfLanes, category, tags
    
    return waySection
# ----------------------------------------------------------------

class StreetGenerator():
    
    def __init__(self, styleStore, getStyle, leftHandTraffic=True):
        self.styleStore = styleStore
        self.getStyle = getStyle
        # Interface via manager
        self.leftHandTraffic = leftHandTraffic
        self.intersectionAreas = None
        self.wayClusters = None
        self.waySectionLines = None
        self.networkGraph = None
        self.sectionNetwork = None

        # Internal use
        self.parallelSectionKeys = None
        self.waySections = dict()
        self.internalTransitionSideLanes = dict()
        self.internalTransitionSymLanes = dict()
        self.intersections = dict()
        self.waysForClusters = []
        self.longClusterWays = []
        self.processedNodes = set()
        self.waysCoveredByCluster = []
        self.areaIndex = None

        self.allWays = True

    def do(self, manager):
        self.wayManager = manager
        self.waymap = manager.waymap
        self.intersections = manager.intersections
        self.transitionSideLanes = manager.transitionSideLanes
        self.transitionSymLanes = manager.transitionSymLanes
        self.wayClusters = manager.wayClusters
        self.waySectionLines = manager.waySectionLines
        NetSection.ID = 0   # This class variable doesn't get reset with new instance of StreetGenerator!!

        self.useFillet = False          # Don't change, it does not yet work!

        self.findSelfIntersections()
        self.createWaySectionNetwork()
        self.createEmptyWaymap()
        # self.createWaySections()
        # self.createTransitionLanes()

        # plotPureNetwork(self.sectionNetwork)
        self.createParallelSections()
        self.createSymSideLanes()
        # plotEnd()

        self.updateIntersections()
        # self.experimentalClusters()
        # self.experimentalSimpleClusters()
        self.finalizeOutput()

        # self.detectWayClusters()
        # self.createLongClusterWays()

        # self.createIntersectionAreas()
        # self.mergeOverlappingIntersections()
        # self.mergeOverlapsBySymLanesAndIntersections()
        # self.createOutput()
        # missing = [idx for idx in set(abs(idx) for t in manager.transitionSymLanes for idx in t.connectors) if idx not in manager.waySectionLines]
        # print(missing)
        # plotPureNetwork(self.sectionNetwork)

        # self.detectWayClusters()
        # self.createLongClusterWays()
        # self.createClusterTransitionAreas()
        # self.createClusterIntersections()
        # self.createClippedClusterEnds()
        # self.createWayClusters()
        # self.cleanCoveredWays()
        # self.createOutput()

    def findSelfIntersections(self):
        # some way tags to exclude, used also in createWaySectionNetwork(),
        # ExcludedWayTags is defined in <defs>.
        uniqueSegments = defaultdict(set)
        if self.allWays: 
            getWays = self.wayManager.getAllWays()
        else:
            getWays = self.wayManager.getAllVehicleWays()
        for way in getWays:#self.wayManager.getAllWays():#getAllVehicleWays():
            if [tag for tag in ExcludedWayTags if tag in way.category]:
                continue
            for segment in way.segments:
                v1, v2 = (segment.v1[0],segment.v1[1]),  (segment.v2[0],segment.v2[1])
                if v1 not in uniqueSegments.get(v2,[]):
                    uniqueSegments[v1].add(v2)
        cleanedSegs = [(v1,v2) for v1 in uniqueSegments for v2 in uniqueSegments[v1]]

        intersector = SweepIntersector()
        self.intersectingSegments = intersector.findIntersections(cleanedSegs)

    # Creates the network graph <self.sectionNetwork> for way-sctions (ways between crossings)
    def createWaySectionNetwork(self):
        wayManager = self.wayManager

        # prepare clipper for this frame
        clipper = SCClipper(self.app.minX, self.app.maxX, self.app.minY, self.app.maxY)

        # Not really used. This is a relict from way_clustering.py
        wayManager.junctions = (
            [],#mainJunctions,
            []#smallJunctions
        )

        # create full way network
        wayManager.networkGraph = self.networkGraph = WayNetwork(self.leftHandTraffic)

        # some way tags to exclude, used also in createWaySectionNetwork(),
        # ExcludedWayTags is defined in <defs>.
        if self.allWays: 
            getWays = self.wayManager.getAllWays()
        else:
            getWays = self.wayManager.getAllVehicleWays()
        for way in getWays:#self.wayManager.getAllWays():#getAllVehicleWays():
            # Exclude ways with unwanted tags
            if [tag for tag in ExcludedWayTags if tag in way.category]:
                continue

            for waySegment in way.segments:
                # Check for segments splitted by self-intersections
                segments = []
                newSegments = self.intersectingSegments.get( (tuple(waySegment.v1),tuple(waySegment.v2)), None)
                if newSegments:
                    for v1,v2 in zip(newSegments[:-1],newSegments[1:]):
                        segments.append((v1,v2))
                else:
                    segments.append((waySegment.v1,waySegment.v2))

                for segment in segments:
                    v1, v2 = Vector(segment[0]),Vector(segment[1])
                    accepted, v1, v2 = clipper.clip(v1,v2)
                    if accepted:
                        netSeg = NetSection(v1,v2,way.category,way.element.tags,(v2-v1).length)
                        wayManager.networkGraph.addSegment(netSeg,False)

        borderPolygon = clipper.getPolygon()
        for v1,v2 in zip(borderPolygon[:-1],borderPolygon[1:]):
            netSeg = NetSection(v1,v2,'scene_border',None, (v2-v1).length) 
            wayManager.networkGraph.addSegment(netSeg)

        # create way-section network
        wayManager.sectionNetwork = self.sectionNetwork = createSectionNetwork(wayManager.networkGraph,self.leftHandTraffic)

    def createEmptyWaymap(self):
        for net_section in self.sectionNetwork.iterAllForwardSegments():
            if net_section.category != 'scene_border':

                # Create Section from net-section, including style block parameters
                section = Section(net_section,PolyLine(net_section.path),self.sectionNetwork)
                oneway = 'oneway' in section.tags and section.tags['oneway'] != 'no'
                street = Street(section.src, section.dst)
                street.append(section)
                streetStyle = self.styleStore.get( self.getStyle(street) )
                street.style = streetStyle
                street.setStyleBlockFromTop(streetStyle)
                section.street = street

                # Derive Section attributes
                if oneway:
                    totalNumLanesOneway = street.getStyleBlockAttr("totalNumLanesOneway")
                    nrLanes = totalNumLanesOneway if totalNumLanesOneway else street.getStyleBlockAttr("totalNumLanes")
                else:
                    nrLanes = street.getStyleBlockAttr("totalNumLanes")
                props = { 
                    'nrLanes' : nrLanes,
                    'laneWidth' : street.getStyleBlockAttr("laneWidth")
                }
                _,fwdPattern,bwdPattern,bothLanes = lanePattern(section.category,section.tags,self.leftHandTraffic,props)
                section.setSectionAttributes(oneway, fwdPattern, bwdPattern, bothLanes, props)

                self.waymap.addStreetNode(Intersection(section.src))
                self.waymap.addStreetNode(Intersection(section.dst))
                self.waymap.addSection(street)

                # If there are corners, the section must be split to enable finding of parallel sections
                # corners = section.polyline.getCorners(0.6) if section.category in ['footway', 'cycleway'] else []

                # if False and corners and self.app.type == AppType.commandLine:
                #     from debug import plt, plotPureNetwork
                #     for nextCorner in corners:
                #         c = section.polyline[nextCorner]
                #         plt.plot(c[0],c[1],'ro',markersize=8,zorder=999,markeredgecolor='red', markerfacecolor='none')

                # if corners:
                #     corners.append(len(section.polyline)-1)
                #     self.waymap.addStreetNode(Intersection(section.src))
                #     lastCorner = 0
                #     for nextCorner in corners:
                #         splitline = PolyLine( section.polyline[lastCorner:nextCorner+1] )
                #         subsection = Section(net_section,splitline,self.sectionNetwork)
                #         subsection.setSectionAttributes(oneway,fwdPattern,bwdPattern,bothLanes,props)

                #         street = Street(subsection.src, subsection.dst)
                #         street.append(subsection)                       
                #         street.setStyle(streetStyle)

                #         self.waymap.addStreetNode(Corner(subsection.dst))
                #         self.waymap.addSection(street)
                #         lastCorner = nextCorner
                #     self.waymap.replaceStreetNodeBy(Intersection(subsection.dst))
                # else:
                #     Add section, we do not yet know the type of the intersections
                #     self.waymap.addStreetNode(Intersection(section.src))
                #     self.waymap.addStreetNode(Intersection(section.dst))

                #     street = Street(section.src, section.dst)
                #     section.street = street
                #     street.append(section)
                #     street.setStyle(streetStyle)
                
                #     self.waymap.addSection(street)

        # Add ways to intersections
        for location, intersection in self.waymap.iterNodes(Intersection):
            inStreets, outStreets = self.waymap.getInOutSections(location)
            intersection.update(inStreets, outStreets)


    # def createWaySections(self):
    #     for net_section in self.sectionNetwork.iterAllForwardSegments():
    #         if net_section.category != 'scene_border':
    #             section = WaySection(net_section,self.sectionNetwork)

    #             # Find lanes and their widths
    #             isOneWay,fwdPattern,bwdPattern,bothLanes = lanePattern(section.category,section.tags,self.leftHandTraffic)
    #             section.isOneWay = isOneWay
    #             section.lanePatterns = (fwdPattern,bwdPattern)
    #             section.totalLanes = len(fwdPattern) + len(bwdPattern) + bothLanes
    #             section.forwardLanes = len(fwdPattern)
    #             section.backwardLanes = len(bwdPattern)
    #             section.bothLanes = bothLanes
    #             estimateWayWidths(section)

    #             self.waySections[net_section.sectionId] = section

    # def createTransitionLanes(self):
    #     for i,node in enumerate(self.sectionNetwork):
    #         outSections = [s for s in self.sectionNetwork.iterOutSegments(node) if s.category != 'scene_border']
    #         # Find transition nodes
    #         if len(outSections) == 2:
    #             # Only intersections of two ways (which are in fact just continuations) are checked.
    #             # This is not really correct, but how to decide in the other cases?
    #             way1 = self.waySections[outSections[0].sectionId]
    #             way2 = self.waySections[outSections[1].sectionId]
    #             widerWay = way1 if way1.totalLanes > way2.totalLanes else way2
    #             hasTurns = bool( re.search(r'[^N]', widerWay.lanePatterns[0] ) )
    #             if hasTurns:
    #                 wayIDs, laneL, laneR = createSideLaneData(node,way1,way2)
    #                 sideLane = TransitionSideLane()
    #                 sideLane.ways = wayIDs
    #                 sideLane.laneL = laneL
    #                 sideLane.laneR = laneR
    #                 self.internalTransitionSideLanes[node] = sideLane
    #             else:
    #                 area, connectors = createSymLaneData(node,way1,way2)
    #                 symLane = TransitionSymLane()
    #                 symLane.polygon = area
    #                 symLane.connectors = connectors
    #                 self.internalTransitionSymLanes[node] = symLane

    def createParallelSections(self):
        # Create spatial index (R-tree) of sections
        candidateIndex = StaticSpatialIndex()
        index2DictKey = dict()    # Dictionary from index to dictKey, which is (src,dst,multKey)
        boxes = dict()            # Bounding boxes of sections
        # Add boxes of all sections
        for src, dst, multKey, street in self.waymap.edges(data='object',keys=True):
            # multKey is 0,1,.. for multiple edges between equal nodes (waymap is a MultiDiGraph of networkx)
            # It must be included in the dictionary entry key to distinguish them.
            dictKey = (src,dst,multKey)
            section = street.head
            
            if section.category in ('steps', 'footway', 'cycleway', 'path', 'service'):
                continue

            # # DEBUG Show section id
            # if False and self.app.type == AppType.commandLine:
            #     from debug import plt, plotPolygon, randomColor, plotLine
            #     p = sum((v for v in section.polyline),Vector((0,0)) ) / len(section.polyline)
            #     p0 = section.polyline[len(section.polyline)//2]
            #     L = section.polyline.length()
            #     plt.plot([p0[0],p[0]],[p0[1],p[1]],'r')
            #     # plt.text(p[0],p[1],' %4.2f'%(L))
            #     plt.text(p[0],p[1],'%3d'%(section.id))
            #     # EDGY - TEST
            #     ds = (section.polyline[0]-section.polyline[-1]).length / section.polyline.length()
            #     if ds < 1.0:
            #         plt.text(p[0],p[1],'%4.2f'%(ds))
            #     if isEdgy(section.polyline):
            #         plt.text(p[0],p[1],'         %s'%('EDGY'),color='red')
            #     # END EDGY - TEST
            # # END DEBUG

            # Some are excluded
            ds = (section.polyline[0]-section.polyline[-1]).length / section.polyline.length()
            if isEdgy(section.polyline) and ds < 0.9:
                continue
            # if isEdgy(section.polyline): 
            #     vu = section.polyline.unitVectors()
            #     sumv = sum((abs(v1.cross(v2)) for v1,v2 in pairs(vu)),0.)/len(section.polyline)
            #     ds = (section.polyline[0]-section.polyline[-1]).length / section.polyline.length()
            #     # pt = sum((v for v in section.polyline), Vector((0,0)))/len(section.polyline)
            #     # plt.plot(pt[0],pt[1],'ko')
            #     # plt.text(pt[0],pt[1],'   %5.2f'%(ds) )
            #     if ds < 0.95:
            #         continue
            if section.polyline.length() < min(minTemplateLength,minNeighborLength):
                continue
            min_x = min(v[0] for v in section.polyline.verts)
            min_y = min(v[1] for v in section.polyline.verts)
            max_x = max(v[0] for v in section.polyline.verts)
            max_y = max(v[1] for v in section.polyline.verts)
            bbox = BBox(None,min_x,min_y,max_x,max_y)
            index = candidateIndex.add(min_x,min_y,max_x,max_y)
            index2DictKey[index] = dictKey
            bbox.index = index
            boxes[dictKey] = (min_x,min_y,max_x,max_y)
        candidateIndex.finish()

        self.parallelSectionKeys = DisjointSets()
        # Use every section, that has been inserted into the spatial index, 
        # as template and create a buffer around it.
        for src, dst, multKey, templateStreet in self.waymap.edges(data='object',keys=True):
            # multKey is 0,1,.. for multiple edges between equal nodes (waymap is a MultiDiGraph of networkx)
            # It must be included in the dictionary entry key to distinguish them.
            dictKey = (src,dst,multKey)
            template = templateStreet.head

            if dictKey in boxes:

                # Create buffer polygon with a width according to the category of the section.
                bufferWidth = searchDist[template.category]
                bufferPoly = template.polyline.buffer(bufferWidth,bufferWidth)

                # Create a line clipper using this polygon.
                clipper = LinePolygonClipper(bufferPoly.verts)

                # Get neighbors of this template section from the static spatial index, using its
                # bounding box, expanded by the buffer width as additionla search range.
                min_x,min_y,max_x,max_y = boxes[dictKey]
                results = stack = []
                neighborIndices = candidateIndex.query(min_x-bufferWidth,min_y-bufferWidth,
                                               max_x+bufferWidth,max_y+bufferWidth,results,stack)

                # Now test all these neighbors of the template for parallelism.
                for neigborIndex in neighborIndices:
                    neighDictKey = index2DictKey[neigborIndex]
                    if neighDictKey == dictKey:
                        continue # Skip, the template is its own neighbor.

                    srcNeigbor, dstNeigbor, neighMultKey = neighDictKey
                    neighbor = self.waymap[srcNeigbor][dstNeigbor][neighMultKey]['object'].head # Section from multigraph edge

                    # If the polyline of this neighbor ...
                    neighborLine = neighbor.polyline

                    # ... is longer than a minimal length, ...
                    if neighborLine.length() > minNeighborLength:
                        # ... then clip it with the buffer polygon
                        inLine, inLineLength, nrOfON = clipper.clipLine(neighborLine.verts)

                        if inLineLength/neighborLine.length() < 0.1:
                            continue # discard short inside lines. At least 10% must be inside.

                        # To check the quality of parallelism, some kind of "slope" relative 
                        # to the template's line is evaluated.
                        p1, d1 = template.polyline.distTo(inLine[0][0])     # distance to start of inLine
                        p2, d2 = template.polyline.distTo(inLine[-1][-1])   # distance to end of inLine
                        slope = abs(d1-d2)/inLineLength if inLineLength else 1.
    
                        # Conditions for acceptable inside line.
                        if slope < 0.15 and min(d1,d2) <= bufferWidth and nrOfON <= 2:
                            # Accept this pair as parallel.
                            self.parallelSectionKeys.addSegment((src,dst),(srcNeigbor,dstNeigbor))

        # DEBUG: Show clusters of parallel way-sections.
        # The plotting functions for this debug part are at the end of this module
        if self.app.type == AppType.commandLine:
            from debug import plt, plotPureNetwork, randomColor, plotEnd

            inBundles = False

            if not inBundles:
                plotPureNetwork(self.sectionNetwork)
            colorIter = randomColor(10)
            import numpy as np
            for bIndx,sectKeys in enumerate(self.parallelSectionKeys):
                if inBundles:
                    plotPureNetwork(self.sectionNetwork)
                    plt.title("Bundle "+str(bIndx))
                color = next(colorIter)
                for src,dst in sectKeys:
                    width = 2
                    if inBundles: 
                        color = "red"
                        width = 3
                    polyline = self.waymap.getSectionObject(src,dst,0).head.polyline
                    polyline.plot(color,width,'solid')
                    if inBundles: 
                        plt.scatter(polyline[0][0], polyline[0][1], s=80, facecolors='none', edgecolors='g',zorder=999)
                        plt.scatter(polyline[-1][0], polyline[-1][1], s=80, facecolors='none', edgecolors='g',zorder=999)
                        # plt.plot(polyline[0][0], polyline[0][1], 'go', markersize=8,zorder=999)
                        # plt.plot(polyline[-1][0], polyline[-1][1], 'go', markersize=8,zorder=999)
                if inBundles:
                    plotEnd()
            if not inBundles:
                plotEnd()
            # END DEBUG
                            
    def experimentalSimpleClusters(self):
        def mainWays(node):
            lowWayCategories = ["pedestrian", "track", "footway", "path", "cycleway", "bridleway" ]
            mainWayCount = 0
            for src,dst in self.waymap.in_edges(node):
                if self.waymap.getSectionObject(src,dst,0).head.category not in lowWayCategories:
                    mainWayCount += 1
            for src,dst in self.waymap.out_edges(node):
                if self.waymap.getSectionObject(src,dst,0).head.category not in lowWayCategories:
                    mainWayCount += 1
            # plt.text(node[0],node[1],str(mainWayCount),color='green')
            return mainWayCount

        points = set()
        for node in self.waymap.iterNodes(Intersection):
            location = node[0] 
            wayCount = mainWays(location)
            points.add( (location,wayCount) )

        areas = dict()
        vertices = set()
        for (location,wayCount) in points:
            if wayCount > 3:
                intersection = self.waymap.getStreetNode(location)
                if intersection:
                    areas[location] = intersection['object'].area
                    for p in areas[location]:
                        vertices.add( (p.freeze(),location) )

        groups = dbClusterScan(list(vertices), 10, 8)

        colorIter = randomColor(10)
        color = 'darkorange'#next(colorIter)
        for group in groups:
            locationsInGroup = list( set([p[1] for p in group]) )
            pointsInGroup = []
            for location in locationsInGroup:
                pointsInGroup.extend(areas[location])

            if len(locationsInGroup) < 2:
                continue

            # Show these points
            for p in locationsInGroup:
                plt.plot(p[0],p[1],'k1',zorder=900)
            for p in pointsInGroup:
                plt.scatter(p[0],p[1], s=50, marker='p', edgecolor='black', facecolor='white', zorder=900)

            convex = True
            if convex:
                from lib.CompGeom.ConvexHull import ConvexHull
                convHull = ConvexHull()
                hull = convHull.convexHull(pointsInGroup)
                plotPolygon(list(hull),False,'k',color,2,True,0.6,120)
                color = next(colorIter)
            else:
                from lib.CompGeom.AlphaShaper import AlphaShaper
                alphaShaper = AlphaShaper(pointsInGroup)
                hull = alphaShaper.alphaShapeAuto(1)
                hullVerts = [pointsInGroup[p] for p in hull]
                plotPolygon(hullVerts,False,'k',color,2,True,0.6,120)
                color = next(colorIter)

            
    def experimentalClusters(self):
        def intersectionType(node):
            lowWayCategories = ["pedestrian", "track", "footway", "path", "cycleway", "bridleway" ]
            mainWayCount = 0
            wayCount = 0
            for src,dst in self.waymap.in_edges(node):
                wayCount += 1
                if self.waymap.getSectionObject(src,dst,0).head.category not in lowWayCategories:
                    mainWayCount += 1
            for src,dst in self.waymap.out_edges(node):
                wayCount += 1
                if self.waymap.getSectionObject(src,dst,0).head.category not in lowWayCategories:
                    mainWayCount += 1
            return 'major' if mainWayCount>2 else 'main' if mainWayCount > 1 else 'low'


                        # all/parallel, major/,main/,low location/isect concave/convex
        detectionType = ['all',        'major','main',         'location',   'convex' ]


        # EITHER ENDS OF PARALLEL SECTION ...     
        if 'parallel' in detectionType:
            points = set()
            for cIndx,sectKeys in enumerate(self.parallelSectionKeys):
                # Section ends in this bundle are those, that appear only once
                sectionEnds = defaultdict(int)
                for src,dst in sectKeys:
                    sectionEnds[src] += 1
                    sectionEnds[dst] += 1

                # If their intersections contain >2 main ways, keep them as end points
                for node, count in sectionEnds.items():
                    typ = intersectionType(node)
                    points.add( (node,typ) )

        # OR CENTRES OF INTERSECTIONS ...
        else:
            points = set()
            for node in self.waymap.iterNodes(Intersection):
                location = node[0] 
                typ = intersectionType(location)
                points.add( (location,typ) )


        isectGroups = dbClusterScan(list(points), dbScanDist, 2)

        for cnt,group in enumerate(isectGroups):
            mjorIsects = [g[0] for g in group if g[1]=='major']
            mainIsects = [g[0] for g in group if g[1]=='main']
            lowIsects = [g[0] for g in group if g[1]=='low']

            usedIsects = []
            if "major" in detectionType:
                usedIsects.extend(mjorIsects)
            if "main" in detectionType:
                usedIsects.extend(mainIsects)
            if "low" in detectionType:
                usedIsects.extend(lowIsects)

            from debug import plt, plotPolygon, randomColor

            # usedIsects is list of locations of intersections
            # Create dict of areas of these intersection and collection of points
            points = set()
            areas = dict()
            for location in usedIsects:
                if 'location' in detectionType:
                    points.add( (location.freeze(),location) )
                else:
                    isect = self.waymap.getStreetNode(location)
                    if isect:
                        areas[location] = isect['object'].area
                        for p in areas[location]:
                            points.add( (p.freeze(),location) )
                    else:
                        plt.plot(location[0],location[1],'ro',markersize=20)

            # Group area points by density
            if 'location' in detectionType:
                groups = dbClusterScan(list(points), 20, 3)
            else:
                groups = dbClusterScan(list(points), 10, 9)
            # Show content of group
            # plt.close()
            colorIter = randomColor(10)
            color = 'darkorange'#next(colorIter)
            for cnt,group in enumerate(groups):
                locationsInGroup = list( set([p[1] for p in group]) )
                pointsInGroup = []
                for location in locationsInGroup:
                    if 'location' in detectionType:
                        pointsInGroup.append(location)
                    else:
                        pointsInGroup.extend(areas[location])

                if len(locationsInGroup) < 3:
                    continue

                # Show these points
                for p in locationsInGroup:
                    plt.plot(p[0],p[1],'k1',zorder=900)
                for p in pointsInGroup:
                    plt.scatter(p[0],p[1], s=50, marker='p', edgecolor='black', facecolor='white', zorder=900)
                for location in locationsInGroup:
                    if 'location' in detectionType:
                        if "major" in detectionType:
                            for p in mjorIsects:
                                plt.scatter(p[0],p[1], s=50, marker='p', edgecolor='black', facecolor='red', zorder=900)
                        if "main" in detectionType:
                            for p in mainIsects:
                                plt.scatter(p[0],p[1], s=50, marker='p', edgecolor='black', facecolor='white', zorder=900)
                        if "low" in detectionType:
                            for p in lowIsects:
                                plt.scatter(p[0],p[1], s=30, marker='p', edgecolor='black', facecolor='white', zorder=900)
                    else:
                        area = areas[location]
                        plotPolygon(area, False, 'k:','k',1,False,0.2,900)

                if 'concave' in detectionType:
                    # compute concave hull
                    from lib.CompGeom.ConcaveHull import ConcaveHull
                    concHull = ConcaveHull()
                    hull = concHull.concaveHull(pointsInGroup, 12)
                    plotPolygon(hull,False,'k',color,2,True,0.6,120)
                    color = next(colorIter)
                elif 'convex' in detectionType:
                    # compute convex hull
                    from lib.CompGeom.ConvexHull import ConvexHull
                    convHull = ConvexHull()
                    hull = convHull.convexHull(pointsInGroup)
                    plotPolygon(list(hull),False,'k',color,2,True,0.6,120)
                    color = next(colorIter)
                elif 'alpha' in detectionType:
                    from lib.CompGeom.AlphaShaper import AlphaShaper
                    alphaShaper = AlphaShaper(pointsInGroup)
                    hull = alphaShaper.alphaShapeAuto(1)
                    hullVerts = [pointsInGroup[p] for p in hull]
                    plotPolygon(hullVerts,False,'k',color,2,True,0.6,120)
                    color = next(colorIter)

                # plotEnd()

    def detectWayClusters(self):
        # Create spatial index (R-tree) of way sections
        spatialIndex = StaticSpatialIndex()
        spatialIndx2wayId = dict()  # Dictionary from index to Id
        boxes = dict()              # Bounding boxes of way-sections
        for Id, section in self.waySections.items():
            # exclude by some criteria
            if isEdgy(section.polyline):
                continue
            if section.polyline.length() < min(minTemplateLength,minNeighborLength):
                continue

            min_x = min(v[0] for v in section.polyline.verts)
            min_y = min(v[1] for v in section.polyline.verts)
            max_x = max(v[0] for v in section.polyline.verts)
            max_y = max(v[1] for v in section.polyline.verts)
            bbox = BBox(None,min_x,min_y,max_x,max_y)
            index = spatialIndex.add(min_x,min_y,max_x,max_y)
            spatialIndx2wayId[index] = Id
            bbox.index = index
            boxes[Id] = (min_x,min_y,max_x,max_y)
        spatialIndex.finish()

        self.waysForClusters = DisjointSets()
        # Use every way-section, that has been inserted into spatial index, 
        # as template and create a buffer around it.
        for Id, section in ((Id, section) for Id, section in self.waySections.items() if Id in boxes):
            # The template must have a minimum length.
            if section.polyline.length() > minTemplateLength:
                # Create buffer width width depending on category.
                bufferWidth = searchDist[section.originalSection.category]
                bufferPoly = section.polyline.buffer(bufferWidth,bufferWidth)

                # Create line clipper with this polygon.
                clipper = LinePolygonClipper(bufferPoly.verts)

                # Get neighbors of template way with its bounding box, expanded 
                # by the buffer width, from static spatial index .
                min_x,min_y,max_x,max_y = boxes[Id]
                results = stack = []
                neighbors = spatialIndex.query(min_x-bufferWidth,min_y-bufferWidth,
                                               max_x+bufferWidth,max_y+bufferWidth,results,stack)

                # Now test all these neighbors of the template for parallelism
                for neigborIndx in neighbors:
                    neighborID = spatialIndx2wayId[neigborIndx]

                    if neighborID == Id:
                        continue # Skip, the template is its own neighbor

                    # Check in table if pairing as cluster is allowed
                    neighborCategory = self.waySections[neighborID].originalSection.category
                    if not canPair(section.originalSection.category,neighborCategory):
                        continue

                    # If the polyline of this neighbor ...
                    neighborLine = self.waySections[neighborID].polyline

                    # ... is longer than a minimal length, ...
                    if neighborLine.length() > minNeighborLength:
                        # ... then clip it with the buffer polygon
                        inLine, inLineLength, nrOfON = clipper.clipLine(neighborLine.verts)

                        if inLineLength < 0.1:
                            continue # discard short inside lines

                        # To check the quality of parallelism, some kind of "slope" relative 
                        # to the template's line is evaluated.
                        p1, d1 = section.polyline.distTo(inLine[0][0])     # distance to start of inLine
                        p2, d2 = section.polyline.distTo(inLine[-1][-1])   # distance to end of inLine
                        p_dist = (p2-p1).length
                        slope = abs(d1-d2)/p_dist if p_dist else 1.

                        # Conditions for acceptable inside line.
                        acceptCond = slope < 0.15 and min(d1,d2) <= bufferWidth and \
                                     nrOfON <= 2 and inLineLength > bufferWidth/2
                        if acceptCond:
                            # Accept this pair as parallel.
                            self.waysForClusters.addSegment(Id,neighborID)

    def createLongClusterWays(self):
        for cIndx,wayIndxs in enumerate(self.waysForClusters):
            # Get the sections that belong to this cluster from the disjoint set.
            sections = [self.waySections[Id] for Id in wayIndxs]

            # Find their endpoints (which are ends of the cluster)
            # Note: way sections in a cluster may have common endpoints,
            # these are not yet detected here.
            sectionEnds = defaultdict(list)
            for i,section in enumerate(sections):
                s,t = section.originalSection.s, section.originalSection.t
                sV, tV = section.sV, section.tV
                Id = wayIndxs[i]
                # Store the section ID, start- and endpoint, the vertices of the polyline and the
                # direction vectors at start and end. The latter will be used to detect common endpoints.
                sectionEnds[s].append({'ID':Id,'start':s,'end':t,'verts':section.polyline.verts,      'startV':sV,'endV':-tV})
                sectionEnds[t].append({'ID':Id,'start':t,'end':s,'verts':section.polyline.verts[::-1],'startV':tV,'endV':-sV})
            # Entries in this dictionary, that have only one element (either start 's' or end 't' vertex of the
            # segment), are endpoints of the cluster.
            clusterEnds = [k for k, v in sectionEnds.items() if len(v) == 1]

            if not clusterEnds:
                continue 

            # What follows now is quite complicated. We try to connect the way sections, that belong to a cluster,
            # to lines, that may go through intersections along them.
            endpoints = clusterEnds.copy()
            lines = []
            intersectionsAll = []
            sectionsIDsAll = []
            inClusterIsects = []
            while endpoints:
                line = []
                intersectionsThis = []
                sectionsIDs = []                # Id's of the sections that belong to a line
                # Take an endpoint <ep> of the cluster and try to follow a line along its sections.
                ep = endpoints.pop()            # Endpoint to start with
                currSec = sectionEnds[ep][0]    # Section it belongs to
                while True:
                    # The reverted vertices of the current sections are part of the line.
                    line.extend(currSec['verts'][:-1])  
                    sectionsIDs.append(currSec['ID'])
                    end = currSec['end'] # The end of this section may be the start of a next section ...
                    if len(sectionEnds[end]) == 2: # ... if there are two common vertices in the dictionary ...
                        # we have (maybe) such a new section
                        newSec = [sec for sec in sectionEnds[end] if sec['end'] != currSec['start']][0]
                        if currSec['endV'].dot(newSec['startV']) < 0.:
                            # Does the new section point backwards?
                            # Then we have a common endpoint here that does not exist in <endpoints>.
                            # We end the current line here and start a new one (backwards).
                            # But remember this point as an additional possible cluster endpoint.
                            intersectionsThis.append(currSec['end'])
                            line.append(end)
                            inClusterIsects.append(end)
                            lines.append(PolyLine(line))
                            intersectionsAll.append(intersectionsThis)
                            sectionsIDsAll.append(sectionsIDs)
                            line = []
                            intersectionsThis = []
                            sectionsIDs = []
                            currSec = newSec
                        else:
                            intersectionsThis.append(currSec['end'])
                            currSec = newSec
                    else:
                        if end in endpoints:
                            endpoints.remove(end)
                        break
                line.append(end)    # Add the endpoint of the last section.
                lines.append(PolyLine(line))
                intersectionsAll.append(intersectionsThis)
                sectionsIDsAll.append(sectionsIDs)

            # In the next step, in-cluster intersections are processed. As a first
            # measure, ways, that have such intersections on both sides are removed.
            if inClusterIsects:
                toRemove = []
                for i,line in enumerate(lines):
                    if (line[0] in inClusterIsects or len(sectionEnds[line[0]]) > 2) and \
                       (line[-1] in inClusterIsects  or len(sectionEnds[line[-1]]) > 2):
                        toRemove.append(i)
                if toRemove:
                    # print('removed',toRemove)
                    for i in sorted(toRemove, reverse = True):
                        p0 = lines[i][0]
                        if p0 in inClusterIsects:
                            inClusterIsects.remove(p0)
                        for k,isects in enumerate(intersectionsAll):
                            if p0 in isects:
                                intersectionsAll[k].remove(p0)
                        p0 = lines[i][-1]
                        if p0 in inClusterIsects:
                            inClusterIsects.remove(p0)
                        for k,isects in enumerate(intersectionsAll):
                            if p0 in isects:
                                intersectionsAll[k].remove(p0)
                        del lines[i]
                        del intersectionsAll[i]
                        self.waysCoveredByCluster.extend(sectionsIDsAll[i])
                        del sectionsIDsAll[i]

            # All lines found should be ordered in the same direction. To realize this,
            # find the longest polyline and project the first and last vertex of every line
            # onto this reference polyline. The line parameter of the first vertex must
            # then be smaller than the one of the last vertex. Store the line parameters
            # for later use.
            referenceLine = max( (line for line in lines), key=lambda x: x.length() )
            params0 = []
            params1 = []
            for indx,line in enumerate(lines):
                # Project the first and last vertex of <line> onto reference. 
                # Smaller parameter <tx> means closer to start of reference.
                _,t0 = referenceLine.orthoProj(line[0])
                _,t1 = referenceLine.orthoProj(line[-1])
                if t0 > t1: # reverse this line, if True
                    lines[indx].toggleView()
                    sectionsIDsAll[indx] = sectionsIDsAll[indx][::-1]
                    intersectionsAll[indx] = intersectionsAll[indx][::-1]
                    params0.append(t1)
                    params1.append(t0)
                else:
                    params0.append(t0)
                    params1.append(t1)

            # Sometimes, there have been gaps left due to short way-sections.
            # During cluster detection, lines shorter than <minTemplateLength>
            # have been eliminated. We search now for gaps between start- and
            # end-points of lines, that are shorter than this value.
            # Brute force search, but number of lines in a cluster is small.
            perms = permutations(range(len(lines)),2)
            possibleGaps = []
            for i0,i1 in perms:
                if params0[i1] < params1[i0]: # prevent short cluster to be merged
                    d = (lines[i0][0]-lines[i1][-1]).length
                    if d < minTemplateLength:
                        possibleGaps.append( (i0,i1) )

            # Check for possible bifurcation at one end of the gap. These result in duplicates
            # either in i0s (multiple targets) or i1s (multiple sources)
            linesToRemove = []
            d0, d1 = defaultdict(list), defaultdict(list)
            for i, (i0,i1) in enumerate(possibleGaps):
                d0[i0].append(i)
                d1[i1].append(i)
            dupsS = { k:v for k, v in d0.items() if len(v) > 1 }   # duplicates of gap targets
            dupsE = { k:v for k, v in d1.items() if len(v) > 1 }   # duplicates of gap sources

            if dupsS:   # The gap has one source (end of line) and two targets (starts of lines)
                for source, targets in dupsS.items():
                    parallelity = [] 
                    for i,target in enumerate(targets):           
                        vGap = lines[target][0] - lines[source][-1]         # vector of gap
                        vGap /= vGap.length                                 # make unit vector
                        vTarget = lines[target][1] - lines[target][0]       # first vector of target
                        vTarget /= vTarget.length                           # make unit vector
                        parallelity.append( (i,abs(vGap.cross(vTarget))) )  # cross product == sine of vectors
                    removeIndx = targets[ max( (p for p in parallelity), key=lambda x: x[1] )[0] ]
                    linesToRemove.append(removeIndx)
                    self.waysCoveredByCluster.extend(sectionsIDsAll[removeIndx])
                    gapToRemove = (source,linesToRemove[-1])
                    possibleGaps.remove(gapToRemove)

            if dupsE:
                for target, sources in dupsE.items():
                    parallelity = [] 
                    for i,source in enumerate(sources):           
                        vGap = lines[target][0] - lines[source][-1]         # vector of gap
                        vGap /= vGap.length                                 # make unit vector
                        vTarget = lines[target][1] - lines[target][0]       # first vector of target
                        vTarget /= vTarget.length                           # make unit vector
                        parallelity.append( (i,abs(vGap.cross(vTarget))) )  # cross product == sine of vectors
                    removeIndx = sources[ max( (p for p in parallelity), key=lambda x: x[1] )[0] ]
                    linesToRemove.append(removeIndx)
                    self.waysCoveredByCluster.extend(sectionsIDsAll[removeIndx])
                    gapToRemove = (linesToRemove[-1],target)
                    possibleGaps.remove(gapToRemove)

            # If there are such gaps, the corresponding lines are merged, if there
            # exists a real way-section that can fill this gap.
            if possibleGaps:
                # In case of bifurcations, we connect only to one line. Clean
                # possible gaps by removing duplicates in first or second index.
                i0s, i1s = [], []
                cleanedPossibleGaps = []
                for i0,i1 in possibleGaps:
                    if i0 not in i0s and i1 not in i1s:
                        cleanedPossibleGaps.append( (i0,i1) )
                        i0s.append(i0)
                        i1s.append(i1)

                # Merge lines through gaps.
                for i0,i1 in cleanedPossibleGaps:
                    # Vertices of lines i0 and i1 become gap vertices
                    gap0, gap1 = lines[i0][0], lines[i1][-1]
                    # If gap vertices 
                    # if i0Start in self.sectionNetwork and i1End in self.sectionNetwork:
                    if gap1 in self.sectionNetwork[gap0]:
                        sectionId = self.sectionNetwork[gap0][gap1][0].sectionId
                    elif gap0 in self.sectionNetwork[gap1]:
                        sectionId = self.sectionNetwork[gap1][gap0][0].sectionId
                    else:
                        print('Problem',i0,i1,cIndx)
                        continue

                    # Merge lines through gap
                    lines[i1] = PolyLine( lines[i1][:] + lines[i0][:])
                    sectionsIDsAll[i1] += [sectionId] + sectionsIDsAll[i0]
                    intersectionsAll[i1].extend(intersectionsAll[i0] + [gap0, gap1])
                    linesToRemove.append(i0)

                # Book-keeping, remove merged parts.
                if linesToRemove:
                    for index in sorted(linesToRemove, reverse=True):
                        del lines[index]                    
                        del intersectionsAll[index]
                        del sectionsIDsAll[index]

            # We like to have the start points close together. The line ends that have a
            # smaller median absolute deviation are selected as start points.
            params0 = [referenceLine.orthoProj(line[0])[1] for line in lines]
            params1 = [referenceLine.orthoProj(line[-1])[1] for line in lines]
            mad0 = mad(params0)
            mad1 = mad(params1)
            # But in-cluster intersections are only allowed at the end of the cluster.
            # They will be corrected by the 'clip parallel cluster' process.
            endPoints = [line[-1] for line in lines]
            selfIsectAtEnd = len(set(endPoints)) != len(endPoints)
            startPoints = [line[0] for line in lines]
            selfIsectAtStart = len(set(startPoints)) != len(startPoints)
            if (mad1 < mad0 and not selfIsectAtEnd) or selfIsectAtStart:
                for indx,line in enumerate(lines):
                    lines[indx].toggleView()
                    sectionsIDsAll[indx] = sectionsIDsAll[indx][::-1]
                    intersectionsAll[indx] = intersectionsAll[indx][::-1]

            # As last step, the lines have to be ordered from left to right.
            # The first segment of an arbitrary line is turned perpendicularly
            # to the right and all startpoints are projected onto this vector,
            # delivering a line parameter t. The lines are then sorted by 
            # increasing parameter values.
            vec = lines[0][1] - lines[0][0] # first segment
            p0 = lines[0][0]
            perp = Vector((vec[1],-vec[0])) # perpendicular to the right
            # Sort by projection order
            sortedIndices = sorted( (i for i in range(len(lines))), key=lambda i: (lines[i][0]-p0).dot(perp) )
            lines = [lines[i] for i in sortedIndices]
            intersectionsAll = [intersectionsAll[i] for i in sortedIndices]
            sectionsIDsAll = [sectionsIDsAll[i] for i in sortedIndices]

            # Check ratio of endpoînt distances of outermost lines to decide
            # whether a check for parallelity is required.
            left = lines[0]
            right = lines[-1]
            d0 = (left[0]-right[0]).length
            d1 = (left[-1]-right[-1]).length
            ratio = min(d0,d1)/max(d0,d1)

            # check cluster for parallelity, if too asymetric ends.
            clipped = False
            if ratio < 0.6:
                lines, clipped = clipParallelPart(lines)
                # Clean eventual intersections or way-sections, that have been clipped.
                if clipped == 'clipL':
                    intersectionsAll[0] = [isect for isect in intersectionsAll[0] if isect in lines[0]]
                    sectionsIDsAll[0] = [Id for Id in sectionsIDsAll[0] if self.waySections[Id].polyline[0] in lines[0] or \
                                         self.waySections[Id].polyline[-1] in lines[0] ]
                elif clipped == 'clipR':
                    intersectionsAll[-1] = [isect for isect in intersectionsAll[-1] if isect in lines[-1]]
                    sectionsIDsAll[-1] = [Id for Id in sectionsIDsAll[-1] if self.waySections[Id].polyline[0] in lines[-1] or \
                                         self.waySections[Id].polyline[-1] in lines[-1] ]
                elif clipped == 'clipB':
                    for i in range(len(lines)):
                        intersectionsAll[i] = [isect for isect in intersectionsAll[i] if isect in lines[i]]
                        sectionsIDsAll[i] = [Id for Id in sectionsIDsAll[i] if self.waySections[Id].polyline[0] in lines[i] or \
                                            self.waySections[Id].polyline[-1] in lines[i] ]
                else:
                    print('????')

            # Create and simplify cluster centerline
            centerline = centerlineOf(lines[0][:],lines[-1][:])
            centerline = simplifyRDP(centerline,0.1)
            centerline = simplifyEnds(centerline,10.)
            centerline = PolyLine( centerline)
 
            # If there are inner lines in clusters that are much shorter than the
            # outer lines, we remove them for now.
            if len(lines)>2:
                # We will need distances on the centerline, therefore an interpolator from 
                # parameters <t> to distance <d> is required.                
                ts = [i for i in range(len(centerline.verts))] # <t> along PolyLine (see there)
                ds = list( accumulate([0]+[(v2-v1).length for v1,v2 in pairs(centerline)]) ) # cumulated distance <d>
                interp_t2d = LinearInterpolator(ts,ds)
                linesToRemove = set()
                for indx,line in enumerate(lines[1:-1]):
                    # check start
                    _,t = centerline.orthoProj(line[0])
                    if abs(interp_t2d(t)) > 20.: # If start of centerline > 20 m away: remove this line
                        linesToRemove.add(indx+1)
                    # check end
                    _,t = centerline.orthoProj(line[-1])
                    if abs(centerline.length() - interp_t2d(t)) > 20.: # If end of centerline > 20 m away: remove this line
                        linesToRemove.add(indx+1)
                if linesToRemove:
                    for index in sorted(list(linesToRemove), reverse=True):
                        del lines[index]                    
                        del intersectionsAll[index]
                        self.waysCoveredByCluster.extend(sectionsIDsAll[index])
                        del sectionsIDsAll[index]

            # As a last book-keeping. Some ways have not been detected as cluster ways, because they
            # have not been parallel, but they are covered by the cluster. They must be invalidated
            # and their intersections removed.

            # First find all true intersections (including line ends) and the line ends that have
            # been clipped. Create a map <isect2line> with all line numbers that contain an intersection.
            intersections = []
            lineEnds = []
            isect2line = defaultdict(set)
            for i in range(len(lines)):
                # True intersections in lines.
                for isect in intersectionsAll[i]:
                    isect2line[isect].add(i)
                    intersections.append(isect)
                # Line ends ...
                for isect in [lines[i][0],lines[i][-1]]:
                    # ... that are true intersections
                    if isect in self.sectionNetwork:
                        isect2line[isect].add(i)
                        intersections.append(isect)
                    # ... or that are clipped
                    else:
                        lineEnds.append(isect)

            # Create the polygon around the cluster, that covers all ways inside.
            clusterPoly = lines[0][:]
            clusterPoly.extend([line[-1] for line in lines[1:-1]])
            clusterPoly.extend(lines[-1][::-1])
            clusterPoly.extend([line[0] for line in lines[1:-1][::-1]])
            clusterPolyClipper = LinePolygonClipper(clusterPoly)

            for node in intersections:
                # for every intersection node, find its outgoing section
                for seg in self.sectionNetwork.iterOutSegments(node):
                    if seg.category=='scene_border':
                        continue
                    # Classify this section
                    # seg.t is a line end.
                    isLineEnd = seg.t in isect2line
                    # The end-points of the segment are on different cluster lines
                    differentLines = isLineEnd and not isect2line[seg.t].intersection(isect2line[node])
                    # The end of the segment is in the cluster polygon, but not a line end
                    polyVertex = not isLineEnd and pointInPolygon(clusterPoly,seg.t) == 'IN' and seg.t not in lineEnds
                    # Most part of the segment is within the cluster polygon
                    covered = clusterPolyClipper.clipLine(seg.path)[1] > 0.5*seg.length

                    if isLineEnd:
                      if differentLines and covered:
                        # It is something like a diagonal. Invalidate it.
                        self.waysCoveredByCluster.append(seg.sectionId)
                        # ... and remove its intersections.
                        if self.sectionNetwork.borderlessOrder(node) < 4:
                            lineIndx = list(isect2line[node])[0]
                            if node in intersectionsAll[lineIndx]:
                                intersectionsAll[lineIndx].remove(node)
                        if self.sectionNetwork.borderlessOrder(seg.t) < 4:
                            lineIndx = list(isect2line[seg.t])[0]
                            if seg.t in intersectionsAll[lineIndx]:
                                intersectionsAll[lineIndx].remove(seg.t)
                    elif polyVertex:
                        # It starts a dangling way and must be invalidated.
                        self.waysCoveredByCluster.append(seg.sectionId)
                        # ... and remove its node
                        if self.sectionNetwork.borderlessOrder(node) < 4:
                            lineIndx = list(isect2line[node])[0]
                            if node in intersectionsAll[lineIndx]:
                                intersectionsAll[lineIndx].remove(node)
                    elif not clipped and covered:
                        # it intersects the polygon and has a significant length inside the polygon,
                        # but is not a clipped cluster end. Invalidate it
                        self.waysCoveredByCluster.append(seg.sectionId)
                        # ... and remove its node
                        if self.sectionNetwork.borderlessOrder(node) < 4:
                            lineIndx = list(isect2line[node])[0]
                            if node in intersectionsAll[lineIndx]:
                                intersectionsAll[lineIndx].remove(node)

            # longCluster = LongClusterWay(self)
            # for i in range(len(lines)):
            #     longCluster.sectionIDs.append(sectionsIDsAll[i])
            #     longCluster.intersections.append(intersectionsAll[i])
            #     longCluster.polylines.append(lines[i])
            #     longCluster.centerline = centerline
            #     longCluster.clipped= clipped

            # for Id in self.waysCoveredByCluster:
            #     self.waySections[Id].isValid = False

            # # Split the long cluster in smaller subclusters of class <SubCluster>,
            # # stored in the instance of <longCluster>.
            # longCluster.split()

            # self.longClusterWays.append(longCluster)

    def cleanCoveredWays(self):
        # This process removes all way-sections, that are by whatever reason
        # completely covered by cluster areas.

        # Create spatial index (R-tree) of all valid way sections
        spatialAllWayIndex = StaticSpatialIndex()
        allSpatialIndx2wayId = dict()
        boxes = dict()      # Bounding boxes of way-sections
        for Id, section in self.waySections.items():
            if section.isValid and Id not in self.waysCoveredByCluster:
                min_x = min(v[0] for v in section.polyline.verts)
                min_y = min(v[1] for v in section.polyline.verts)
                max_x = max(v[0] for v in section.polyline.verts)
                max_y = max(v[1] for v in section.polyline.verts)
                index = spatialAllWayIndex.add(min_x,min_y,max_x,max_y)
                allSpatialIndx2wayId[index] = Id
                bbox = BBox(None,min_x,min_y,max_x,max_y)
                bbox.index = index
                boxes[Id] = (min_x,min_y,max_x,max_y)
        spatialAllWayIndex.finish()

        # Find the IDs of all ways that have end-points covered by a way-cluster.
        # At the same time, prepare a spatial index for the sub-clusters.
        startEndIDs = set()
        stopEndIDs = set()
        spatialAllClustIndex = StaticSpatialIndex()
        allSpatialIndx2Poly = dict()
        for longCluster in self.longClusterWays:
            if longCluster.clipped:
                continue
            for subCluster in longCluster.subClusters:
                if not subCluster.valid:
                    continue
                # create polygon of this subcluster, its bounding box and add that
                # to the spatial index.
                trimmedCenterline = subCluster.centerline.trimmed(subCluster.trimS,subCluster.trimT)
                poly = trimmedCenterline.buffer(subCluster.outWL(),subCluster.outWR())
                min_x = min(v[0] for v in poly)
                min_y = min(v[1] for v in poly)
                max_x = max(v[0] for v in poly)
                max_y = max(v[1] for v in poly)
                index = spatialAllClustIndex.add(min_x,min_y,max_x,max_y)
                allSpatialIndx2Poly[index] = poly

                # For every neighbor way of this sub-cluster, find those that have
                # either a start- or an end-point in the cluster area.
                results = stack = []
                neighborWayIndices = spatialAllWayIndex.query(min_x,min_y,max_x,max_y,results,stack)
                for wayIndx in neighborWayIndices:
                    wayID = allSpatialIndx2wayId[wayIndx]
                    section = self.waySections[wayID]
                    if section.isValid:
                        startPointInPoly = pointInPolygon(poly,section.polyline[0]) == 'IN'
                        endPointInPoly = pointInPolygon(poly,section.polyline[-1]) == 'IN'

                        # Store its ID if in polygon.
                        if startPointInPoly:
                            startEndIDs.add(wayID)
                        if endPointInPoly:
                            stopEndIDs.add(wayID)

        spatialAllClustIndex.finish()

        # For way-sections, that have both ends in a cluster area, we need 
        # to check if all their vertices are in one of the cluster areas.
        # Brute force, very costly, but there are only few such ways.
        bothEndIDs = startEndIDs.intersection(stopEndIDs)
        for Id in bothEndIDs:
            isInCluster = True
            for v in self.waySections[Id].polyline:
                results = stack = []
                neighborPolyIndices = spatialAllClustIndex.query(v[0],v[1],v[0]+0.001,v[1]+0.001,results,stack)
                polygons = [allSpatialIndx2Poly[indx] for indx in neighborPolyIndices]
                if all(pointInPolygon(poly,v) != 'IN' for poly in polygons):
                    isInCluster = False
                    break

            # Clean this way, it is completely covered by cluster areas.
            if isInCluster:
                self.waySections[Id].isValid = False

        # There are some stubborn ways where a long part, but not the
        # whole way, is inside a way cluster. There is nothing left but
        # to find the total length 'within' the cluster and set a limit.
        # Example: milano_01.osm.
        for Id, section in self.waySections.items():
            if section.isValid:
                min_x,min_y,max_x,max_y = boxes[Id]
                results = stack = []
                neighborPolyIndices = spatialAllClustIndex.query(min_x,min_y,max_x,max_y,results,stack)
                polygons = [allSpatialIndx2Poly[indx] for indx in neighborPolyIndices]
                lenSum = 0.
                for poly in polygons:
                    clipper = LinePolygonClipper(poly)
                    _, totalLength, _ = clipper.clipLine(section.polyline)
                    lenSum += totalLength
                if lenSum > 30:
                    self.waySections[Id].isValid = False


    def createClusterTransitionAreas(self):
        areas = []
        for longClusterWay in self.longClusterWays:
            if len(longClusterWay.subClusters) > 1:
                blockedWayIDs = longClusterWay.sectionIDs[0] + longClusterWay.sectionIDs[-1]
                skipSmall = False
                for clust1Nr,(cluster1,cluster2) in enumerate(pairs(longClusterWay.subClusters)):
                    if skipSmall: # Skip short cluster transition already handled by createMultiTransition()
                        skipSmall = False
                        continue
                    if cluster2.length < 3.:   # Short cluster requires special treatement by createMultiTransition
                        skipSmall = True
                        area, wayConnectors, clustConnectors = createShortClusterIntersection(self,cluster1,cluster2,longClusterWay.subClusters[clust1Nr+2],blockedWayIDs)
                    else:
                        clustConnectors = dict()
                        wayConnectors = dict()
                        if cluster1.endSplit.type == 'left':
                            node = cluster1.endSplit.posL.freeze()
                            order = self.sectionNetwork.borderlessOrder(node)
                            if order == 2:
                                area, clustConnectors = createLeftTransition(self,cluster1,cluster2)
                            else:
                                area, clustConnectors, wayConnectors = createLeftIntersection(self,cluster1,cluster2,node,blockedWayIDs)
                        elif cluster1.endSplit.type == 'right':
                            node = cluster1.endSplit.posR.freeze()
                            order = self.sectionNetwork.borderlessOrder(node)
                            if order == 2:
                                area, clustConnectors = createRightTransition(self,cluster1,cluster2)
                            else:
                                area, clustConnectors, wayConnectors = createRightIntersection(self,cluster1,cluster2,node,blockedWayIDs)
                        elif cluster1.endSplit.type == 'inner':
                            node = cluster1.endSplit.posR.freeze()
                            order = self.sectionNetwork.borderlessOrder(node)
                            if order == 2:
                                area, clustConnectors = createRightTransition(self,cluster1,cluster2)

                    # Create the final cluster area instance
                    isectArea = IntersectionArea()
                    isectArea.polygon = area
                    isectArea.connectors = wayConnectors
                    isectArea.clusterConns = clustConnectors
                    self.intersectionAreas.append(isectArea)

                    # if cluster1.endSplit.type == 'inner':
                    #     plotPureNetwork(self.sectionNetwork)
                    #     poly = cluster1.centerline.buffer(cluster1.width/2.,cluster1.width/2.)
                    #     plotPolygon(poly,False,'g','g',1.,True,0.3,120)
                    #     poly = cluster2.centerline.buffer(cluster2.width/2.,cluster2.width/2.)
                    #     plotPolygon(poly,False,'b','b',1.,True,0.3,120)
                    #     plotPolygon(area,False,'r','r',1)
                    #     plotEnd()
                    #     plt.close()

                    areas.append(area)
            else:   # only one cluster way
                pass

    def createClusterIntersections(self):
        # Find groups of neighboring endpoints of cluster ways,
        # using density based scan
        endPoints = []
        for longClusterWay in self.longClusterWays:
            start = longClusterWay.subClusters[0]
            end = longClusterWay.subClusters[-1]
            endPoints.append( (start.centerline[0],start,'start',longClusterWay.id) )
            if not longClusterWay.clipped:
                endPoints.append( (end.centerline[-1],end,'end',longClusterWay.id) )
        clusterGroups = dbClusterScan(endPoints, dbScanDist, 2)

        # <clusterGroups> is a list that contains lists of clusterways, where their
        # endpoints are neighbors. These will form intersection clusters, probably 
        # with additional outgoing way-sections. A list entry for a clusterway
        # is a <clusterGroup> and is formed as
        # [centerline-endpoint, cluster-way, type ('start' or 'end')]

        # Create an intersection cluster area for every <clusterGroup>
        for cnt,clusterGroup in enumerate(clusterGroups):
            # Sometimes there are mor than 4 cluster meeting at an intersection. As
            # a heuristic, we assume that there is a cluster in the center of the
            # intersection (example: osm_extracts/streets/taipei.osm). Then, start 
            # and end of the same cluster are in the <clusterGroup>. Remove this
            # cluster.
            if len(clusterGroup) > 4:
                # Find indices of duplicates using defaultdict
                indxOfIDs = defaultdict(list)
                for indx,cluster in enumerate(clusterGroup):
                    indxOfIDs[cluster[1].id].append(indx)
                for indices in indxOfIDs.values():
                    if len(indices) > 1:
                        clusterGroup[indices[0]][1].valid = False   # invalidate this subcluster
                        for indx in sorted(indices,reverse=True):   # remove entries in clusterGroup
                            del clusterGroup[indx]

            # If there is only one cluster in the group, we have an cluster end.
            # Because clusters can only be clipped at the end and are excluded by
            # the loop above, they don't appear here. Ends of clusters have require
            # a special handling, if they have overlapping outways.           
            if len(clusterGroup) == 1:
                if not clusterGroup[0][1].longCluster.clipped:
                    endwayHandler = EndWayHandler(self,clusterGroup[0][1], clusterGroup[0][2])
                    if endwayHandler.hasValidWays:
                        isectArea = IntersectionArea()
                        isectArea.polygon = endwayHandler.area
                        isectArea.connectors = endwayHandler.wayConnectors
                        isectArea.clusterConns = endwayHandler.clustConnectors
                        self.intersectionAreas.append(isectArea)
                    continue

            # Create an instance of <IntersectionCluster>, which will form the
            # cluster area and its connectors.
            isectCluster = IntersectionCluster()

            # Collect all intersection nodes of the way-sections at the cluster.
            # TODO Possibly there are more nodes within the cluster area.           
            nodes = set()
            for cluster in clusterGroup:
                if cluster[2] == 'start':
                    nodes.update( cluster[1].startSplit.posW )
                elif cluster[2] == 'end':
                    nodes.update( cluster[1].endSplit.posW )
                else:
                    assert False, 'Should not happen'

            # Insert centerlines and widths of clusters to <IntersectionCluster>. Keep a map
            # <ID2Object> for the ID in <IntersectionCluster> to the inserted object.
            # Collect way IDs (key of <self.waySections>) so that the cluster ways may later
            # be excluded from outgoing ways at end-nodes.
            wayIDs = []
            ID2Object = dict()
            for cluster in clusterGroup:
                if cluster[2] == 'start':
                    wayIDs.extend( cluster[1].startSplit.currWIds )
                    Id = isectCluster.addWay(cluster[1].centerline,cluster[1].outWL(),cluster[1].outWR())
                    ID2Object[Id] = cluster
                elif cluster[2] == 'end':
                    wayIDs.extend( cluster[1].endSplit.currWIds )
                    Id = isectCluster.addWay(PolyLine(cluster[1].centerline[::-1]),cluster[1].outWR(),cluster[1].outWL())
                    ID2Object[Id] = cluster
                else:
                    assert False, 'Should not happen'

            # If possible, create polygon from these nodes.
            nodePoly = None
            if len(nodes) > 3:
                nodePoly = orderAsPolygon(nodes)

            # Find all outgoing way-sections that are connected to cluster nodes, but exlude
            # those that belong to the clusters, using <wayIDs>. Insert them together with their
            # widths into <IntersectionCluster>. Extend the map <ID2Object> for the ID in
            # <IntersectionCluster> to the inserted object.
            for node in nodes:
                if node in self.sectionNetwork:
                    for net_section in self.sectionNetwork.iterOutSegments(node):
                        if net_section.category != 'scene_border':
                            if net_section.sectionId not in wayIDs:
                                section = self.waySections[net_section.sectionId]
                                # If one of the section ends is not a node
                                if not (net_section.s in nodes and net_section.t in nodes): 
                                    # If this section end is inside a cluster area, eliminate this section.
                                    endOfSection = section.originalSection.t if section.originalSection.s == node else section.originalSection.s
                                    if any(cluster[1].pointInCluster(endOfSection) for cluster in clusterGroup):
                                        section.isValid = False  
                                        self.processedNodes.add(endOfSection.freeze())
                                        continue   
                                    # If this section end is inside the polygon built by the end nodes of
                                    # the clusters, , eliminate this section.
                                    if nodePoly and pointInPolygon(nodePoly,endOfSection) in ['IN','ON']:
                                        section.isValid = False  
                                        self.processedNodes.add(endOfSection.freeze())
                                        continue   

                                    # Add this section to the intersection cluster.
                                    if section.originalSection.s == node:
                                        Id = isectCluster.addWay(section.polyline,section.leftWidth,section.rightWidth)
                                        ID2Object[Id] = (section,True)  # forward = True
                                    else:
                                        Id = isectCluster.addWay(PolyLine(section.polyline[::-1]),section.rightWidth,section.leftWidth)
                                        ID2Object[Id] = (section,False) # forward = False
                                else:
                                    # It's a way between cluster nodes, so invalidate it. This should not
                                    # be done at cluster-edns, but these are already excluded here.
                                    section.isValid = False

            # An intersection area can only be constructed, when more than one way present.
            if len(isectCluster.outWays) > 1:
                # Create the intersection area and their connectors5 
                area, connectors = isectCluster.create()

                # Transfer the trim values to the objects
                for outWay in isectCluster.outWays:
                    object = ID2Object[outWay.id]
                    if isinstance(object[0],WaySection):
                        if object[1]:   # if forward
                            object[0].trimS = max(object[0].trimS,outWay.trim_t)
                        else:
                            t = len(object[0].polyline)-1 - outWay.trim_t
                            object[0].trimT = min(object[0].trimT, t)
                    else:   # it's a cluster way
                        if object[2] == 'start':
                            object[1].trimS = max(object[1].trimS,outWay.trim_t)
                        else:
                            t = len(object[1].centerline)-1 - outWay.trim_t
                            object[1].trimT = min(object[1].trimT,t)

                # Create connectors for this area.
                clustConnectors = dict()
                wayConnectors = dict()
                conOffset = 0
                for con in connectors:
                    object = ID2Object[con[2]]
                    if isinstance(object[0],WaySection):
                        Id = object[0].id if object[1] else -object[0].id
                        wayConnectors[Id] = con[0]+conOffset
                    else:
                        if object[2] == object[1].transitionPos:
                            if object[2] == 'start':
                                transitionLength = object[1].transitionWidth/transitionSlope
                                dTrans = transitionLength
                                tTrans = object[1].centerline.d2t(dTrans)
                                if tTrans > object[1].trimS:
                                    p1 = object[1].centerline.offsetPointAt(tTrans,-object[1].outWR())
                                    p2 = object[1].centerline.offsetPointAt(tTrans,object[1].outWL())
                                    object[1].trimS = max(object[1].trimS,tTrans)
                                    area.insert(con[0]+conOffset+1,p1)
                                    clustConnectors[object[1].id] = con[0]+conOffset+1
                                    area.insert(con[0]+conOffset+2,p2)
                                    conOffset += 2
                                else:
                                    clustConnectors[object[1].id] = con[0]+conOffset
                            else:   # object[2] == 'end'
                                transitionLength = object[1].transitionWidth/transitionSlope
                                dTrans = object[1].centerline.length() - transitionLength
                                tTrans = object[1].centerline.d2t(dTrans)
                                if tTrans < object[1].trimT:
                                    p1 = object[1].centerline.offsetPointAt(tTrans,object[1].outWL())
                                    p2 = object[1].centerline.offsetPointAt(tTrans,-object[1].outWR())
                                    object[1].trimT = min(object[1].trimT,tTrans)
                                    area.insert(con[0]+conOffset+1,p1)
                                    clustConnectors[-object[1].id] = con[0]+conOffset+1
                                    area.insert(con[0]+conOffset+2,p2)
                                    conOffset += 2
                                else:
                                    clustConnectors[-object[1].id] = con[0]+conOffset
                        else:
                            Id = object[1].id if object[2]=='start' else -object[1].id
                            clustConnectors[Id] = con[0]+conOffset

                isectArea = IntersectionArea()
                isectArea.polygon = area
                isectArea.connectors = wayConnectors
                isectArea.clusterConns = clustConnectors
                self.intersectionAreas.append(isectArea)

    def createClippedClusterEnds(self):
        for longClusterWay in self.longClusterWays:
            if longClusterWay.clipped:
                endsubCluster = longClusterWay.subClusters[-1]
                area, clustConnectors, wayConnectors = createClippedEndArea(self,longClusterWay,endsubCluster)
                isectArea = IntersectionArea()
                isectArea.polygon = area
                isectArea.connectors = wayConnectors
                isectArea.clusterConns = clustConnectors
                self.intersectionAreas.append(isectArea)

    def createWayClusters(self):
        # Collect connection ids to set the connection ends.
        startConnections = set()
        endConnections = set()
        for area in self.intersectionAreas:
            for conId,con in area.clusterConns.items():
                if conId>0:
                    startConnections.add(abs(conId))
                else:
                    endConnections.add(abs(conId))

        for longCluster in self.longClusterWays:
            for cluster in longCluster.subClusters:
                if not cluster.valid:
                    continue
                # Special case first. The intersection areas on both sides of the
                # cluster overlap. These areas are merged and the cluster is invalidated.
                if cluster.trimS >= cluster.trimT:
                    cluster.valid = False
                    # Find and merge conflicting areas
                    conflcitingIndxsAreas = [(indx,area) for indx,area in enumerate(self.intersectionAreas) for conId,_ in area.clusterConns.items() if abs(conId) == cluster.id]
                    conflictingIndices = [ia[0] for ia in conflcitingIndxsAreas]
                    conflcitingAreas = [ia[1] for ia in conflcitingIndxsAreas]
                    try:
                        ret = boolPolyOp(conflcitingAreas[0].polygon,conflcitingAreas[1].polygon,'union')
                    except Exception:
                        print('Problem')
                        break
                    mergedPoly = ret[0]
                    # The merged poygon is now in <mergedPoly>. Be sure that it 
                    # is ordered counter-clockwise.
                    area = sum( (p2[0]-p1[0])*(p2[1]+p1[1]) for p1,p2 in zip(mergedPoly,mergedPoly[1:]+[mergedPoly[0]]))
                    if area > 0.:
                        mergedPoly.reverse()
                    # An instance of IntersectionArea is now constructed with its remaining connectors.
                    mergedArea = IntersectionArea()
                    mergedArea.polygon = mergedPoly

                    # Create connectors for merged area
                    for conflictingArea in conflcitingAreas:
                        connectors = conflictingArea.connectors
                        for signedKey,connector in connectors.items():
                            v0 = conflictingArea.polygon[connector]
                            newConnector = mergedPoly.index(v0)
                            mergedArea.connectors[signedKey] = newConnector

                        clusterConns = conflictingArea.clusterConns
                        for signedKey,connector in clusterConns.items():
                            # Keep connectors that are not for the conflicting cluster
                            if abs(signedKey) != cluster.id:
                                v0 = conflictingArea.polygon[connector]
                                newConnector = mergedPoly.index(v0)
                                mergedArea.clusterConns[signedKey] = newConnector

                    # Now remove conflicting areas from list and add the merged area
                    for i in sorted(conflictingIndices, reverse = True):
                            del self.intersectionAreas[i]
                    self.intersectionAreas.append(mergedArea)
                
                # Normal case. The cluster way is constructed.
                else:
                    wayCluster = WayCluster()
                    wayCluster.centerline = cluster.centerline.trimmed(cluster.trimS,cluster.trimT)[:]
                    wayCluster.distToLeft = cluster.width/2.

                    for lineNr,wayID in enumerate(cluster.wayIDs):
                        section = self.waySections[wayID]
                        fwd = section.originalSection.s == cluster.startSplit.posL
                        if section.isOneWay:
                            nrOfLanes = (section.nrRightLanes)
                        else:
                            nrOfLanes = (section.nrLeftLanes, section.nrRightLanes) if fwd else (section.nrRightLanes, section.nrLeftLanes)
                        if lineNr == 0:
                            offset = -cluster.width/2.
                        elif lineNr == len(cluster.wayIDs)-1:
                            offset = cluster.width/2.
                        else:
                            v = cluster.centerline[1] - cluster.centerline[0]
                            v = Vector((-v[1],v[0])) / v.length
                            p = cluster.startSplit.posW[lineNr]
                            offset = -v.dot(p-cluster.centerline[0])

                        wayCluster.waySections.append(
                            createStreetSection(
                                offset,
                                section.rightWidth+section.leftWidth,
                                nrOfLanes,
                                section.originalSection.category,
                                section.originalSection.tags
                            )
                        )
                        
                    # Set connection types
                    wayCluster.startConnected = True if cluster.id in startConnections else False
                    wayCluster.endConnected = True if cluster.id in endConnections else False
                    self.wayClusters[cluster.id] = wayCluster

                # Some bookkeeping
                # TODO expand this for multiple ways
                # for node in cluster.startPoints[1:3]:
                for node in [cluster.startSplit.posL,cluster.startSplit.posR]:
                    self.processedNodes.add(node.freeze())
                # for node in cluster.endPoints[1:3]:
                for node in [cluster.endSplit.posL,cluster.endSplit.posR]:
                    self.processedNodes.add(node.freeze())

            # Cleanup cluster ways, except clipped ones
            for wayIDs in longCluster.sectionIDs:
                for wayID in wayIDs:
                    section = self.waySections[wayID]
                    if not section.isClipped:
                        section.isValid = False

            for wayIDs in longCluster.sectionIDs:
                section = self.waySections[wayIDs[-1]]
                if not section.isClipped:
                    section.isValid = False

        # Remove ways that have been covered by the cluster
        for wayID in self.waysCoveredByCluster:
            self.waySections[wayID].isValid=False

    def createSymSideLanes(self):
        nr = 0
        toReplace = []
        for location, intersection in self.waymap.iterNodes(Intersection):
            if intersection.order == 2:
                section0 = intersection.leaveWays[0].section
                section1 = intersection.leaveWays[1].section
                streetIds = (intersection.leaveWays[0].street.id, intersection.leaveWays[1].street.id)
                incoming, outgoing = (section0, section1) if section0.totalLanes < section1.totalLanes else (section1, section0)
                hasTurns = bool( re.search(r'[^N]', outgoing.lanePatterns[0] ) )
                if hasTurns:    # it's a side lane
                    sideLane = SideLane(location, incoming, outgoing)
                    street = Street(incoming.src, outgoing.dst)
                    street.append(incoming)
                    street.append(sideLane)
                    street.append(outgoing)
                    section0.street = street
                    sideLane.street = street
                    section1.street = street
                    toReplace.append( (location, street, streetIds) )
                    self.transitionSideLanes.append(sideLane)
                else:   # it's a sym lane
                    # the above definition does not hold for SymLanes
                    incoming, outgoing = (section0, section1) if section0.dst == section1.src else (section1, section0)
                    symLane = SymLane(location, incoming, outgoing)

                    street = Street(incoming.src, outgoing.dst)
                    street.append(incoming)
                    street.append(symLane)
                    street.append(outgoing)
                    section0.street = street
                    symLane.street = street
                    section1.street = street
                    toReplace.append( (location, street, streetIds) )
                    self.transitionSymLanes.append(symLane)
            nr += 1

        for location, street, streetIds in toReplace:
            # At this stage, intersections do not yet have connectors. They will get them 
            # when processIntersection() is called (which occurs at self.updateIntersections).
            # But the leaving ways structure of the intersections at the end of the new
            # Street needs to be updated.
            predIsect = self.waymap.getStreetNode(street.src)['object']
            for way in predIsect.leaveWays:
                if way.street.id in streetIds:
                    way.street = street
            
            succIsect = self.waymap.getStreetNode(street.dst)['object']
            for way in succIsect.leaveWays:
                if way.street.id in streetIds:
                    way.street = street

            self.waymap.removeStreetNode(location)
            street.style = self.styleStore.get( self.getStyle(street) )
            street.setStyleBlockFromTop(street.style)
            self.waymap.addSection(street)

    def updateIntersections(self):
        # At this stage, it is assumed, that SideLanes, SymLanes and clustered intersections are already built
        # and that their nodes are stored in <self.processedNodes>.
        nr = 0
        for location, intersection in self.waymap.iterNodes(Intersection):
            if location in self.processedNodes:
                continue

            # # DEBUG: Show clusters of parallel way-sections.
            # # The plotting functions for this debug part are at the end of this module
            # if False and self.app.type == AppType.commandLine:
            #     from debug import plt, plotPolygon, plotEnd
            #     plt.close()
            #     for i,way in enumerate(intersection.leaveWays):
            #         plotPolygon(way.polygon,False,'k')
            #         p = way.polyline[-1]
            #         plt.text(p[0],p[1],str(i))
            #     plt.title(str(nr))
            #     plotEnd()

            # # TODO treat somehow short ways, this here does not work
            shortWays = intersection.cleanShortWays(False)
            if shortWays:
                for way in shortWays:
                    way.section.valid = False

            intersection.processIntersection()
            nr += 1

            # # DEBUG: Show intersections.
            # # The plotting functions for this debug part are at the end of this module
            # if False and self.app.type == AppType.commandLine:
            #     from debug import plt, plotPolygon
            #     plotPolygon(intersection.area,False,'k','r',1,True,0.4,999)

            self.processedNodes.add(location)
            if intersection.area and intersection.leaveWays:
                self.intersections.append(intersection)

    def finalizeOutput(self):
        for _, _, _, street in self.waymap.edges(data='object',keys=True):
            for item in street.iterItems():
                if isinstance(item,Section):
                    section = item
                    if section.trimS < section.trimT:
                        pass # QUICKTEST
                        # section.centerline = section.polyline.trimmed(section.trimS,section.trimT)[::]
                    else:
                        section.valid = False

        for _, _, _, street in self.waymap.edges(data='object',keys=True):
            if isinstance(street, Street):
                #if not street.style:
                #    street.style = self.styleStore.get( self.getStyle(street) )
                street.setStyleForItems()

        # # DEBUG: Show intersections.
        # # The plotting functions for this debug part are at the end of this module
        # if False and self.app.type == AppType.commandLine:
        #     from debug import plt
        #     for p, intersection in self.waymap.iterNodes(Corner):
        #         plt.plot(p[0],p[1],'mD',markersize=8,zorder=900)


    def createIntersectionAreas(self):
        nodesAlreadyProcessed = self.processedNodes
        nodesAlreadyProcessed.update([node for node in self.internalTransitionSideLanes.keys()])
        nodesAlreadyProcessed.update([node for node in self.internalTransitionSymLanes.keys()])

        # Create intersections and check for conflicts
        for nr,node in enumerate(self.sectionNetwork):
            if node not in nodesAlreadyProcessed:
                intersection = Intersection(node, self.sectionNetwork, self.waySections)

                if intersection.order <= 1:
                    continue
                self.intersections[node] = intersection


        node2isectArea = dict()
        # Now, the normal intersection areas are constructed
        for node,intersection in self.intersections.items():

            # # TODO treat somehow short ways, this here does not work
            shortWays = intersection.cleanShortWays(False)
            if shortWays:
                for way in shortWays:
                    way.section.isValid = False
            #         polygon = way.polyline.buffer(way.leftW,-way.rightW)
            #         isectArea = IntersectionArea()
            #         isectArea.connectors = dict()
            #         isectArea.polygon = polygon
            #         self.intersectionAreas.append(isectArea)
            #     continue

            if intersection.order > 2:
                polygon = None
                try:
                    polygon, connectors = intersection.intersectionPoly()
                except Exception:
                    pass
                if polygon:
                    isectArea = IntersectionArea()
                    isectArea.polygon = polygon
                    isectArea.connectors = connectors
                    node2isectArea[node] = len(self.intersectionAreas)
                    self.intersectionAreas.append(isectArea)

        # Find conflicting intersection areas from over-trimmed way-sections
        conflictingSets = DisjointSets()
        for sectionNr,section in self.waySections.items():
            if section.trimT <= section.trimS:
                if section.originalSection.s not in node2isectArea or section.originalSection.t not in node2isectArea:
                    continue
                indxS = node2isectArea[section.originalSection.s] # index of isectArea at source
                indxT = node2isectArea[section.originalSection.t] # index of isectArea at target
                conflictingSets.addSegment(indxS,indxT)

        self.mergeConflictingAreas(conflictingSets)

    def mergeConflictingAreas(self, conflictingSets):
        # merge conficting intersection areas
        # adapted from:
        # https://stackoverflow.com/questions/7150766/union-of-many-more-than-two-polygons-without-holes
        conflictingAreasIndcs = []
        mergedAreas = []
        for nr,conflicting in enumerate(conflictingSets):
            conflictingAreasIndcs.extend(conflicting)
            waiting = conflicting.copy()  # indices of unprocessed conflicting areas
            merged = []                   # indices of processed (merged) conflicting areas
            while waiting:
                p = waiting.pop()
                merged.append(p)
                mergedPoly = self.intersectionAreas[p].polygon
                changed = True
                while changed:
                    changed = False
                    for q in waiting:
                        for s in merged:
                            if q in conflictingSets.G[s]: # True, if areas s and q conflict (intersect)
                                waiting.remove(q)
                                changed = True
                                try:
                                    ret = boolPolyOp(mergedPoly,self.intersectionAreas[q].polygon,'union')
                                except Exception:
                                    print('Problem')
                                    break
                                mergedPoly = ret[0]
                                merged.append(q)
                                break

            # The merged poygon is now in <mergedPoly>. Be sure that it 
            # is ordered counter-clockwise.
            area = sum( (p2[0]-p1[0])*(p2[1]+p1[1]) for p1,p2 in zip(mergedPoly,mergedPoly[1:]+[mergedPoly[0]]))
            if area > 0.:
                mergedPoly.reverse()

            # An instance of IntersectionArea is now constructed with
            # its remaining connectors.
            mergedArea = IntersectionArea()
            mergedArea.polygon = mergedPoly
            # Find duplicate connectors, which have to be removed.
            connectorIDs = [abs(id) for indx in conflicting for id in self.intersectionAreas[indx].connectors]
            seenIDs = set()
            dupIDs = [x for x in connectorIDs if x in seenIDs or seenIDs.add(x)]

            # for indx in conflicting:
            #     print(self.intersectionAreas[indx].connectors)
            # plotPolygon(mergedPoly,True)
            # plotEnd()

            for indx in conflicting:
                conflictingArea = self.intersectionAreas[indx]
                connectors = conflictingArea.connectors
                for signedKey,connector in connectors.items():
                    key = abs(signedKey)
                    # Keep connectors that don't have duplicate IDs
                    if key not in dupIDs:#if v0 in mergedPoly and v1 in mergedPoly:
                        v0 = conflictingArea.polygon[connector]
                        if v0 in mergedPoly:
                            newConnector = mergedPoly.index(v0)
                            mergedArea.connectors[signedKey] = newConnector
                        # else:
                        #     plotPolygon(conflictingArea.polygon,True,'m','m',4)


            mergedAreas.append(mergedArea)

            # avoid a connector to reach over polygon end point index
            mc = max(c for c in mergedArea.connectors.values())
            if mc >= len(mergedArea.polygon)-1:
                mergedArea.polygon = mergedArea.polygon[1:] + mergedArea.polygon[:1]
                for key,connector in mergedArea.connectors.items():
                    mergedArea.connectors[key] = connector-1

        # Now remove conflicting (overlapping) areas from list and add the merged areas
        for i in sorted(conflictingAreasIndcs, reverse = True):
                del self.intersectionAreas[i]
        self.intersectionAreas.extend(mergedAreas)

    def mergeOverlappingIntersections(self):
        # Create spatial index for intersection areas to find overlapping areas faster
        self.areaIndex = StaticSpatialIndex()
        areaIndx2Indx = dict()    # Dictionary from index to Id
        boxes = dict()      # Bounding boxes of way-sections
        for Id, area in enumerate(self.intersectionAreas):
            min_x = min(v[0] for v in area.polygon)
            min_y = min(v[1] for v in area.polygon)
            max_x = max(v[0] for v in area.polygon)
            max_y = max(v[1] for v in area.polygon)
            bbox = BBox(None,min_x,min_y,max_x,max_y)
            index = self.areaIndex.add(min_x,min_y,max_x,max_y)
            areaIndx2Indx[index] = Id
            bbox.index = index
            boxes[Id] = (min_x,min_y,max_x,max_y)
        self.areaIndex.finish()

        # Check intersection areas for overlap
        overlappingAreas = DisjointSets()
        for indx in range(len(self.intersectionAreas)):
            results = stack = []
            overlapIndxs = self.areaIndex.query(*boxes[indx],results,stack)
            if overlapIndxs:
                area1 = self.intersectionAreas[indx]
                for overlapIndx in overlapIndxs:
                    if overlapIndx != indx:
                        area2 = self.intersectionAreas[overlapIndx]
                        # area1 and area2 have overlapping boxes, check if they really overlap
                        overlap = boolPolyOp(area1.polygon,area2.polygon, 'intersection')
                        if (overlap):
                            overlappingAreas.addSegment(indx,overlapIndx)

        # Merge overlapping areas
        self.mergeConflictingAreas(overlappingAreas)

    def mergeOverlapsBySymLanesAndIntersections(self):
        # Special case: Did transition sym lanes overlap intersections?
        lanesToRemove = []

        # Recreate index of interesection areas, because there may have been merged        
        self.areaIndex = StaticSpatialIndex()
        areaIndx2Indx = dict()    # Dictionary from index to Id
        boxes = dict()      # Bounding boxes of way-sections
        for Id, area in enumerate(self.intersectionAreas):
            min_x = min(v[0] for v in area.polygon)
            min_y = min(v[1] for v in area.polygon)
            max_x = max(v[0] for v in area.polygon)
            max_y = max(v[1] for v in area.polygon)
            bbox = BBox(None,min_x,min_y,max_x,max_y)
            index = self.areaIndex.add(min_x,min_y,max_x,max_y)
            areaIndx2Indx[index] = Id
            bbox.index = index
            boxes[Id] = (min_x,min_y,max_x,max_y)
        self.areaIndex.finish()

        for node,lane in self.internalTransitionSymLanes.items():
            min_x = min(v[0] for v in lane.polygon)
            min_y = min(v[1] for v in lane.polygon)
            max_x = max(v[0] for v in lane.polygon)
            max_y = max(v[1] for v in lane.polygon)
            results = []
            stack= []

            # Check for overlap with an intersection area (using already existing index)
            areaIDs = self.areaIndex.query(min_x, min_y, max_x, max_y, results, stack)
            for areaID in areaIDs: # Should be only one
                if areaID >= len(self.intersectionAreas):
                    continue
                areaPoly = self.intersectionAreas[areaID].polygon
                lanePoly = lane.polygon
                overlap = boolPolyOp(lanePoly, areaPoly, 'intersection')

                # If there is an overlap, we merge the sym lane to the intersection area,
                # remove it and also invalidate the way that connects both
                if overlap:
                    # Merge sym lane to the intersection area
                    try:
                        ret = boolPolyOp(areaPoly,lanePoly,'union')
                        mergedPoly = ret[0]
                    except Exception:
                        print('Problem')
                        break

                    # mergedPoly must be ordered counter-clockwise.
                    area = sum( (p2[0]-p1[0])*(p2[1]+p1[1]) for p1,p2 in zip(mergedPoly,mergedPoly[1:]+[mergedPoly[0]]))
                    if area > 0.:
                        mergedPoly.reverse()

                    # Get original connectors with vertices
                    connectorIDs = dict()
                    connectorIDs.update( {id:lanePoly[val] for id,val in lane.connectors.items()} )
                    connectorIDs.update( {id:areaPoly[val] for id,val in self.intersectionAreas[areaID].connectors.items()} )
                    # Find duplicate connectors, which have to be removed.
                    seenIDs = set()
                    dupIDs = [x for x in connectorIDs if abs(x) in seenIDs or seenIDs.add(abs(x))]

                    # Create new connector for merged area
                    newConnectors = dict()
                    for signedKey,vertex in connectorIDs.items():
                        key = abs(signedKey)
                        # Keep only connectors that don't have duplicate IDs and have a vertex in the merged area
                        if key not in dupIDs:
                            if vertex in mergedPoly:
                                newConnectors[signedKey] = mergedPoly.index(vertex)

                    # Update the intersection area
                    self.intersectionAreas[areaID].polygon = mergedPoly
                    self.intersectionAreas[areaID].connectors = newConnectors

                    # Remove this sym lane
                    lanesToRemove.append(node)

        for node in lanesToRemove: # remove overlapping TransitionSymLane
            self.internalTransitionSymLanes.pop(node, None)    

    def createOutput(self):
        # Finally, the StreetSections of the ways are constructed
        for section in self.waySections.values():
            if not section.isValid:
                continue
            waySlice = None
            if section.trimT > section.trimS + 1.e-5:
                waySlice = section.polyline.trimmed(section.trimS,section.trimT)
                section_gn = StreetSection()

                section_gn.start = None     # Set below
                section_gn.end = None       # Set below

                section_gn.centerline = waySlice.verts
                section_gn.category = section.originalSection.category
                section_gn.forwardLanes = section.forwardLanes
                section_gn.backwardLanes = section.backwardLanes
                section_gn.bothLanes = section.bothLanes
                section_gn.totalLanes = section.forwardLanes + section.backwardLanes + section.bothLanes
                section_gn.laneL = bool(section.fwdLaneL) or bool(section.bwdLaneR)
                section_gn.laneR = bool(section.bwdLaneL) or bool(section.fwdLaneR)
                section_gn.width = section.width
                section_gn.offset = section.offset
                section_gn.tags = section.originalSection.tags
                section_gn.numPoints = len(section_gn.centerline)

                self.waySectionLines[section.id] = section_gn
            # else:
            #     # already treated as reason for conflicting areas

        for transition in self.internalTransitionSideLanes.values():
            way1 = self.waySectionLines.get(abs(transition.ways[0]),None)
            way2 = self.waySectionLines.get(abs(transition.ways[1]),None)
            if not way1 or not way2:
                continue
            preTurnWay, turnWay = (way1, way2) if way1.totalLanes < way2.totalLanes else (way2, way1)

            transition.incoming = preTurnWay
            transition.outgoing = turnWay

            if transition.ways[0]<0:
                preTurnWay.end = transition
            else:
                preTurnWay.start = transition

            if transition.ways[1]<0:
                turnWay.end = transition
            else:
                turnWay.start = transition


            
            transition.totalLanesIncreased = transition.outgoing.totalLanes > transition.incoming.totalLanes
            self.transitionSideLanes.append(transition)

        for transition in self.internalTransitionSymLanes.values():
            for streetSectionIdx in transition.connectors:
                if abs(streetSectionIdx) in self.waySectionLines:
                    streetSection = self.waySectionLines[abs(streetSectionIdx)]
                    if streetSectionIdx > 0:
                        streetSection.start = transition
                    else:
                        streetSection.end = transition
            self.transitionSymLanes.append(transition)

        for intersection in self.intersectionAreas:
            for streetSectionIdx in intersection.connectors:
                if abs(streetSectionIdx) in self.waySectionLines:
                    streetSection = self.waySectionLines[abs(streetSectionIdx)]
                    if streetSectionIdx > 0:
                        streetSection.start = intersection
                    else:
                        streetSection.end = intersection

# === debug plotting functions ================================================

# from matplotlib import pyplot as plt

# def plotNetwork(network,waySections=None):
#     from mpl.renderer.road_polygons import RoadPolygonsRenderer
#     # for section in waySections.values():
#     #     seg = section.originalSection
#     for count,seg in enumerate(network.iterAllSegments()):
#         # plt.plot(seg.s[0],seg.s[1],'k.')
#         # plt.plot(seg.t[0],seg.t[1],'k.')
#         color = 'r' if seg.category=='scene_border' else 'y'

#         for v1,v2 in zip(seg.path[:-1],seg.path[1:]):
#             plt.plot( (v1[0], v2[0]), (v1[1], v2[1]), **RoadPolygonsRenderer.styles[seg.category], zorder=50 )
#             # plt.plot( (v1[0], v2[0]), (v1[1], v2[1]), 'k', 0.5, zorder=50)


# def plotPureNetwork(network,arrows=False,showIDs=False):
#     from itertools import tee
#     def pairs(iterable):
#         # s -> (s0,s1), (s1,s2), (s2, s3), ...
#         p1, p2 = tee(iterable)
#         next(p2, None)
#         return zip(p1,p2)

#     for count,seg in enumerate(network.iterAllSegments()):
#         color = 'g' if seg.category=='scene_border' else 'k'
#         if arrows:
#             width = 2
#             for v0,v1 in pairs(seg.path):
#                 x = (v0[0]+v1[0])/2
#                 y = (v0[1]+v1[1])/2
#                 arrowprops=dict(color='r', width=width, shrink=0.05, headwidth=width*3, headlength=5*width)
#                 plt.gca().annotate("", xy=(x,y), xytext=(v0[0],v0[1]),arrowprops=arrowprops)
#                 plt.plot([v0[0],v1[0]],[v0[1],v1[1]],color=color, linewidth=width)
#         else:
#             plotLine(seg.path,False,color,1)
#         plt.plot(seg.s[0],seg.s[1],'k.')
#         plt.plot(seg.t[0],seg.t[1],'k.')
#         if showIDs:
#             c = sum(seg.path, Vector((0,0)))/len(seg.path)
#             plt.text(c[0],c[1],'  '+str(seg.sectionId) )
            
# def plotLine(line,vertsOrder,lineColor='k',width=1.,order=100):
#     x = [n[0] for n in line]
#     y = [n[1] for n in line]
#     plt.plot(x,y,color=lineColor,linewidth=width,zorder=order)
#     if vertsOrder:
#         for i,(xx,yy) in enumerate(zip(x,y)):
#             plt.text(xx,yy,str(i),fontsize=12)

# def plotPolygon(poly,vertsOrder,lineColor='k',fillColor='k',width=1.,fill=False,alpha = 0.2,order=100):
#     if not poly:
#         return
#     x = [n[0] for n in poly] + [poly[0][0]]
#     y = [n[1] for n in poly] + [poly[0][1]]
#     if fill:
#         plt.fill(x[:-1],y[:-1],color=fillColor,alpha=alpha,zorder = order)
#     plt.plot(x,y,color=lineColor,linestyle=':',linewidth=width,zorder=order)
#     if vertsOrder:
#         for i,(xx,yy) in enumerate(zip(x[:-1],y[:-1])):
#             plt.text(xx,yy,str(i),fontsize=12)

# def randomColor(n, name='hsv'):
#     cmap = plt.cm.get_cmap(name, n)
#     cmapList = [ cmap(i) for i in range(n)]
#     import random
#     random.shuffle(cmapList)
#     i = 0
#     while True:
#         yield cmapList[i]
#         i = (i+1)%n

# def plotEnd():
#     plt.gca().axis('equal')
#     plt.show()
