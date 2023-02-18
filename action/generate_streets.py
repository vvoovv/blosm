from collections import defaultdict
from itertools import tee, islice, cycle, permutations
from statistics import median
from mathutils import Vector

from way.way_network import WayNetwork, NetSection
from way.way_algorithms import createSectionNetwork
from way.way_section import WaySection
from way.way_cluster import LongClusterWay, createLeftTransition,\
                            createRightTransition, createLeftIntersection, createRightIntersection
from way.way_intersections import Intersection
from way.intersection_cluster import IntersectionCluster
from defs.road_polygons import ExcludedWayTags
from defs.way_cluster_params import minTemplateLength, minNeighborLength, searchDist,\
                                    canPair, dbScanDist, transitionSlope
from lib.SweepIntersectorLib.SweepIntersector import SweepIntersector
from lib.CompGeom.StaticSpatialIndex import StaticSpatialIndex, BBox
from lib.CompGeom.algorithms import SCClipper
from lib.CompGeom.BoolPolyOps import boolPolyOp
from lib.CompGeom.GraphBasedAlgos import DisjointSets
from lib.CompGeom.PolyLine import PolyLine
from lib.CompGeom.LinePolygonClipper import LinePolygonClipper
from lib.CompGeom.centerline import centerlineOf
from lib.CompGeom.dbscan import dbClusterScan


class BaseWaySection:
    
    def getNormal(self, index, left):
        centerline = self.centerline
        
        if index == 0:
            vector = centerline[1] - centerline[0]
        elif index == -1:
            vector = centerline[-1] -  centerline[-2]
        
        vector.normalize()
        
        return Vector((-vector[1], vector[0])) if left else Vector((vector[1], -vector[0]))
    

class TrimmedWaySection(BaseWaySection):
    
    def __init__(self):
        # The trimmed centerline in forward direction (first vertex is start.
        # and last is end). A Python list of vertices of type mathutils.Vector.
        self.centerline = None
        
        # If it is a separate way section (not a part of a cluster), it's always equal to zero.
        # It it is a part of cluster, it is a signed distance from the cluster's centerline
        # to the way's centerline. Ways to the left from the cluster's centerline (relative
        # to the direction of the centerline) have negative offset, otherwise they have positive offset.
        self.offset = 0.
        
        # The width of the way.
        self.width = None
        
        # The category of the way-section.
        self.category = None
        
        # The left and right number of lanes, seen relative to the direction
        # of the centerline. A tuple of integers. For one-way streets, only one
        # integer with the number of lanes is in the tuple.
        self.nrOfLanes = None
        
        # The OSM tags of the way-section.
        self.tags = None

        # True, if start of way is connected to other ways, else dead-end
        self.startConnected = None

        # True, if end of way is connected to other ways, else dead-end
        self.endConnected = None


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


class IntersectionArea():
    
    def __init__(self):
        # The vertices of the polygon in counter-clockwise order. A Python
        # list of vertices of type mathutils.Vector.
        self.polygon = None
        
        # The connectors of this polygon to the way-sections.
        # A dictionary of tuples, where the key is the ID of the corresponding
        # way-section key in the dictionary of TrimmedWaySection. The tuple has two
        # elements, <indx> and <end>. <indx> is the first index in the intersection
        # polygon for this connector, the second index is <indx>+1. <end> is
        # the type 'S' for the start or 'E' for the end of the TrimmedWaySection
        # connected here.
        self.connectors = dict()
        
        # The connectors of this polygon to the way-clusters. A dictionary of tuples,
        # where the key is also the key of the corresponding way-cluster in the 
        # dictionary <wayClusters> of The manager. The tuple has three elements, <indx>,
        # <end> and <way>. <indx> is the first index in the intersection
        # polygon for this connector, the second index is <indx>+1. <end> is
        # the type 'S' for the start or 'E' for the end of the way cluster
        # connected here. <way> is -1 if the whole way-cluster connects there,
        # or the index in the list of way descriptors <waySections> in <WayCluster>,
        # if only a single way of the cluster connects there.
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
        """
        if not self.connectorsInfo:
            self.numPoints = len(self.polygon)
            
            connectorsInfo = self.connectorsInfo = []
            if self.connectors:
                connectorsInfo.extend(
                    # <False> means that it isn't a cluster of street segments
                    (connectorInfo[0], False, segmentId, connectorInfo[1]) for segmentId, connectorInfo in self.connectors.items()
                )
            if self.clusterConns:
                connectorsInfo.extend(
                    # <True> means that it is a cluster of street segments
                    (connectorInfo[0], True, segmentId, connectorInfo[1]) for segmentId, connectorInfo in self.clusterConns.items()
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
        if abs(v1.cross(v2)) > 0.65:
            return True
    return False

def mad(x):
    # median absolute deviation
    # https://en.wikipedia.org/wiki/Median_absolute_deviation
    m = median(x)
    return median( abs(xi-m) for xi in x)

def createWaySection(offset, width, nrOfLanes, category, tags):
    waySection = TrimmedWaySection()
    
    waySection.offset, waySection.width, waySection.nrOfLanes, waySection.category, waySection.tags =\
        offset, width, nrOfLanes, category, tags
    
    return waySection
# ----------------------------------------------------------------

class StreetGenerator():
    
    def __init__(self,isectShape='common'): # isectShape -> 'common' or 'separated'
        self.isectShape = isectShape

        # Interface via manager
        self.intersectionAreas = None
        self.wayClusters = None
        self.waySectionLines = None
        self.networkGraph = None
        self.sectionNetwork = None

        # Internal use
        self.waySections = dict()
        self.intersections = dict()
        self.waysForClusters = None
        self.longClusterWays = []
        self.virtualWayIndx = -1
        self.processedNodes = set()

    def do(self, manager):
        self.wayManager = manager
        self.intersectionAreas = manager.intersectionAreas
        self.wayClusters = manager.wayClusters
        self.waySectionLines = manager.waySectionLines
        NetSection.ID = 0   # This class variable doesn't get reset with new instance of StreetGenerator!!

        self.useFillet = False          # Don't change, it does not yet work!
        self.findSelfIntersections()
        self.createWaySectionNetwork()
        self.createWaySections()
        self.detectWayClusters()
        self.createLongClusterWays()
        self.createClusterTransitionAreas()
        self.createClusterIntersections()
        self.createWayClusters()
        self.createOutput()

    def findSelfIntersections(self):
        # some way tags to exclude, used also in createWaySectionNetwork(),
        # ExcludedWayTags is defined in <defs>.
        uniqueSegments = defaultdict(set)
        for way in self.wayManager.getAllVehicleWays():
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
        wayManager.networkGraph = self.networkGraph = WayNetwork()

        # some way tags to exclude, used also in createWaySectionNetwork(),
        # ExcludedWayTags is defined in <defs>.
        for way in wayManager.getAllVehicleWays():
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
        wayManager.sectionNetwork = self.sectionNetwork = createSectionNetwork(wayManager.networkGraph)

    def createWaySections(self):
        for net_section in self.sectionNetwork.iterAllForwardSegments():
            if net_section.category != 'scene_border':
                section = WaySection(net_section,self.sectionNetwork)
                self.waySections[net_section.sectionId] = section

    def detectWayClusters(self):
        # Create spatial index (R-tree) of way sections
        spatialIndex = StaticSpatialIndex()
        indx2Id = dict()    # Dictionary from index to Id
        boxes = dict()      # Bounding boxes of way-sections
        for Id, section in self.waySections.items():
            # exclusions by some criteria
            if isEdgy(section.polyline): continue
            if section.polyline.length() < min(minTemplateLength,minNeighborLength):
                continue

            min_x = min(v[0] for v in section.polyline.verts)
            min_y = min(v[1] for v in section.polyline.verts)
            max_x = max(v[0] for v in section.polyline.verts)
            max_y = max(v[1] for v in section.polyline.verts)
            bbox = BBox(None,min_x,min_y,max_x,max_y)
            index = spatialIndex.add(min_x,min_y,max_x,max_y)
            indx2Id[index] = Id
            bbox.index = index
            boxes[Id] = (min_x,min_y,max_x,max_y)
        spatialIndex.finish()

        self.waysForClusters = DisjointSets()
        # Use every way-section, that has been inserted into spatial index 
        # as template and create a buffer around it.
        for Id, section in ((Id, section) for Id, section in self.waySections.items() if Id in boxes):
            # The template must have a minimum length.
            if section.polyline.length() > minTemplateLength:
                # Create buffer width width depending on category.
                bufferWidth = searchDist[section.originalSection.category]
                bufferPoly = section.polyline.buffer(bufferWidth,bufferWidth)

                # Create line clipper with this polygon.
                clipper = LinePolygonClipper(bufferPoly.verts)

                # Get neighbors of template way from static spatial index with its
                # bounding box, expanded by the buffer width.
                min_x,min_y,max_x,max_y = boxes[Id]
                results = stack = []
                neighbors = spatialIndex.query(min_x-bufferWidth,min_y-bufferWidth,
                                               max_x+bufferWidth,max_y+bufferWidth,results,stack)

                # Now test all neighbors of the remplate for parallelism
                for neigborIndx in neighbors:
                    neighborID = indx2Id[neigborIndx]
                    # The template is its own neighbor
                    if neighborID == Id: continue

                    # Check in table if pairing as cluster is allowed
                    neighborCategory = self.waySections[neighborID].originalSection.category
                    if not canPair[section.originalSection.category][neighborCategory]: continue

                    # Polyline of this neighbor ...
                    neighborLine = self.waySections[neighborID].polyline

                    # ... must be longer than minimal length ...
                    if neighborLine.length() > minNeighborLength:
                        # ... then clip it with buffer polygon
                        inLine, inLineLength, nrOfON = clipper.clipLine(neighborLine.verts)

                        if inLineLength < 0.1: continue # discard short inside lines

                        # To check the quality of parallelism, the "slope" relative to the template's
                        # line is evaluated.
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
        # for cIndx,wayIndxs in enumerate(self.waysForClusters):
        #     # Get the sections included in this cluster
        #     sections = [self.waySections[Id] for Id in wayIndxs]

        #     # Find their endpoints (ends of the cluster)
        #     # Note: Clusters may have common endpoints,
        #     # which are not yet detected here.
        #     sectionEnds = defaultdict(list)
        #     for i,section in enumerate(sections):
        #         s,t = section.originalSection.s, section.originalSection.t
        #         sV, tV = section.sV, section.tV
        #         Id = wayIndxs[i]
        #         # Store the section ID, start- and endpoint, the vertices of the polyline and the
        #         # direction vectors at start and end. The latter will be used to detect common endpoints.
        #         sectionEnds[s].append({'ID':Id,'start':s,'end':t,'verts':section.polyline.verts,      'startV':sV,'endV':-tV})
        #         sectionEnds[t].append({'ID':Id,'start':t,'end':s,'verts':section.polyline.verts[::-1],'startV':tV,'endV':-sV})
        #     clusterEnds = [k for k, v in sectionEnds.items() if len(v) == 1]

        #     if not clusterEnds:
        #         continue

        #     # Connect the segments to polylines for long clusters. Collect the section IDs
        #     # and the intersections along these lines spearately in the same order.
        #     endpoints = clusterEnds.copy()
        #     lines = []
        #     intersectionsAll = []
        #     sectionsIDsAll = []
        #     inClusterIsects = []
        #     while endpoints:
        #         ep = endpoints.pop()
        #         line = []
        #         intersectionsThis = []
        #         sectionsIDs = []
        #         currSec = sectionEnds[ep][0]
        #         while True:
        #             line.extend(currSec['verts'][:-1])  # The end of this section will be the start of the next section.
        #             sectionsIDs.append(currSec['ID'])
        #             end = currSec['end']
        #             if len(sectionEnds[end]) == 2:
        #                 newSec = [sec for sec in sectionEnds[end] if sec['end'] != currSec['start']][0]
        #                 if currSec['endV'].dot(newSec['startV']) < 0.:
        #                     # We have a common endpoint here that does not exist in <endpoints>.
        #                     # We end the current line here and start a new one, that continues.
        #                     # But remember this point as an additional possible cluster endpoint.
        #                     # sectionsIDs.append(currSec['ID'])
        #                     intersectionsThis.append(currSec['end'])
        #                     line.append(end)
        #                     inClusterIsects.append(end)
        #                     lines.append(PolyLine(line))
        #                     intersectionsAll.append(intersectionsThis)
        #                     sectionsIDsAll.append(sectionsIDs)
        #                     line = []
        #                     intersectionsThis = []
        #                     sectionsIDs = []
        #                     currSec = newSec
        #                 else:
        #                     intersectionsThis.append(currSec['end'])
        #                     currSec = newSec
        #             else:
        #                 if end in endpoints:
        #                     endpoints.remove(end)
        #                 break
        #         line.append(end)    # Add the endpoint of the last section.
        #         lines.append(PolyLine(line))
        #         intersectionsAll.append(intersectionsThis)
        #         sectionsIDsAll.append(sectionsIDs)

        #     # All lines found should be ordered in the same direction. To realize this,
        #     # find the longest polyline and project the first and last vertex of every line
        #     # onto this reference polyline. The line parameter of the first vertex must
        #     # then be smaller than the one of the last vertex. Store the line parameters
        #     # for later use.
        #     referenceLine = max( (line for line in lines), key=lambda x: x.length() )
        #     params0 = []
        #     params1 = []
        #     for indx,line in enumerate(lines):
        #         _,t0 = referenceLine.orthoProj(line[0])
        #         _,t1 = referenceLine.orthoProj(line[-1])
        #         if t0 > t1: # reverse this line, if True
        #             lines[indx].toggleView()
        #             params0.append(t1)
        #             params1.append(t0)
        #         else:
        #             params0.append(t0)
        #             params1.append(t1)

        #     # Sometimes, there have been gaps left due to short way-sections.
        #     # During cluster detection, lines shorter than <minTemplateLength>
        #     # have been eliminated. We search now for gaps between start- and
        #     # end-points of lines, that are shorter than this value.
        #     perms = permutations(range(len(lines)),2)
        #     possibleGaps = []
        #     for i0,i1 in perms:
        #         if params0[i1] < params1[i0]: # prevent short cluster to be merged
        #             d = (lines[i0][0]-lines[i1][-1]).length
        #             if d < minTemplateLength:
        #                 possibleGaps.append( (i0,i1) )

        #     # If there are such gaps, the corresponding lines are merged, if there
        #     # exists a real way-section that can fill this gap.
        #     if possibleGaps:
        #         # In case of bifurcations, we connect only to one line. Clean
        #         # possible gaps by removing duplicates in first or second index.
        #         i0s, i1s = [], []
        #         cleanedPossibleGaps = []
        #         for i0,i1 in possibleGaps:
        #             if i0 not in i0s and i1 not in i1s:
        #                 cleanedPossibleGaps.append( (i0,i1) )
        #                 i0s.append(i0)
        #                 i1s.append(i1)

        #         # Merge lines through gaps.
        #         linesToRemove = []
        #         for i0,i1 in cleanedPossibleGaps:
        #             if lines[i0][0] in self.sectionNetwork and lines[i1][-1] in self.sectionNetwork:
        #                 if lines[i1][-1] in self.sectionNetwork[lines[i0][0]]:
        #                     sectionId = self.sectionNetwork[lines[i0][0]][lines[i1][-1]][0].sectionId
        #                 elif lines[i0][0] in self.sectionNetwork[lines[i1][-1]]:
        #                     sectionId = self.sectionNetwork[lines[i1][-1]][lines[i0][0]][0].sectionId
        #                 else:
        #                     print('Problem',i0,i1,cIndx)
        #                     continue
        #                 lines[i1] = PolyLine( lines[i1][:] + lines[i0][:])
        #                 sectionsIDsAll[i1].extend(sectionsIDsAll[i0] + [sectionId])
        #                 intersectionsAll[i1].extend(intersectionsAll[i0] + [lines[i0][0],lines[i1][-1]])
        #                 linesToRemove.append(i0)

        #         # Remove merged parts.
        #         if linesToRemove:
        #             for index in sorted(linesToRemove, reverse=True):
        #                 del lines[index]                    

        #     # We like to have the start points close together. The line ends that have a
        #     # smaller median absolute deviation are selected as start points.
        #     params0 = [referenceLine.orthoProj(line[0])[1] for line in lines]
        #     params1 = [referenceLine.orthoProj(line[-1])[1] for line in lines]
        #     mad0 = mad(params0)
        #     mad1 = mad(params1)
        #     if mad1 < mad0:
        #         for indx,line in enumerate(lines):
        #             lines[indx].toggleView()
        #             sectionsIDsAll[indx] = sectionsIDsAll[indx][::-1]
        #             intersectionsAll[indx] = intersectionsAll[indx][::-1]

        #     # As last step, the lines have to be ordered from left to right.
        #     # The first segment of an arbitrary line is turned perpendicularly
        #     # to the right and all startpoints are projected onto this vector,
        #     # delivering a line parameter t. The lines are the sorted by 
        #     # increasing parameter values.
        #     vec = lines[0][1] - lines[0][0] # first segment
        #     p0 = lines[0][0]
        #     perp = Vector((vec[1],-vec[0])) # perpendicular to the right
        #     sIndx = [ i for i in range(len(lines))]
        #     sIndx.sort(key=lambda x: (lines[x][0]-p0).dot(perp))

        #     # From here, only the two outermost ways in the cluster are used,
        #     # in order to be able the old 2-ways processes behind. This part
        #     # will be refactored, when these processes ar eready.
        #     left = lines[sIndx[0]]
        #     right = lines[sIndx[-1]]
        #     leftSectionIDs = sectionsIDsAll[sIndx[0]]
        #     rightSectionIDs = sectionsIDsAll[sIndx[-1]]
        #     intersectionsThis = [intersectionsAll[sIndx[0]], intersectionsAll[sIndx[-1]] ]

        #     # Check ratio of endpoînt distances to decide if valid cluster
        #     d0 = (left[0]-right[0]).length
        #     d1 = (left[-1]-right[-1]).length
        #     ratio = min(d0,d1)/max(d0,d1)

        #     # Discard cluster if too asymetric ends.
        #     # TODO: Will require more sophisticated test
        #     if ratio < 0.5 or inClusterIsects:
        #         continue

        #     centerline = PolyLine( centerlineOf(left[:],right[:]) )

        #     longCluster = LongClusterWay(self)
        #     longCluster.leftSectionIDs = leftSectionIDs
        #     longCluster.rightSectionIDs = rightSectionIDs
        #     longCluster.intersectionsAll = intersectionsThis
        #     longCluster.left = left
        #     longCluster.right = right
        #     longCluster.centerline = centerline

        #     # Split the long cluster in smaller pieces of type <ClusterWay>,
        #     # stored in <longCluster>.
        #     longCluster.split()

        #     self.longClusterWays.append(longCluster)
        for cIndx,wayIndxs in enumerate(self.waysForClusters):
            # Get the sections included in this cluster
            sections = [self.waySections[Id] for Id in wayIndxs]

            # Find their endpoints (ends of the cluster)
            # TODO: In the general case, clusters may have common endpoints 
            sectionEnds = defaultdict(list)
            for section in sections:
                s,t = section.originalSection.s, section.originalSection.t
                sectionEnds[s].append( section )
                sectionEnds[t].append( section )
            endpoints = [k for k, v in sectionEnds.items() if len(v) == 1]

            # Collect adjacent way-sections and join them to lines
            # TODO: Again, in the general case, clusters may have common endpoints
            sectionIDs = wayIndxs.copy()
            lines = []
            intersectionsAll = []
            sectionsIDsAll = []
            while endpoints:
                cur = endpoints.pop()
                line = [cur]
                intersectionsThis = []
                sectionsIDs = []
                while True:
                    curID = next(filter(lambda x: cur in [self.waySections[x].originalSection.s,self.waySections[x].originalSection.t], sectionIDs) )
                    curSec = self.waySections[curID]
                    sectionIDs.remove(curID)
                    sectVerts = curSec.polyline.verts if curSec.originalSection.s==cur else curSec.polyline.verts[::-1]
                    line.extend(sectVerts[1:])
                    sectionsIDs.append(curID)
                    cur = sectVerts[-1]
                    if cur in endpoints:
                        endpoints.remove(cur)
                        lines.append(line)
                        intersectionsAll.append(intersectionsThis)
                        sectionsIDsAll.append(sectionsIDs)
                        break
                    else:
                        intersectionsThis.append(cur)

            # project endpoints of line0 onto line1 to find corresponding ends.
            # Reverse line1, if required, so that both starts are on the same side.
            line0, line1 = PolyLine(lines[0]), PolyLine(lines[1])
            sectionsIDs0, sectionsIDs1 = sectionsIDsAll[0], sectionsIDsAll[1]
            _,t0 = line1.orthoProj(line0[0])
            _,t1 = line1.orthoProj(line0[-1])
            if t1 < t0:
                line1 = PolyLine(line1.verts[::-1])
                sectionsIDs1 = sectionsIDs1[::-1]

            # Check ratio of endpoînt distances to decide if valid cluster
            d0 = (line0[0]-line1[0]).length
            d1 = (line0[-1]-line1[-1]).length
            ratio = min(d0,d1)/max(d0,d1)

            # Discard cluster if too asymetric ends.
            # TODO: Will require moe sophisticated test
            if ratio < 0.5:
                continue

            centerline = PolyLine( centerlineOf(line0.verts,line1.verts) )
            # find order of lines from left to right
            side = (line0[1]-line0[0]).cross(line1[0]-line0[0]) < 0.
            left, right = (line0,line1) if side else (line1,line0)
            leftSectionIDs, rightSectionIDs = (sectionsIDs0,sectionsIDs1) if side else (sectionsIDs1,sectionsIDs0)

            longCluster = LongClusterWay(self)
            longCluster.leftSectionIDs = leftSectionIDs
            longCluster.rightSectionIDs = rightSectionIDs
            longCluster.intersectionsAll = intersectionsAll
            longCluster.left = left
            longCluster.right = right
            longCluster.centerline = centerline

            # Split the long cluster in smaller pieces of type <ClusterWay>,
            # stored in <longCluster>.
            longCluster.split()

            self.longClusterWays.append(longCluster)

    def createClusterTransitionAreas(self):
        areas = []
        for nr,longClusterWay in enumerate(self.longClusterWays):
            if len(longClusterWay.clusterWays) > 1:
                for cluster1,cluster2 in pairs(longClusterWay.clusterWays):
                    clustConnectors = dict()
                    wayConnectors = dict()
                    if cluster1.endPoints[0] == 'left':
                        node = cluster1.endPoints[1].freeze()
                        order = self.sectionNetwork.borderlessOrder(node)
                        if order == 2:
                            area, clustConnectors = createLeftTransition(self,cluster1,cluster2)
                        else:
                            area, clustConnectors, wayConnectors = createLeftIntersection(self,cluster1,cluster2,node)
                    elif cluster1.endPoints[0] == 'right':
                        node = cluster1.endPoints[2].freeze()
                        order = self.sectionNetwork.borderlessOrder(node)
                        if order == 2:
                            area, clustConnectors = createRightTransition(self,cluster1,cluster2)
                        else:
                            area, clustConnectors, wayConnectors = createRightIntersection(self,cluster1,cluster2,node)

                    # Create the final cluster area instance
                    isectArea = IntersectionArea()
                    isectArea.polygon = area
                    isectArea.connectors = wayConnectors
                    isectArea.clusterConns = clustConnectors
                    self.intersectionAreas.append(isectArea)

                    areas.append(area)

                    # Eventually we need to create a short way at a transition
                    if self.isectShape == 'separated':
                        if cluster1.endPoints[0] == 'left':
                            s1 = cluster1.centerline.offsetPointAt(cluster1.trimT,-cluster1.clusterWidth/2.)
                            s2 = cluster2.centerline.offsetPointAt(cluster2.trimS,-cluster2.clusterWidth/2.) 
                            Id = cluster1.rightWayID    # cluster1.rightWayID == cluster2.rightWayID)
                        if cluster1.endPoints[0] == 'right':
                            s1 = cluster1.centerline.offsetPointAt(cluster1.trimT,cluster1.clusterWidth/2.)
                            s2 = cluster2.centerline.offsetPointAt(cluster2.trimS,cluster2.clusterWidth/2.) 
                            Id = cluster1.leftWayID     # cluster1.leftWayID == cluster2.leftWayID)

                        section = self.waySections[Id]
                        section_gn = TrimmedWaySection()
                        section_gn.centerline = [s1,s2]
                        section_gn.category = section.originalSection.category
                        if section.isOneWay:
                            section_gn.nrOfLanes = (section.nrRightLanes)
                        else:
                            section_gn.nrOfLanes = (section.nrLeftLanes, section.nrRightLanes)
                        section_gn.endWidths = section_gn.startWidths = (section.leftWidth, section.rightWidth)
                        section_gn.tags = section.originalSection.tags
                        if section.turnParams:
                            section_gn.endWidths = (section.leftWidth+section.turnParams[0], section.rightWidth+section.turnParams[1])
                        self.waySectionLines[self.virtualWayIndx] = section_gn
                        self.virtualWayIndx -= 1    # Use negative index to avoid conflicts with ordinary ways

            else:   # only one cluster way
                cluster = longClusterWay.clusterWays[0]

    def createClusterIntersections(self):
        # Find groups of neighboring endpoints of cluster ways,
        # using density based scan
        endPoints = []
        for longClusterWay in self.longClusterWays:
            start = longClusterWay.clusterWays[0]
            end = longClusterWay.clusterWays[-1]
            endPoints.append( (start.centerline[0],start,'start') )
            endPoints.append( (end.centerline[-1],end,'end') )
        clusterGroups = dbClusterScan(endPoints, dbScanDist, 2)

        # <clusterGroups> is a list that contains lists of clusterways, where their
        # endpoints are neighbors. These will form intersection clusters, probably 
        # with additional outgoing way-sections. A list entry for a clusterway
        # is a <clusterGroup> and is formed as
        # [centerline-endpoint, cluster-way, type ('start' or 'end')]

        # Create an intersection cluster area for every <clusterGroup>
        for clusterGroup in clusterGroups:
            # Create an instance of <IntersectionCluster>, which will form the
            # cluster area and its connectors.
            isectCluster = IntersectionCluster()

            # Collect all intersection nodes of the way-sections at the cluster.
            # TODO Possibly there are more nodes within the cluster area.           
            nodes = set()
            for cluster in clusterGroup:
                if cluster[2] == 'start':
                    nodes.update( cluster[1].startPoints[1:3])
                elif cluster[2] == 'end':
                    nodes.update( cluster[1].endPoints[1:3])
                else:
                    assert False, 'Should not happen'

            # Insert centerlines and widths of clusters to <IntersectionCluster>. Keep a map
            # <ID2Object> for the ID in <IntersectionCluster> to the inserted object.
            # Collect way IDs (key of <self.waySections>) so that the cluster ways may later
            # be excluded from outgoing ways at end-nodes.
            wayIDs = []
            ID2Object = dict()
            for cluster in clusterGroup:
                wayIDs.extend( cluster[1].startPoints[3:] )
                if cluster[2] == 'start':
                    Id = isectCluster.addWay(cluster[1].centerline,cluster[1].outWL(),cluster[1].outWR())
                    ID2Object[Id] = cluster
                elif cluster[2] == 'end':
                    Id = isectCluster.addWay(PolyLine(cluster[1].centerline[::-1]),cluster[1].outWR(),cluster[1].outWL())
                    ID2Object[Id] = cluster
                else:
                    assert False, 'Should not happen'

            # Find all outgoing way-sections that are connected to cluster nodes, but exlude
            # those that belong to the clusters, using <wayIDs>. Insert them together with their
            # widths into <IntersectionCluster>. Extend the map <ID2Object> for the ID in
            # <IntersectionCluster> to the inserted object.
            for node in nodes:
                for net_section in self.sectionNetwork.iterOutSegments(node):
                    if net_section.category != 'scene_border':
                        if net_section.sectionId not in wayIDs:
                            section = self.waySections[net_section.sectionId]
                            if not (net_section.s in nodes and net_section.t in nodes):                               
                                if section.originalSection.s == node:
                                    Id = isectCluster.addWay(section.polyline,section.leftWidth,section.rightWidth)
                                    ID2Object[Id] = (section,True)  # forward = True
                                else:
                                    Id = isectCluster.addWay(PolyLine(section.polyline[::-1]),section.rightWidth,section.leftWidth)
                                    ID2Object[Id] = (section,False) # forward = False
                            else:
                                # way-sections that connect nodes internally in the cluster area.
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
                        wayConnectors[object[0].id] = ( con[0]+conOffset, 'S' if object[1] else 'E')
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
                                    clustConnectors[object[1].id] = ( con[0]+conOffset+1, 'S', -1)
                                    area.insert(con[0]+conOffset+2,p2)
                                    conOffset += 2
                                else:
                                    clustConnectors[object[1].id] = ( con[0]+conOffset, 'S', -1)
                            else:   # object[2] == 'end'
                                transitionLength = object[1].transitionWidth/transitionSlope
                                dTrans = object[1].centerline.length() - transitionLength
                                tTrans = object[1].centerline.d2t(dTrans)
                                if tTrans < object[1].trimT:
                                    p1 = object[1].centerline.offsetPointAt(tTrans,object[1].outWL())
                                    p2 = object[1].centerline.offsetPointAt(tTrans,-object[1].outWR())
                                    object[1].trimT = min(object[1].trimT,tTrans)
                                    area.insert(con[0]+conOffset+1,p1)
                                    clustConnectors[object[1].id] = ( con[0]+conOffset+1, 'E', -1)
                                    area.insert(con[0]+conOffset+2,p2)
                                    conOffset += 2
                                else:
                                    clustConnectors[object[1].id] = ( con[0]+conOffset, 'E', -1)
                        else:
                            clustConnectors[object[1].id] = ( con[0]+conOffset, 'S' if object[2]=='start' else 'E', -1)

                isectArea = IntersectionArea()
                isectArea.polygon = area
                isectArea.connectors = wayConnectors
                isectArea.clusterConns = clustConnectors
                self.intersectionAreas.append(isectArea)

            else:
                if clusterGroup[0][2] == 'end':
                    clusterGroup[0][1].endConnected = False
                else:
                    clusterGroup[0][1].startConnected = False

    def createWayClusters(self):
        for longClusterWay in self.longClusterWays:
            for cluster in longClusterWay.clusterWays:
                wayCluster = WayCluster()
                wayCluster.centerline = cluster.centerline.trimmed(cluster.trimS,cluster.trimT)[:]
                wayCluster.distToLeft = cluster.clusterWidth/2.
                wayCluster.startConnected = cluster.startConnected
                wayCluster.endConnected = cluster.endConnected

                for wayID in [cluster.leftWayID, cluster.rightWayID]:
                    section = self.waySections[wayID]
                    fwd = section.originalSection.s == cluster.startPoints[1]
                    if section.isOneWay:
                        nrOfLanes = (section.nrRightLanes)
                    else:
                        nrOfLanes = (section.nrLeftLanes, section.nrRightLanes) if fwd else (section.nrRightLanes, section.nrLeftLanes)
                    wayCluster.waySections.append(
                        createWaySection(
                            -cluster.clusterWidth/2. if wayID==cluster.leftWayID else cluster.clusterWidth/2.,
                            section.rightWidth+section.leftWidth,
                            nrOfLanes,
                            section.originalSection.category,
                            section.originalSection.tags
                        )
                    )
                    # cleanup
                    section.isValid = False
                self.wayClusters[cluster.id] = wayCluster

                # Some bookkeeping
                # TODO expand this for multiple ways
                for node in cluster.startPoints[1:3]:
                    self.processedNodes.add(node.freeze())
                for node in cluster.endPoints[1:3]:
                    self.processedNodes.add(node.freeze())

    def createOutput(self):
        # Find first nodes that will produce conflicting intersections
        conflictingNodes = DisjointSets()
        for node in self.sectionNetwork:
            if node not in self.processedNodes:
                intersection = Intersection(node, self.sectionNetwork, self.waySections)
                conflictNodes = intersection.checkForConflicts()
                if conflictNodes:
                    for conflict in conflictNodes:
                            conflictingNodes.addSegment(node,conflict)

        # Create intersections from nodes, that did not produce conflicts
        for node in self.sectionNetwork:
            if node not in self.processedNodes:
                if node not in conflictingNodes.G:
                    intersection = Intersection(node, self.sectionNetwork, self.waySections)
                    self.intersections[node] = intersection

        # Create Intersections from conflicting nodes
        for conflictingCluster in conflictingNodes:
            intersection = Intersection(conflictingCluster, self.sectionNetwork, self.waySections)
            self.intersections[intersection.position.freeze()] = intersection

        node2isectArea = dict()
        # Transitions have to be processed first, because way widths may be altered.
        for node,intersection in self.intersections.items():
            if intersection.order == 2:
                polygon, connectors = intersection.findTransitionPoly()
                if polygon:
                    isectArea = IntersectionArea()
                    isectArea.polygon = polygon
                    isectArea.connectors = connectors
                    node2isectArea[node] = len(self.intersectionAreas)
                    self.intersectionAreas.append(isectArea)

        # Now, the normal intersection areas are constructed
        for node,intersection in self.intersections.items():
            if intersection.order > 2:
                polygon = None
                if self.useFillet:
                    try:
                        polygon, connectors = intersection.intersectionPoly()
                    except:
                        pass
                else:
                    try:
                        polygon, connectors = intersection.intersectionPoly_noFillet()
                    except:
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

        # merge conficting intersection areas
        # adapted from:
        # https://stackoverflow.com/questions/7150766/union-of-many-more-than-two-polygons-without-holes
        conflictingAreasIndcs = []
        mergedAreas = []
        for conflicting in conflictingSets:
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
                                except:
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

            # An instance of IntersectionArea is now constrcuted with
            # its remaining connectors.
            mergedArea = IntersectionArea()
            mergedArea.polygon = mergedPoly
            for indx in conflicting:
                conflictingArea = self.intersectionAreas[indx]
                connectors = conflictingArea.connectors
                for key,connector in connectors.items():
                    # vertices of this connector
                    v0 = conflictingArea.polygon[connector[0]]
                    v1 = conflictingArea.polygon[connector[0]+1]
                    # if both vertices of the connector are still here,
                    # the connector has survived.
                    if v0 in mergedPoly and v1 in mergedPoly:
                        newConnector = (mergedPoly.index(v0),connector[1])
                        mergedArea.connectors[key] = newConnector
            mergedAreas.append(mergedArea)

            # avoid a connector to reach over polygon end point index
            mc = max(c[0] for c in mergedArea.connectors.values())
            if mc >= len(mergedArea.polygon)-1:
                mergedArea.polygon = mergedArea.polygon[1:] + mergedArea.polygon[:1]
                for key,connector in mergedArea.connectors.items():
                    mergedArea.connectors[key] = (connector[0]-1,connector[1])

        # Now remove conflicting areas from list and add the merged areas
        for i in sorted(conflictingAreasIndcs, reverse = True):
                del self.intersectionAreas[i]
        self.intersectionAreas.extend(mergedAreas)

        # Finally, the trimmed centerlines of the ways are constructed
        for sectionNr,section in self.waySections.items():
            if not section.isValid:
                continue
            waySlice = None
            if section.trimT > section.trimS + 1.e-5:
                waySlice = section.polyline.trimmed(section.trimS,section.trimT)
                section_gn = TrimmedWaySection()

                section_gn.startConnected = self.sectionNetwork.borderlessOrder(section.originalSection.s) != 1
                section_gn.endConnected = self.sectionNetwork.borderlessOrder(section.originalSection.t) != 1

                section_gn.centerline = waySlice.verts
                section_gn.category = section.originalSection.category
                if section.isOneWay:
                    section_gn.nrOfLanes = (section.nrRightLanes)
                else:
                    section_gn.nrOfLanes = (section.nrLeftLanes, section.nrRightLanes)
                section_gn.width = section.leftWidth + section.rightWidth
                section_gn.tags = section.originalSection.tags
                # if section.turnParams:
                #     section_gn.endWidths = (section.leftWidth+section.turnParams[0], section.rightWidth+section.turnParams[1])
                self.waySectionLines[section.id] = section_gn
            # else:
            #     # already treated as reason for conflicting areas
