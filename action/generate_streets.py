from collections import defaultdict
from itertools import tee, islice, cycle
from mathutils import Vector
import re

from app import AppType

from defs.road_polygons import ExcludedWayTags
from defs.way_cluster_params import minTemplateLength, minNeighborLength, searchDist, dbScanDist

from way.item import Intersection, IntConnector, Section, Street, SideLane, SymLane
from way.item.bundle import Bundle, mergePseudoMinors, removeSplittingStreets, orderHeadTail, \
                                    findInnerStreets, canBeMerged, mergeBundles, intersectBundles, endBundleIntersection
from way.way_network import WayNetwork, NetSection
from way.way_algorithms import createSectionNetwork
from way.way_properties import lanePattern

from lib.SweepIntersectorLib.SweepIntersector import SweepIntersector

from lib.CompGeom.StaticSpatialIndex import StaticSpatialIndex, BBox
from lib.CompGeom.algorithms import SCClipper
from lib.CompGeom.GraphBasedAlgos import DisjointSets
from lib.CompGeom.PolyLine import PolyLine
from lib.CompGeom.LinePolygonClipper import LinePolygonClipper
from lib.CompGeom.dbscan import dbClusterScan


# helper functions -----------------------------------------------
def pairs(iterable):
    # iterable -> (p0,p1), (p1,p2), (p2, p3), ...
    p1, p2 = tee(iterable)
    next(p2, None)
    return zip(p1,p2)

def isEdgy(polyline):
    vu = polyline.unitVectors()
    for v1,v2 in pairs(vu):
        if abs(v1.cross(v2)) > 0.6:
            return True
    return False

def _pseudoangle(d):
    p = d[0]/(abs(d[0])+abs(d[1])) # -1 .. 1 increasing with x
    return 3 + p if d[1] < 0 else 1 - p 
# ----------------------------------------------------------------


class StreetGenerator():
    
    def __init__(self, styleStore, getStyle, leftHandTraffic=True):
        self.styleStore = styleStore
        self.getStyle = getStyle
        self.leftHandTraffic = leftHandTraffic

        self.networkGraph = None
        self.sectionNetwork = None

        self.internalTransitionSideLanes = dict()
        self.internalTransitionSymLanes = dict()
        self.intersections = dict()
        self.processedNodes = set()

        # If True: wayManager.getAllWays() else wayManager.getAllVehicleWays()
        self.allWays = True

    def do(self, manager):
        self.wayManager = manager
        self.waymap = manager.waymap
        self.majorIntersections = manager.majorIntersections
        self.minorIntersections = manager.minorIntersections
        self.transitionSideLanes = manager.transitionSideLanes
        self.transitionSymLanes = manager.transitionSymLanes
        self.streets = manager.streets
        self.bundles = manager.bundles
        self.wayClusters = manager.wayClusters
        self.waySectionLines = manager.waySectionLines

        NetSection.ID = 0   # This class variable of NetSection is not reset with new instance of StreetGenerator!!

        self.findSelfIntersections()
        self.createWaySectionNetwork()
        self.createEmptyWaymap()
        self.createSymSideLanes()
        self.updateIntersections()
        self.createStreets()
        # self.circularStreets()
        self.createParallelStreets()
        self.createBundles()
        self.mergeBundles()
        self.createBundleIntersections()

    def findSelfIntersections(self):
        uniqueSegments = defaultdict(set)
        if self.allWays: 
            getWays = self.wayManager.getAllWays()
        else:
            getWays = self.wayManager.getAllVehicleWays()
        for way in getWays:
            # ExcludedWayTags is defined in <defs>.
            # It is used also in createWaySectionNetwork().
            if [tag for tag in ExcludedWayTags if tag in way.category]:
                continue
            for segment in way.segments:
                v1, v2 = (segment.v1[0],segment.v1[1]),  (segment.v2[0],segment.v2[1])
                if v1 not in uniqueSegments.get(v2,[]):
                    uniqueSegments[v1].add(v2)
        cleanedSegs = [(v1,v2) for v1 in uniqueSegments for v2 in uniqueSegments[v1]]

        intersector = SweepIntersector()
        self.intersectingSegments = intersector.findIntersections(cleanedSegs)

    # Creates the network graph <self.sectionNetwork> for way-sections (ways between crossings)
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
                street.insertEnd(section)
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

                self.waymap.addNode(Intersection(section.src))
                self.waymap.addNode(Intersection(section.dst))
                street.street = street # Fill superclass Item
                self.waymap.addEdge(street)

                # If there are corners, the section must be split to enable finding of parallel sections
                # corners = section.polyline.getCorners(0.6) if section.category in ['footway', 'cycleway'] else []

                # if False and corners and self.app.type == AppType.commandLine:
                #     from debug import plt, plotPureNetwork
                #     for nextCorner in corners:
                #         c = section.polyline[nextCorner]
                #         plt.plot(c[0],c[1],'ro',markersize=8,zorder=999,markeredgecolor='red', markerfacecolor='none')

                # if corners:
                #     corners.append(len(section.polyline)-1)
                #     self.waymap.addNode(Intersection(section.src))
                #     lastCorner = 0
                #     for nextCorner in corners:
                #         splitline = PolyLine( section.polyline[lastCorner:nextCorner+1] )
                #         subsection = Section(net_section,splitline,self.sectionNetwork)
                #         subsection.setSectionAttributes(oneway,fwdPattern,bwdPattern,bothLanes,props)

                #         street = Street(subsection.src, subsection.dst)
                #         street.append(subsection)                       
                #         street.setStyle(streetStyle)

                #         self.waymap.addNode(Corner(subsection.dst))
                #         self.waymap.addEdge(street)
                #         lastCorner = nextCorner
                #     self.waymap.replaceStreetNodeBy(Intersection(subsection.dst))
                # else:
                #     Add section, we do not yet know the type of the intersections
                #     self.waymap.addNode(Intersection(section.src))
                #     self.waymap.addNode(Intersection(section.dst))

                #     street = Street(section.src, section.dst)
                #     section.street = street
                #     street.append(section)
                #     street.setStyle(streetStyle)
                
                #     self.waymap.addEdge(street)

        # Add ways to intersections
        for location, intersection in self.waymap.iterNodes(Intersection):
            inStreets, outStreets = self.waymap.getInOutEdges(location)
            intersection.update(inStreets, outStreets)

    def createSymSideLanes(self):
        def findSymSideIsects(street):
            srcIsectObj = self.waymap.getNode(street.src)
            if srcIsectObj and isinstance(srcIsectObj['object'], Intersection):
                srcIsect = srcIsectObj['object']
                if srcIsect:
                    if srcIsect.order==2:
                        street0 = srcIsect.leaveWays[0].section.street
                        street1 = srcIsect.leaveWays[1].section.street
                        arriving, leaving = (street0, street1) if street0.dst == street1.src else (street1, street0)
                        # Check directions
                        if arriving.dst == srcIsect.location and leaving.src == srcIsect.location:
                            hasTurns = bool( re.search(r'[^N]', leaving.head.lanePatterns[0] ) )
                            srcIsect = {'isect':srcIsect, 'arriving':arriving, 'leaving':leaving, 'hasTurns':hasTurns}
                        else:
                            srcIsect = None
                    else:
                        srcIsect = None
            else:
                srcIsect = None

            dstIsectObj = self.waymap.getNode(street.dst)
            if dstIsectObj and isinstance(dstIsectObj['object'], Intersection):
                dstIsect = dstIsectObj['object']
                if dstIsect:
                    if dstIsect.order==2:
                        street0 = dstIsect.leaveWays[0].section.street
                        street1 = dstIsect.leaveWays[1].section.street
                        arriving, leaving = (street0, street1) if street0.dst == street1.src else (street1, street0)
                        # Check directions
                        if arriving.dst == dstIsect.location and leaving.src == dstIsect.location:
                            hasTurns = bool( re.search(r'[^N]', leaving.head.lanePatterns[0] ) )
                            dstIsect = {'isect':dstIsect, 'arriving':arriving, 'leaving':leaving, 'hasTurns':hasTurns}
                        else:
                            dstIsect = None
                    else:
                        dstIsect = None
            else:
                dstIsect = None

            return srcIsect, dstIsect

        processedStreets = set()
        nodesToRemove = []
        longStreets = []
        for _, _, _, street in self.waymap.edges(data='object',keys=True):
            if street in processedStreets:
                continue

            srcIsectInit, dstIsectInit = findSymSideIsects(street)
            hasSymSide = srcIsectInit or dstIsectInit

            if hasSymSide:
                longStreet = Street(street.src,street.dst)
                longStreet.insertStreetEnd(street)
                processedStreets.add(street)
                longStreet.pred = street.pred

                if srcIsectInit: # # SymSide intersection at the at the front of this street
                    if srcIsectInit['hasTurns']:
                        newLane = SideLane(srcIsectInit['isect'].location, srcIsectInit['arriving'].head, srcIsectInit['leaving'].head)
                        self.transitionSideLanes.append(newLane)
                    else:
                        newLane = SymLane(srcIsectInit['isect'].location, srcIsectInit['arriving'].head, srcIsectInit['leaving'].head)
                        self.transitionSymLanes.append(newLane)
                    newLane.street = longStreet
                    longStreet.insertFront(newLane)   # insert new lane object
                    nodesToRemove.append(srcIsectInit['isect'])
                    srcIsectCurr = srcIsectInit
                    while True: # Continue, if there are more SymSide intersections
                        prevStreet = srcIsectCurr['arriving']
                        if prevStreet in processedStreets:
                            break
                        longStreet.insertStreetFront(prevStreet)
                        processedStreets.add(prevStreet)
                        srcIsectCurr, _ = findSymSideIsects(prevStreet)
                        if not srcIsectCurr:
                            break
                        if srcIsectCurr['hasTurns']:
                            newLane = SideLane(srcIsectCurr['isect'].location, srcIsectCurr['arriving'].head, srcIsectCurr['leaving'].head)
                            self.transitionSideLanes.append(newLane)
                        else:
                            newLane = SymLane(srcIsectCurr['isect'].location, srcIsectCurr['arriving'].head, srcIsectCurr['leaving'].head)
                            self.transitionSymLanes.append(newLane)
                        newLane.street = longStreet
                        longStreet.insertFront(newLane)   # insert minor intersection object
                        nodesToRemove.append(srcIsectInit['isect'])

                if dstIsectInit: # # SymSide intersection at the at the end of this street
                    if dstIsectInit['hasTurns']:
                        newLane = SideLane(dstIsectInit['isect'].location, dstIsectInit['arriving'].head, dstIsectInit['leaving'].head)
                        self.transitionSideLanes.append(newLane)
                    else:
                        newLane = SymLane(dstIsectInit['isect'].location, dstIsectInit['arriving'].head, dstIsectInit['leaving'].head)
                        self.transitionSymLanes.append(newLane)
                    newLane.street = longStreet
                    longStreet.insertEnd(newLane)   # insert new lane object
                    nodesToRemove.append(dstIsectInit['isect'])
                    dstIsectCurr = dstIsectInit
                    while True: # Continue, if there are more SymSide intersections
                        nextStreet = dstIsectCurr['leaving']
                        if nextStreet in processedStreets:
                            break
                        longStreet.insertStreetEnd(nextStreet)
                        processedStreets.add(nextStreet)

                        dstIsectCurr, _ = findSymSideIsects(nextStreet)
                        if not dstIsectCurr:
                            break
                        if dstIsectCurr['hasTurns']:
                            newLane = SideLane(dstIsectCurr['isect'].location, dstIsectCurr['arriving'].head, dstIsectCurr['leaving'].head)
                            self.transitionSideLanes.append(newLane)
                        else:
                            newLane = SymLane(dstIsectCurr['isect'].location, dstIsectCurr['arriving'].head, dstIsectCurr['leaving'].head)
                            self.transitionSymLanes.append(newLane)
                        newLane.street = longStreet
                        longStreet.insertEnd(newLane)   # insert minor intersection object
                        nodesToRemove.append(dstIsectInit['isect'])

                longStreets.append(longStreet)

        # At this stage, intersections do not yet have connectors. They will get them 
        # when processIntersection() is called (which occurs at self.updateIntersections).
        # But the leaving ways structure of the intersections at the end of the new
        # long Street needs to be updated.
        streetIDs = [s.id for s in processedStreets]
        for longStreet in longStreets:
            node = self.waymap.getNode(longStreet.src)
            if node:
                srcIsect = node['object']
                for way in srcIsect.leaveWays:
                    if way.street.id in streetIDs:
                        way.street = longStreet
            node = self.waymap.getNode(longStreet.dst)
            if node:
                dstIsect = node['object']
                for way in dstIsect.leaveWays:
                    if way.street.id in streetIDs:
                        way.street = longStreet

        for node in nodesToRemove:
            self.waymap.removeNode(node.location)

        for longStreet in longStreets:
            self.waymap.addEdge(longStreet)

    def updateIntersections(self):
        # At this stage, it is assumed, that SideLanes, SymLanes are already built
        # and that their nodes are stored in <self.processedNodes>.
        for location, intersection in self.waymap.iterNodes(Intersection):
            if location in self.processedNodes:
                continue

            intersection.processIntersection()
            if intersection.isMinorIntersection():
                intersection.transformToMinor()
                # see https://github.com/prochitecture/blosm/issues/106#issuecomment-2305297075
                if intersection.isMinor:
                    self.minorIntersections[intersection.location] = intersection
                else:
                    if intersection.order > 1:
                        self.majorIntersections[intersection.location] = intersection
            else:
                if intersection.order > 1:
                    self.majorIntersections[intersection.location] = intersection

            self.processedNodes.add(location)

    def createStreets(self):
        for street in self.wayManager.iterStreetsFromWaymap():
            self.streets[street.id] = street

        doDebug = False and self.app.type == AppType.commandLine

        if doDebug:
            from debug import plt, plotQualifiedNetwork, randomColor, plotEnd

            def plotStreet(street,color, arrows=False):
                for item in street.iterItems():
                    if isinstance(item, Section):
                        width = 6 if item.id==398 else 3
                        if arrows:
                            item.polyline.plotWithArrows(color,width,0.5,'solid',False,950)
                        else:
                            item.polyline.plot(color,width,'solid',False,950)

            plotStreet(self.streets[564],'red')
            p = self.streets[564].dst
            # plotStreet(self.streets[510],'blue')
            plt.plot(p[0],p[1],'co',markersize=12,alpha=0.4)
            plotQualifiedNetwork(self.sectionNetwork)
            plotEnd()

    def circularStreets(self):
        doDebug = True and self.app.type == AppType.commandLine

        def tagsOfStreet(street):
            for item in street.iterItems():
                if isinstance(item, Section):
                    return item.tags

        def centerlineOfStreet(street):
            # Find the centerline of the whole street.
            centerlineVerts = []
            for item in street.iterItems():
                if isinstance(item, Section):
                    centerlineVerts.extend( item.centerline)

            # Remove duplicates and create polyLine
            centerlineVerts = list(dict.fromkeys(centerlineVerts))
            centerline = PolyLine(centerlineVerts)
            return centerline, centerlineVerts


        # Add the bounding boxes of all streets to the index.
        circularStreets = []
        for street in self.wayManager.iterStreets():
            tags = tagsOfStreet(street)
            possibleCircular = 'junction' in tags and tags['junction']=='circular'
            possibleCircular = possibleCircular or ('junction' in tags and tags['junction']=='roundabout')

            if possibleCircular:
                circularStreets.append(street)

        if doDebug:
            from debug import plt, plotQualifiedNetwork, randomColor, plotEnd

            plotQualifiedNetwork(self.sectionNetwork,False)
            for street in circularStreets:
                centerline,verts = centerlineOfStreet(street)
                centerline.plot('red',2,'solid')
                centerline.plotWithArrows('red',1,0.5,'solid',False,950)
            plotEnd()

    def createParallelStreets(self):
        doDebug = True and self.app.type == AppType.commandLine

        def categoryOfStreet(street):
            for item in street.iterItems():
                if isinstance(item, Section):
                    break
            return item.category
        
        def centerlineOfStreet(street):
            # Find the centerline of the whole street.
            centerlineVerts = []
            for item in street.iterItems():
                if isinstance(item, Section):
                    centerlineVerts.extend( item.centerline)

            # Remove duplicates and create polyLine
            centerlineVerts = list(dict.fromkeys(centerlineVerts))
            centerline = PolyLine(centerlineVerts)
            return centerline, centerlineVerts

        # Spatial index (R-tree) of candidate Streets
        candidateIndex = StaticSpatialIndex()

        # Dictionary from index in candidateIndex to street.
        index2Street = dict()

        # The bounding boxes of the streets. The dictionary key is <dictKey>.
        boxes = dict()

        # Some computed attributes of the streets. The dictionary key is <dictKey>.
        attributes = dict()

        # Add the bounding boxes of all streets to the index.
        for street in self.wayManager.iterStreets():
            # Some categories are excluded.
            category =  categoryOfStreet(street)
            if category in ('steps', 'footway', 'cycleway', 'path', 'service'):
                continue

            # Find the centerline of the whole street.
            centerline, centerlineVerts = centerlineOfStreet(street)

            # Exclude if too curvy
            ds = (centerline[0]-centerline[-1]).length / centerline.length()
            if isEdgy(centerline) and ds < 0.9:
                continue
            
            # Exclude if too short
            if centerline.length() < min(minTemplateLength,minNeighborLength):
                continue

            # Find bounding box and fill in index
            min_x = min(v[0] for v in centerlineVerts)
            min_y = min(v[1] for v in centerlineVerts)
            max_x = max(v[0] for v in centerlineVerts)
            max_y = max(v[1] for v in centerlineVerts)
            bbox = BBox(None,min_x,min_y,max_x,max_y)
            index = candidateIndex.add(min_x,min_y,max_x,max_y)
            index2Street[index] = street
            bbox.index = index
            boxes[street] = (min_x,min_y,max_x,max_y)
            attributes[street] = ( category, centerline, centerlineVerts )

        # Finalize the index for usage.   
        candidateIndex.finish()

        # This is the structure we use to collect the parallel streets
        self.parallelStreets = DisjointSets()

        # Every street that was inserted into the spatial index becomes now 
        # as sample. We expand it to a buffer area around it.
        for sampleStreet in self.wayManager.iterStreets():
            # Use only accepted streets
            if sampleStreet in boxes:
                sampleCategory, sampleCenterline, _ = attributes[sampleStreet]
                # Create buffer polygon around the sample street with a width according
                # to the category of the sample street.
                bufferWidth = searchDist[sampleCategory]
                bufferPoly = sampleCenterline.buffer(bufferWidth,bufferWidth)

                # Create a line clipper using this polygon.
                clipper = LinePolygonClipper(bufferPoly.verts)

                # Get neighbors of this sample street from the static spatial index, using its
                # bounding box, expanded by the buffer width as additional search range.
                min_x,min_y,max_x,max_y = boxes[sampleStreet]
                results = stack = []
                neighborIndices = candidateIndex.query(min_x-bufferWidth,min_y-bufferWidth,
                                               max_x+bufferWidth,max_y+bufferWidth,results,stack)

                # Now test all these neighbors of the sample street for parallelism.
                for neigborIndex in neighborIndices:
                    neighborStreet = index2Street[neigborIndex]
                    if neighborStreet == sampleStreet:
                        continue # Skip, the sample street is its own neighbor.

                    _, neighCenterline, neighCenterlineVerts = attributes[neighborStreet]

                    # If the centerline of this neighbor is longer than a minimal length, ...
                    if neighCenterline.length() > minNeighborLength:
                        # ... then clip it with the buffer polygon
                        inLine, inLineLength, nrOfON = clipper.clipLine(neighCenterlineVerts)

                        if inLineLength/neighCenterline.length() < 0.1:
                            continue # discard short inside lines. At least 10% must be inside.

                        # To check the quality of parallelism, some kind of "slope" relative 
                        # to the template's line is evaluated.
                        p1, d1 = sampleCenterline.distTo(inLine[0][0])     # distance to start of inLine
                        p2, d2 = sampleCenterline.distTo(inLine[-1][-1])   # distance to end of inLine
                        slope = abs(d1-d2)/inLineLength if inLineLength else 1.

                        # Conditions for acceptable inside line.
                        # plotPureNetwork(self.sectionNetwork)
                        if slope < 0.15 and min(d1,d2) <= bufferWidth and nrOfON <= 2:
                            # Accept this pair as parallel.
                            self.parallelStreets.addSegment(sampleStreet,neighborStreet)

        # DEBUG: Show clusters of parallel way-sections.
        # The plotting functions for this debug part are at the end of this module
        if doDebug:
            from debug import plt, plotQualifiedNetwork, randomColor, plotEnd

            inBundles = False

            if not inBundles:
                plotQualifiedNetwork(self.sectionNetwork,False)
            colorIter = randomColor(19)
            for bIndx,streets in enumerate(self.parallelStreets):
                # if bIndx not in [0,1]:
                #     continue
                if inBundles:
                    plotQualifiedNetwork(self.sectionNetwork,False)
                    plt.title("Bundle "+str(bIndx))
                color = next(colorIter)
                allVerts = []
                for street in streets:
                    width = 2
                    if inBundles: 
                        color = "red"
                        width = 3
                    centerline,verts = centerlineOfStreet(street)
                    allVerts.extend(verts)
                    centerline.plot(color,width,'solid')
                    centerline.plotWithArrows(color,1,0.5,'solid',False,950)
                    if inBundles: 
                        plt.scatter(centerline[0][0], centerline[0][1], s=80, facecolors='none', edgecolors='g',zorder=999)
                        plt.scatter(centerline[-1][0], centerline[-1][1], s=80, facecolors='none', edgecolors='g',zorder=999)
                        # plt.plot(polyline[0][0], polyline[0][1], 'go', markersize=8,zorder=999)
                        # plt.plot(polyline[-1][0], polyline[-1][1], 'go', markersize=8,zorder=999)
                center = sum(allVerts,Vector((0,0)))/len(allVerts)
                plt.text(center[0],center[1],str(bIndx),fontsize=10)
                if inBundles:
                    plotEnd()
            if not inBundles:
                plotEnd()
            # END DEBUG
                            
    def createBundles(self):
        doDebug = False and self.app.type == AppType.commandLine

        if doDebug:
            from debug import plt, plotQualifiedNetwork, randomColor, plotEnd
            colorIter = randomColor(19)

            def plotStreetGroup(streetGroup,color='blue',doEnd=False):
                for street in streetGroup:
                    allVertices = []
                    # color = next(colorIter)
                    for item in street.iterItems():
                        if isinstance(item, Section):
                            item.polyline.plotWithArrows(color,2,0.5,'solid',False,950)
                            allVertices.extend(item.centerline)
                    if len(allVertices):
                        c = sum(allVertices,Vector((0,0))) / len(allVertices)
                        plt.text(c[0]+2,c[1]-2,'S '+str(street.id),color='k',fontsize=10,zorder=130,ha='left', va='top', clip_on=True)

                if doEnd:
                    plotEnd()

            def plotStreets(streets,doEnd=True):
                for street in streets:
                    allVertices = []
                    color = next(colorIter)
                    for item in street.iterItems():
                        if isinstance(item, Section):
                            item.polyline.plotWithArrows(color,1,0.5,'solid',False,950)
                            allVertices.extend(item.centerline)
                    if len(allVertices):
                        c = sum(allVertices,Vector((0,0))) / len(allVertices)
                        plt.text(c[0]+2,c[1]-2,'S '+str(street.id),color='k',fontsize=8,zorder=130,ha='left', va='top', clip_on=True)

                streetsThatEndHere = defaultdict(list)  # The streets that end at the key location
                intersectionOfThisEnd = dict()          # The intersection at the key location
                for street in streets:
                    streetsThatEndHere[street.src].append(street)
                    streetsThatEndHere[street.dst].append(street)
                    intersectionOfThisEnd[street.src] = street.pred.intersection if street.pred else None
                    intersectionOfThisEnd[street.dst] = street.succ.intersection if street.succ else None

                innerOrder = dict()     # number of leaving streets, that belong to the bundle
                outerOrder = dict()     # number of leaving streets, that do not belong to the bundle
                isectOrder = dict()
                for location, intersection in intersectionOfThisEnd.items():
                    innerOrder[location] = len(streetsThatEndHere[location])
                    outerOrder[location] = sum(1 for connector in intersection if connector.item not in streets) if intersection else 0
                    isectOrder[location] = intersection.order if intersection else None

                    p = location
                    if innerOrder[location]==1:   # endpoint
                        plt.plot(p[0],p[1],'rx',markersize=8,zorder=999)
                    elif innerOrder[location]==2 and outerOrder[location]==1: # major to minor and connect
                        plt.plot(p[0],p[1],'bo',markersize=8,zorder=999)
                    elif outerOrder[location]==0: # inner intersection, split/merge
                        plt.plot(p[0],p[1],'cs',markersize=8,zorder=999)
                        plt.text(p[0],p[1]+2,str(isectOrder[location]),color='red',zorder=999)
                        plt.text(p[0],p[1],' '+str(innerOrder[location]),color='red',zorder=999)
                        plt.text(p[0],p[1],'     '+str(outerOrder[location]),color='blue',zorder=999)
                    else:
                        plt.plot(p[0],p[1],'mo',markersize=8,zorder=999)
                        plt.text(p[0],p[1]+2,str(isectOrder[location]),color='red',zorder=999)
                        plt.text(p[0],p[1],' '+str(innerOrder[location]),color='red',zorder=999)
                        plt.text(p[0],p[1],'     '+str(outerOrder[location]),color='blue',zorder=999)

                    # plt.text(p[0],p[1]+5,str(isectOrder[location]),color='red',zorder=999)
                    # plt.text(p[0],p[1],'    '+str(innerOrder[location]),color='red',zorder=999)
                    # plt.text(p[0],p[1],'        '+str(outerOrder[location]),color='blue',zorder=999)

                if doEnd:
                    plotEnd()

        # The disjoint set <self.parallelStreets> is not a convenient container for
        # the streets group. Mainly, modified streets cannot be stored there.
        streetGroups = []
        for streetGroup in self.parallelStreets:
            streetGroups.append(streetGroup)

        for streetGroup in streetGroups:
            # see https://github.com/prochitecture/blosm/issues/104#issuecomment-2322836476
            # Major intersections in bundles, with only one side street, are merged into a long street,
            # similar to minor intersections.
            mergePseudoMinors(self, streetGroup)


        # Find intersections between streets of different groups.
        # The dictionary key is the location of the intersection
        # and its  lists contain the indices of the groups that
        # intersect there.
        groupIntersections = defaultdict(list)
        for gIndex,streetGroup in enumerate(streetGroups):
            for street in streetGroup:
                for p in [street.src, street.dst]:
                    groupIntersections[p].append(gIndex)

        newGroups = []
        delGroups = []
        for gIndex,streetGroup in enumerate(streetGroups):
            # Sometimes, createParallelStreets() is not stopped by intersections with
            # other bundles and delivers streets, that pass these intersections. In
            # such a situation, its  proposed group has to be split into two groups, while
            # the splitting streets at the intersection have to be removed.
            wasSplitted, splittedGroups, splittingStreets = removeSplittingStreets(self,gIndex,streetGroup,groupIntersections)
            # if doDebug:
            #     if wasSplitted:
            #         for group in splittedGroups:
            #             plotStreetGroup(streetGroup, 'blue', False)
            #             plotStreetGroup(splittingStreets, 'whitesmoke', False)
            #             plotStreetGroup(group,'red',True)   


            if wasSplitted:
                newGroups.extend(splittedGroups)
                delGroups.append(streetGroup)
                for street in splittingStreets:
                    if street.id in self.streets:
                        del self.streets[street.id]      
        streetGroups.extend(newGroups)
        for group in delGroups:
            streetGroups.remove(group)

        # if doDebug:
        #     for group in streetGroups:
        #         plotStreetGroup(group,'red',True)




        for gIndex,streetGroup in enumerate(streetGroups):
            head, tail = orderHeadTail(streetGroup)

            if doDebug:
                plotQualifiedNetwork(self.sectionNetwork)
                plotStreetGroup(streetGroup)  
                for indx in range(len(head)):  
                    item = head[indx]
                    p = item['firstVert']
                    # plt.text(p[0],p[1],'  '+str(item['i']),fontsize=12)
                    plt.plot(p[0],p[1],'coral',marker='o',markersize=14,zorder=998)
                    plt.text(p[0],p[1],'H'+str(indx),fontsize=10,zorder=999,horizontalalignment='center',verticalalignment='center')

                for indx in range(len(tail)):
                    item = tail[indx]
                    p = item['firstVert']
                    # plt.text(p[0],p[1],'  '+str(item['i']),fontsize=12)
                    plt.plot(p[0],p[1],'skyblue',marker='o',markersize=14,zorder=998)
                    plt.text(p[0],p[1],'T'+str(indx),fontsize=10,zorder=999,horizontalalignment='center',verticalalignment='center')

                plotEnd()

            innerStreets = findInnerStreets(streetGroup,self.leftHandTraffic)

            if doDebug:
                plotQualifiedNetwork(self.sectionNetwork)
                plotStreetGroup(streetGroup)  
                for indx in range(len(head)):  
                    item = head[indx]
                    p = item['firstVert']
                    # plt.text(p[0],p[1],'  '+str(item['i']),fontsize=12)
                    plt.plot(p[0],p[1],'coral',marker='o',markersize=14,zorder=998)
                    plt.text(p[0],p[1],'H'+str(indx),fontsize=10,zorder=999,horizontalalignment='center',verticalalignment='center')

                for indx in range(len(tail)):
                    item = tail[indx]
                    p = item['firstVert']
                    # plt.text(p[0],p[1],'  '+str(item['i']),fontsize=12)
                    plt.plot(p[0],p[1],'skyblue',marker='o',markersize=14,zorder=998)
                    plt.text(p[0],p[1],'T'+str(indx),fontsize=10,zorder=999,horizontalalignment='center',verticalalignment='center')

                for street in innerStreets:
                    allVertices = []
                    for item in street.iterItems():
                        if isinstance(item, Section):
                            item.polyline.plot('red',2,'solid',False,999)
                            allVertices.extend(item.centerline)
                    if len(allVertices):
                        c = sum(allVertices,Vector((0,0))) / len(allVertices)
                        plt.text(c[0],c[1],'S '+str(street.id),color='k',fontsize=8,zorder=130,ha='left', va='top', clip_on=True)


                # plt.title(str(indxG))
                plotEnd()

            # ToDo: street.pred, street.succ ??
            bundle = Bundle()
            for item in head:
                street = item['street']
                street.bundle = bundle
                bundle.streetsHead.append(street)
                bundle.headLocs.append(item['firstVert'])
                if street.id in self.streets:
                    del self.streets[street.id]
            for item in tail:
                street = item['street']
                street.bundle = bundle
                bundle.streetsTail.append(street)
                bundle.tailLocs.append(item['firstVert'])
                if street.id in self.streets:
                    del self.streets[street.id]
            self.bundles[bundle.id] = bundle
            for street in innerStreets:
                street.bundle = bundle

    def mergeBundles(self):
        # Find all ends of streets of all bundles and cluster them to groups, 
        # that are potential intersections.
        endPoints = []
        for id, bundle in self.bundles.items():
            for end, street in zip(bundle.headLocs,bundle.streetsHead):
                endPoints.append( (end,{'end':end, 'type':'head', 'street':street, 'bundle':bundle}) )
            for end, street in zip(bundle.tailLocs,bundle.streetsTail):
                endPoints.append( (end,{'end':end, 'type':'tail', 'street':street, 'bundle':bundle}) )
        isectCandidates = dbClusterScan(endPoints, dbScanDist, 2)

        toBeMerged = []
        for candidates in isectCandidates:
            involvedBundles = defaultdict(list)
            for _,cand in candidates:
                data = {'end':cand['end'], 'type':cand['type'], 'street':cand['street'], 'bundle':cand['bundle']}
                involvedBundles[cand['bundle']].append(data)

            if len(involvedBundles) == 2:
                if len(candidates) == 2:
                    continue
                if canBeMerged(self, involvedBundles):
                    toBeMerged.append(involvedBundles)

        for involvedBundles in toBeMerged:
            mergeBundles(self,involvedBundles)

    def createBundleIntersections(self):
        doDebug = True and self.app.type == AppType.commandLine
        if doDebug:
            from debug import plt, plotQualifiedNetwork, randomColor, plotEnd

        toBeIntersected = []

        endPoints = []
        for id, bundle in self.bundles.items():
            for end, street in zip(bundle.headLocs,bundle.streetsHead):
                endPoints.append( (end,{'end':end, 'type':'head', 'street':street, 'bundle':bundle}) )
            for end, street in zip(bundle.tailLocs,bundle.streetsTail):
                endPoints.append( (end,{'end':end, 'type':'tail', 'street':street, 'bundle':bundle}) )
        isectCandidates = dbClusterScan(endPoints, dbScanDist, 2)

        # <isectCandidates> is a list of potential intersection candidates. 
        # These candidates are lists of dictionaries, that hold the location of the
        # streets end, the street's end type ('head' or 'tail'), the street instance
        # itself and the bundle, they belong to.
        for candidates in isectCandidates:
            involvedBundles = defaultdict(list)
            for _,cand in candidates:
                data = {'end':cand['end'], 'type':cand['type'], 'street':cand['street'], 'bundle':cand['bundle']}
                involvedBundles[cand['bundle']].append(data)

            nrOfBundles = len(involvedBundles)
            involvedBundleIDs = set( b.id for b,_ in involvedBundles.items())
            involvedBundleTypes = []
            for _,data in involvedBundles.items():
                involvedBundleTypes.extend( [d['type'] for d in data] )

            if nrOfBundles==1:
                # These are ends of bundles. Because these ar not detected reliably
                # here, they are processed at the end of this method.
                pass

            if nrOfBundles==2:
                # These are bundles, that touch each other. If there is no street
                # to the inner side, they can be merged using pseudo minors.
                # Else, an intersection needs to be created. 
                if len(candidates) == 2:
                    bundleIDs = [b.id for b,_ in involvedBundles.items()]
                    print('Single common bundle end between bundles', bundleIDs)
                    continue

                toBeIntersected.append(involvedBundles)
                if doDebug:
                    from lib.CompGeom.algorithms import circumCircle
                    ends = set()
                    for bundle,data in involvedBundles.items():
                        ends = ends.union( set(item['end'] for item in data) )
                    center,radius = circumCircle(list(ends))
                    circle = plt.Circle(center, radius*1.1, color='g', alpha=0.6)
                    plt.gca().add_patch(circle)

            if nrOfBundles>2:
                toBeIntersected.append(involvedBundles)

                if doDebug:
                    from lib.CompGeom.algorithms import circumCircle

                    ends = set()
                    for bundle,data in involvedBundles.items():
                        types = set(item['type'] for item in data)
                        if len(types)<2:    
                            ends = ends.union( set(item['end'] for item in data) )

                    center,radius = circumCircle(list(ends))
                    circle = plt.Circle(center, radius*1.1, color='g', alpha=0.6)
                    plt.gca().add_patch(circle)

        for involvedBundles in toBeIntersected:
            intersectBundles(self, involvedBundles)

        # Finally process open Bundle ends.
        for id,bundle in self.bundles.items():
            if not bundle.pred or not bundle.succ:

                if doDebug:
                    from lib.CompGeom.algorithms import circumCircle
                    if not bundle.pred:
                        ends = set(bundle.headLocs)

                        center,radius = None, None
                        if len(ends)>1:
                            center,radius = circumCircle(list(ends))
                        elif len(ends)==1:
                            center,radius = next(iter(ends)), 5.
                        else:
                            pass
                        if center:
                            circle = plt.Circle(center, radius*1.1, color='orange', alpha=0.6)
                            plt.gca().add_patch(circle)

                    if not bundle.succ:
                        ends = set(bundle.tailLocs)

                        center,radius = None, None
                        if len(ends)>1:
                            center,radius = circumCircle(list(ends))
                        elif len(ends)==1:
                            center,radius = next(iter(ends)), 5.
                        else:
                            pass
                        if center:
                            circle = plt.Circle(center, radius*1.1, color='orange', alpha=0.6)
                            plt.gca().add_patch(circle)

                endBundleIntersection(self, bundle)

        TEST=1


 
