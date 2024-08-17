from collections import defaultdict
from itertools import tee, islice, cycle, permutations, accumulate
from statistics import median
from mathutils import Vector
import re

from app import AppType

from defs.road_polygons import ExcludedWayTags
from defs.way_cluster_params import minTemplateLength, minNeighborLength, searchDist

from way.item import Intersection, MinorIntersection, IntConnector, Section, Street, SideLane, SymLane
from way.way_network import WayNetwork, NetSection
from way.way_algorithms import createSectionNetwork
from way.way_properties import lanePattern

from lib.SweepIntersectorLib.SweepIntersector import SweepIntersector

from lib.CompGeom.StaticSpatialIndex import StaticSpatialIndex, BBox
from lib.CompGeom.algorithms import SCClipper
from lib.CompGeom.GraphBasedAlgos import DisjointSets
from lib.CompGeom.PolyLine import PolyLine
from lib.CompGeom.LinePolygonClipper import LinePolygonClipper


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
# ----------------------------------------------------------------


class StreetGenerator():
    
    def __init__(self, styleStore, getStyle, leftHandTraffic=True):
        self.styleStore = styleStore
        self.getStyle = getStyle
        self.leftHandTraffic = leftHandTraffic

        self.networkGraph = None
        self.sectionNetwork = None

        self.parallelSectionKeys = None
        self.internalTransitionSideLanes = dict()
        self.internalTransitionSymLanes = dict()
        self.intersections = dict()
        self.processedNodes = set()

        # If True: wayManager.getAllWays() else wayManager.getAllVehicleWays()
        self.allWays = True

    def do(self, manager):
        self.wayManager = manager
        self.waymap = manager.waymap
        self.intersections = manager.intersections
        self.transitionSideLanes = manager.transitionSideLanes
        self.transitionSymLanes = manager.transitionSymLanes
        self.wayClusters = manager.wayClusters
        self.waySectionLines = manager.waySectionLines

        NetSection.ID = 0   # This class variable of NetSection is not reset with new instance of StreetGenerator!!

        self.findSelfIntersections()
        self.createWaySectionNetwork()
        self.createEmptyWaymap()
        self.createSymSideLanes()
        self.updateIntersections()
        self.extendStreets()
        self.finalizeOutput()

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
                street.street = street # Fill superclass Item
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

    def extendStreets(self):
        def isMinorCategory(section):
            if not section.valid: return False
            return  section.category in  ['footway', 'cycleway','service'] or \
                    ('service' in section.tags and \
                    section.tags['service']=='driveway')

        def isMinorIntersection(intersection):
            minorCount = 0
            majorCount = 0
            majorCategories = set()
            majorIndices = []
            for indx, leaveWay in enumerate(intersection.leaveWays):
                if isMinorCategory(leaveWay.section):
                    minorCount += 1
                else:
                    majorCount += 1
                    majorCategories.add(leaveWay.section.category)
                    majorIndices.append(indx)
            return (majorCount == 2 and minorCount>0 and len(majorCategories)==1)

        intersectionsToReplace = []
        minorsToReInsert = []

        # Search for minor intersections
        for location, intersection in self.waymap.iterNodes(Intersection):
            if isMinorIntersection(intersection):

                # Find leaving major item (in conn)
                for conn in IntConnector.iterate_from(intersection.startConnector):
                    if conn.leaving and not isMinorCategory(conn.item.head):
                        break

                # Create minor intersection from Intersection
                minorIsect = MinorIntersection(intersection.location)

                # conn.item is the leaving major Street. Its content, the list of items,
                # will be filled into the arriving Street. This content is hold by <head>
                # and <tail> of item
                leavingStreet = conn.item
                minorIsect.succ = conn.item.head
                conn.item.head.pred = minorIsect

                # The circular list of connectors of the ordinary Intersection is
                # ordered counter-clockwise. When we start with a leaving section,
                # the first minor sections are on the left.
                for conn in IntConnector.iterate_from(conn.succ):
                    section = conn.item.head if conn.leaving else conn.item.tail
                    if isMinorCategory(section):
                        minorIsect.insertLeftConnector(conn)
                        minorsToReInsert.append(conn.item)
                    else:
                        break # the next major section

                # conn.item is now the arriving major Street. minorIsect is 
                # already linked with the list from the leaving Street.
                arrivingStreet = conn.item
                minorIsect.pred = conn.item.tail
                conn.item.tail.succ = minorIsect
                arrivingStreet.tail = leavingStreet.tail
                

                # Then, the minor sections on the right are collected
                for conn in IntConnector.iterate_from(conn.succ):
                    section = conn.item.head if conn.leaving else conn.item.tail
                    if isMinorCategory(section):
                        minorIsect.insertRightConnector(conn)
                        minorsToReInsert.append(conn.item)
                    else:
                        break # this is again the first major section

                # The content of the leaving major street is now appended 
                # after the minor intersection of the arriving street.
                # The attribute <street> of the super class Item has to be updated. 
                item = minorIsect.succ   # first item of former leaving street
                while item is not None:
                    item.street = arrivingStreet
                    item = item.succ

                # Finally, the minor intersection itself belngs to the arriving street.
                minorIsect.street = arrivingStreet 

                intersectionsToReplace.append( (intersection, location, arrivingStreet))

        for intersection, location, street in intersectionsToReplace:
            # Remove the original intersection with all its edges
            self.waymap.removeStreetNode(location)
            self.intersections.remove(intersection)

            # All items are now linked in <street>. Set sytle ??
            street.style = self.styleStore.get( self.getStyle(street) )
            street.setStyleBlockFromTop(street.style)

            # Reinsert <street>
            self.waymap.addSection(street)

        for street in minorsToReInsert:
            self.waymap.addSection(street)

        test = 1

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

            # TODO treat somehow short ways, this here does not work
            # shortWays = intersection.cleanShortWays(False)
            # if shortWays:
            #     for way in shortWays:
            #         way.section.valid = False
            #         way.section.polyline.plot('c',4)

            intersection.processIntersection()
            nr += 1

            # # DEBUG: Show intersections.
            # # The plotting functions for this debug part are at the end of this module
            # if False and self.app.type == AppType.commandLine:
            #     from debug import plt, plotPolygon
            #     plotPolygon(intersection.area,False,'k','r',1,True,0.4,999)

            self.processedNodes.add(location)
            # if intersection.area and intersection.leaveWays:
            if intersection.order > 1:
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

