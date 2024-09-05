from collections import defaultdict
from itertools import tee
from mathutils import Vector
import re

from app import AppType

from defs.road_polygons import ExcludedWayTags
from defs.way_cluster_params import minTemplateLength, minNeighborLength, searchDist

from way.item import Intersection, Section, Street, SideLane, SymLane
from way.item.bundle import Bundle, mergePseudoMinors, locationsInGroup, forwardOrder
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
        # self.createParallelStreets()
        # self.createBundles()

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

    def createParallelStreets(self):
        from debug import plt, plotPureNetwork, randomColor, plotEnd
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
        if self.app.type == AppType.commandLine:
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
        from debug import plt, plotQualifiedNetwork, randomColor, plotEnd
        colorIter = randomColor(19)

        def plotStreets(streets,doEnd=True):
            for street in streets:
                allVertices = []
                color = next(colorIter)
                for item in street.iterItems():
                    if isinstance(item, Section):
                        item.polyline.plotWithArrows(color,1,0.5,'solid',False,999)
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

            if doEnd:
                plotEnd()

        for indxG,streetGroup in enumerate(self.parallelStreets):

            # see https://github.com/prochitecture/blosm/issues/104#issuecomment-2322836476
            # Major intersections in bundles, with only one side street, are merged into a long street,
            # similar to minor intersections.
            streetEnds, intersections = locationsInGroup(streetGroup)
            streetGroup = mergePseudoMinors(self, streetGroup, streetEnds, intersections)

            # compute this again, because streetGroup may have been changed
            streetEnds, intersections = locationsInGroup(streetGroup)

            # Check if intermediate intersections exist
            hasIntermediates = any(len(end)>1 for end in streetEnds.values())

            if hasIntermediates:
                plotQualifiedNetwork(self.sectionNetwork)
                plt.title(str(indxG))
                plotStreets(streetGroup)
            else:
                # Try to find starts and ends of the streets relative to the bundle. This 
                # 'forwardOrder' determines also the direction of the bundle. See this
                # function in uitilities of Bundle.
                head = []
                tail = []
                for street in streetGroup:
                    fwd = forwardOrder(street.src,street.dst)
                    h, t = (street.src, street.dst) if fwd else (street.dst, street.src)
                    if len(streetEnds[h])==1:  
                            head.append({'street':street, 'firstVert':h, 'lastVert':t})
                    if len(streetEnds[t])==1:  
                            tail.append({'street':street, 'firstVert':t, 'lastVert':h})

                # The streets in head and tail have to be ordered from left to right.
                # The vector of the first segment of an arbitrary street is turned 
                # perpendicularly to the right and all points at this end of the bundle
                # are projected onto this vector, delivering some kind of a line parameter t.
                # The lines are then sorted by increasing values of this parameter, which
                # orders them from left to right, seen in the direction of the bundle.

                # Process head (start) of the bundle
                arbStreet = head[0]['street']   # arbitrary street at the bundle's head
                # forwardVector points from the head into the bundle.
                fwd = forwardOrder(arbStreet.src,arbStreet.dst)
                srcVec, dstVec = arbStreet.endVectors()
                forwardVector = srcVec if fwd else dstVec
                
                # the origin of this vector
                p0 = arbStreet.src if fwd else arbStreet.dst

                # perp is perpendicular to the forwardVector, turned to the right
                perp = Vector((forwardVector[1],-forwardVector[0])) 

                # sort streets along perp from left to right
                sortedIndices = sorted( (i for i in range(len(head))),  key=lambda i: ( head[i]['firstVert'] - p0).dot(perp) )

                plotStreets(streetGroup, False)    
                for k, indx in enumerate(sortedIndices):
                    head[indx]['i'] = k
                    item = head[indx]
                    p = head[indx]['firstVert']
                    plt.text(p[0],p[1],'  '+str(item['i']),fontsize=12)

                # Process tail (end) of the bundle
                # take an aribtrary street
                arbStreet = tail[0]['street']   # arbitrary street at the bundle's tail

                # forwardVector point from tail out of the bundle
                fwd = forwardOrder(arbStreet.src,arbStreet.dst)
                srcVec, dstVec = arbStreet.endVectors()
                forwardVector = -dstVec if fwd else -srcVec
                
                # the origin of this vector
                p0 = arbStreet.dst if fwd else arbStreet.src

                # perp is perpendicular to forwardVector, turned to the right
                perp = Vector((forwardVector[1],-forwardVector[0])) 

                # sort streats along perp from left to right
                sortedIndices = sorted( (i for i in range(len(head))),  key=lambda i: ( head[i]['lastVert'] - p0).dot(perp) )
                                   
                for k, indx in enumerate(sortedIndices):
                    head[indx]['i'] = k
                    item = head[indx]
                    p = head[indx]['lastVert']
                    plt.text(p[0],p[1],'  '+str(item['i']),fontsize=12)

                plt.title(str(indxG))
                plotEnd()

    def createSymSideLanes(self):
        nr = 0
        toReplace = []
        for location, intersection in self.waymap.iterNodes(Intersection):
            if intersection.order == 2:
                section0 = intersection.leaveWays[0].section
                section1 = intersection.leaveWays[1].section
                streetIds = (intersection.leaveWays[0].street.id, intersection.leaveWays[1].street.id)
                incoming, outgoing = (section0, section1) if section0.dst == section1.src else (section1, section0)
                hasTurns = bool( re.search(r'[^N]', outgoing.lanePatterns[0] ) )
                if hasTurns:    # it's a side lane
                    sideLane = SideLane(location, incoming, outgoing)
                    street = Street(incoming.src, outgoing.dst)
                    street.pred = incoming.pred
                    street.insertEnd(incoming)
                    street.insertEnd(sideLane)
                    street.insertEnd(outgoing)
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
                    street.pred = incoming.pred
                    street.insertEnd(incoming)
                    street.insertEnd(symLane)
                    street.insertEnd(outgoing)
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
            node = self.waymap.getNode(street.src)
            if node:
                predIsect = node['object']
                for way in predIsect.leaveWays:
                    if way.street.id in streetIds:
                        way.street = street
            
            node = self.waymap.getNode(street.dst)
            if node:
                succIsect = node['object']
                for way in succIsect.leaveWays:
                    if way.street.id in streetIds:
                        way.street = street

            self.waymap.removeNode(location)
            street.style = self.styleStore.get( self.getStyle(street) )
            street.setStyleBlockFromTop(street.style)
            self.waymap.addEdge(street)

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
                    self.minorIntersections[intersection.id] = intersection
                else:
                    if intersection.order > 1:
                        self.majorIntersections[intersection.id] = intersection
            else:
                if intersection.order > 1:
                    self.majorIntersections[intersection.id] = intersection

            self.processedNodes.add(location)

    def createStreets(self):
        for street in self.wayManager.iterStreetsFromWaymap():
            self.streets[street.id] = street

 
