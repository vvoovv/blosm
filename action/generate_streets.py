from collections import defaultdict
from mathutils import Vector

from way.way_network import WayNetwork, NetSection
from way.way_algorithms import createSectionNetwork
from way.way_section import WaySection
from way.way_intersections import Intersection
from defs.road_polygons import ExcludedWayTags
from lib.SweepIntersectorLib.SweepIntersector import SweepIntersector
from lib.CompGeom.algorithms import SCClipper
from lib.CompGeom.BoolPolyOps import boolPolyOp
from lib.CompGeom.GraphBasedAlgos import DisjointSets

class TrimmedWaySection():
    def __init__(self):
        self.centerline = None      # The trimmed centerline in forward direction (first vertex is start.
                                    # and last is end). A Python list of vertices of type mathutils.Vector.
        self.category = None        # The category of the way-section.
        self.nrOfLanes = None       # The left and right number of lanes, seen relative to the direction
                                    # of the centerline. A tuple of integers. For one-way streets, only one
                                    # integer with the number of lanes is in the tuple.
        self.startWidths = None     # The left and right width at its start, seen relative to the direction
                                    # of the centerline. A tuple of floats.
        self.endWidths = None       # The left and right width at its end, seen relative to the direction
                                    # of the centerline. A tuple of floats.
        self.tags = None            # The OSM tags of the way-section.

class IntersectionArea():
    def __init__(self):
        self.polygon = None         # The vertices of the polygon in counter-clockwise order. A Python
                                    # list of vertices of type mathutils.Vector.
        self.connectors = dict()    # The connectors (class Connector_gn) of this polygon to the way-sections.
                                    # A dictionary of tuples, where the key is the ID of the corresponding
                                    # way-section key in the dictionary of TrimmedWaySection. The tuple has two
                                    # elements, <indx> and <end>. <indx> is the first index in the intersection
                                    # polygon for this connector, the second index is <indx>+1. <end> is
                                    # the type 'S' for the start or 'E' for the end of the TrimmedWaySection
                                    # connected here.

class StreetGenerator():
    
    def __init__(self):
        self.networkGraph = None
        self.sectionNetwork = None
        self.waySections = dict()
        self.intersections = dict()

    def do(self, manager):
        self.wayManager = manager
        self.intersectionAreas = manager.intersectionAreas
        self.waySectionLines = manager.waySectionLines
        
        self.useFillet = False
        self.findSelfIntersections()
        self.createWaySectionNetwork()
        self.createWaySections()
        self.createOutput()

    def findSelfIntersections(self):
        # some way tags to exclude, used also in createWaySectionNetwork(),
        # ExcludedWayTags is defined in <defs>.
        uniqueSegments = defaultdict(set)
        for way in self.wayManager.getAllWays():
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
        # get border polygon (ounter-clockwise) of scene frame
        minX, minY = self.app.projection.fromGeographic(self.app.minLat, self.app.minLon)
        maxX, maxY = self.app.projection.fromGeographic(self.app.maxLat, self.app.maxLon)

        # prepare clipper for this frame
        clipper = SCClipper(minX,maxX,minY,maxY)

        # Not really used. This is a relict from way_clustering.py
        wayManager.junctions = (
            [],#mainJunctions,
            []#smallJunctions
        )

        # create full way network
        wayManager.networkGraph = self.networkGraph = WayNetwork()

        # some way tags to exclude, used also in createWaySectionNetwork(),
        # ExcludedWayTags is defined in <defs>.
        for way in wayManager.getAllIntersectionWays():
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

    def createOutput(self):
        # Find first nodes that will produce conflicting intersections
        conflictingNodes = DisjointSets()
        for node in self.sectionNetwork:
            intersection = Intersection(node, self.sectionNetwork, self.waySections)
            conflictNodes = intersection.checkForConflicts()
            if conflictNodes:
               for conflict in conflictNodes:
                    conflictingNodes.addSegment(node,conflict)
                    # import matplotlib.pyplot as plt
                    # plt.plot(conflict[0],conflict[1],'co',markersize=10,zorder=999)
                    # plt.plot(node[0],node[1],'mx',markersize=15,zorder=999)

        # Create intersections from nodes, that did not produce conflicts
        for node in self.sectionNetwork:
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
                        # import matplotlib.pyplot as plt
                        # plt.plot(node[0],node[1],'co',markersize=5,zorder=999)
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
            waySlice = None
            if section.trimT > section.trimS:
                waySlice = section.polyline.trimmed(section.trimS,section.trimT)
                section_gn = TrimmedWaySection()
                section_gn.centerline = waySlice.verts
                section_gn.category = section.originalSection.category
                if section.isOneWay:
                    section_gn.nrOfLanes = (section.nrRightLanes)
                else:
                    section_gn.nrOfLanes = (section.nrLeftLanes, section.nrRightLanes)
                section_gn.endWidths = section_gn.startWidths = (section.leftWidth, section.rightWidth)
                section_gn.tags = section.originalSection.tags
                if section.turnParams:
                    section_gn.endWidths = (section.leftWidth+section.turnParams[0], section.rightWidth+section.turnParams[1])
                self.waySectionLines[section.id] = section_gn
            # else:
            #     # already treated as reason for conflicting areas
            #     plotWay(section.polyline.verts,False,'c',2,999)
