from collections import defaultdict
from mathutils import Vector

from way.way_network import WayNetwork, NetSection
from way.way_algorithms import createSectionNetwork
from way.way_section import WaySection
from way.way_intersections import Intersection
from way.output_for_GN import WaySection_gn, IntersectionPoly_gn
from defs.road_polygons import ExcludedWayTags
from lib.SweepIntersectorLib.SweepIntersector import SweepIntersector
from lib.CompGeom.algorithms import SCClipper

class StreetGenerator():

    def __init__(self):
        self.networkGraph = None
        self.sectionNetwork = None
        self.waySections = dict()
        self.intersections = dict()
        self.intersectionAreas = []
        self.waySectionLines = dict()

    def do(self,manager):
        self.useFillet = False
        self.findSelfIntersections()
        self.createWaySectionNetwork()
        self.createWaySections()
        self.createOutputForGN(self.useFillet)

        self.debug = False
        self.plotOutput()

    def findSelfIntersections(self):
        wayManager = self.app.managersById["ways"]

        # some way tags to exclude, used also in createWaySectionNetwork(),
        # ExcludedWayTags is defined in <defs>.
        uniqueSegments = defaultdict(set)
        for way in wayManager.getAllWays():
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
        # get border polygon (ounter-clockwise) of scene frame
        minX, minY = self.app.projection.fromGeographic(self.app.minLat, self.app.minLon)
        maxX, maxY = self.app.projection.fromGeographic(self.app.maxLat, self.app.maxLon)

        # prepare clipper for this frame
        clipper = SCClipper(minX,maxX,minY,maxY)

        wayManager = self.app.managersById["ways"]

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
                section = WaySection(net_section)
                self.waySections[net_section.sectionId] = section

    def createOutputForGN(self, useFillet):
        for node in self.sectionNetwork:
            intersection = Intersection(node, self.sectionNetwork, self.waySections)
            self.intersections[node] = intersection

        # Transitions have to be processed first, because way widths may be altered.
        for node,intersection in self.intersections.items():
            if intersection.order == 2:
                polygon, connectors = intersection.findTransitionPoly()
                if polygon:
                    isectArea = IntersectionPoly_gn()
                    isectArea.polygon = polygon
                    isectArea.connectors = connectors
                    self.intersectionAreas.append(isectArea)

        # Now, the normal intersection areas are constructed
        for node,intersection in self.intersections.items():
            if intersection.order > 2:
                if self.useFillet:
                    polygon, connectors = intersection.intersectionPoly()
                else:
                    polygon, connectors = intersection.intersectionPoly_noFillet()
                if polygon:
                    isectArea = IntersectionPoly_gn()
                    isectArea.polygon = polygon
                    isectArea.connectors = connectors
                    self.intersectionAreas.append(isectArea)

        # Finally, the trimmed centerlines of the was are constructed
        for sectionNr,section in self.waySections.items():
            waySlice = None
            if section.trimT > section.trimS:
                waySlice = section.polyline.trimmed(section.trimS,section.trimT)
                section_gn = WaySection_gn()
                section_gn.centerline = waySlice.verts
                section_gn.category = section.originalSection.category
                section_gn.endWidths = section_gn.startWidths = (section.leftWidth, section.rightWidth)
                section_gn.tags = section.originalSection.tags
                if section.turnParams:
                    section_gn.endWidths = (section.leftWidth+section.turnParams[0], section.rightWidth+section.turnParams[1])
                self.waySectionLines[section.id] = section_gn

    def plotOutput(self):
        for isectArea in self.intersectionAreas:
            plotPolygon(isectArea.polygon,False,'r','r',2,True)
            if self.debug:
                for id, connector in isectArea.connectors.items():
                    p = isectArea.polygon[connector[0]]
                    plt.text(p[0],p[1],str(id)+' '+connector[1])

        for sectionNr,section_gn in self.waySectionLines.items():
            if self.debug:
                plotWay(section_gn.centerline,True,'b',2.)
                center = sum(section_gn.centerline,Vector((0,0)))/len(section_gn.centerline)
                plt.text(center[0],center[1],str(sectionNr),zorder=120)
            else:
                plotWay(section_gn.centerline,False,'b',2.)


import matplotlib.pyplot as plt
def plotPolygon(poly,vertsOrder,lineColor='k',fillColor='k',width=1.,fill=False,alpha = 0.2,order=100):
    x = [n[0] for n in poly] + [poly[0][0]]
    y = [n[1] for n in poly] + [poly[0][1]]
    if fill:
        plt.fill(x[:-1],y[:-1],color=fillColor,alpha=alpha,zorder = order)
    plt.plot(x,y,lineColor,linewidth=width,zorder=order)
    if vertsOrder:
        for i,(xx,yy) in enumerate(zip(x[:-1],y[:-1])):
            plt.text(xx,yy,str(i),fontsize=12)

def plotWay(way,vertsOrder,lineColor='k',width=1.,order=100):
    x = [n[0] for n in way]
    y = [n[1] for n in way]
    plt.plot(x,y,lineColor,linewidth=width,zorder=order)
    if vertsOrder:
        for i,(xx,yy) in enumerate(zip(x,y)):
            plt.text(xx,yy,str(i),fontsize=12)

def plotEnd():
    plt.gca().axis('equal')
    plt.show()