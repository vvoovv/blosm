from math import sin, cos, atan2, pi, sqrt
from operator import xor
from mathutils import Vector
from itertools import tee,islice, cycle
from lib.CompGeom.PolyLine import PolyLine
from lib.CompGeom.offset_intersection import offsetPolylineIntersection
from lib.CompGeom.LinePolygonClipper import LinePolygonClipper
from defs.way_cluster_params import transitionSlope


# from osmPlot import *

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

def spline_4p( t, v0, p0, p1, v1 ):
    # (Catmull-Rom) Cubic curve goes from p0 to p1, and outer points v0 and v1
    # determine the slopes at p0 and p1.
    # assert 0 <= t <= 1
    return (
          t*((2-t)*t - 1)     * v0
        + (t*t*(3*t - 5) + 2) * p0
        + t*((4 - 3*t)*t + 1) * p1
        + (t-1)*t*t           * v1 ) / 2

def pointInPolygon(poly,p):
    inside = False
    for p1,p2 in cyclePair(poly):
        if p[1] > min(p1[1],p2[1]):
            if p[1] <= max(p1[1],p2[1]):
                if p[0] <= max(p1[0],p2[0]):
                    if p1[1] != p2[1]:
                        xints = (p[1] - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
                    if p1[0] == p2[0] or p[0] <= xints:
                        inside = not inside
    return inside

# ----------------------------------------------------------------

class OutgoingWay():
    # This class holds a way-section that leaves an intersection. The direction
    # of its polyline, the sides 'left' and 'right', and the start and target
    # trim positions are redirected correctly to the original instance of WaySection.
    def __init__(self,section,fwd):
        # section:  Instance of a WaySection, that holds the way-section in its
        #           original direction.
        # fwd:      Direction of the outgoing section. True, if same as original.
        self.section = section
        self.fwd = fwd
        self.offset = self.section.offset if fwd else -self.section.offset
        self.polyline = section.polyline.clone()
        self.polyline.setView( PolyLine.fwd if fwd else PolyLine.rev)
        self.polygon = self.polyline.buffer(abs(self.leftW),abs(self.rightW))
        self.isLoop = self.section.isLoop

    @property
    def leftW(self):
        if self.offset != 0.:
            return self.section.width/2 - self.offset
        else:
            return self.section.width/2

    @property
    def rightW(self):
        if self.offset != 0.:
            return -self.section.width/2 - self.offset
        else:
            return -self.section.width/2

    def setTrim(self,trim):
        if self.fwd:
            self.section.trimS = trim
        else:
            self.section.trimT = len(self.polyline)-1 - trim

class Intersection():
    def __init__(self, *args):
        if isinstance(args[0],Vector):
            self.initFromNode(args[0], args[1], args[2])
        else:
            self.initFromConflictingNodes(args[0], args[1], args[2])

    def initFromNode(self, position, network, waySections):
        self.position = position
        self.outWays = []
        self.order = 0
        self.network = network

        for net_section in network.iterOutSegments(self.position):
            if net_section.category != 'scene_border':
                if waySections[net_section.sectionId].isValid:
                    self.addSection( position, waySections[net_section.sectionId] )
        self.sortSections(self.position)

    def initFromConflictingNodes(self, positions, network, waySections):
        self.position = sum( (v for v in positions),Vector((0.,0.)) )/len(positions)
        self.outWays = []
        self.order = 0
        self.network = network

        for node in positions:
            for net_section in network.iterOutSegments(node):
                if net_section.category != 'scene_border':
                    if net_section.s not in positions or net_section.t not in positions:
                        waySection = waySections[net_section.sectionId]
                        if waySection.isValid:
                            if waySection.isLoop:
                                self.outWays.append( OutgoingWay(waySection,True))
                                self.outWays.append( OutgoingWay(waySection,False))
                                self.order += 2
                            else:
                                if node == waySection.originalSection.s:
                                    self.outWays.append( OutgoingWay(waySection,True))
                                else:
                                    self.outWays.append( OutgoingWay(waySection,False))
                                self.order += 1

            def sort_key(outway):
                vec = outway.polyline[-1] - self.position
                length = vec.length
                angle = atan2(vec.y,vec.x)
                if angle < 0:
                    return 2*pi+angle, length
                else:
                    return angle, length
            self.outWays= sorted(self.outWays, key = sort_key)

    def addSection(self, position, waySection):
        # The polyline of the outgoing way of the intersection in <self.outWays> starts
        # at the intersection <position>. The instance of the class OutgoingWay controls the
        # relation to the original way-section given in <waySection>. <self.order> is the
        # number of outgoing ways and determines the type of the intersection. Loop ways have
        # their two ends at the intersection position and are marked in <waySection> with 
        # waySection.isLoop == True.
        if waySection.isLoop:
            self.outWays.append( OutgoingWay(waySection,True))
            self.outWays.append( OutgoingWay(waySection,False))
            self.order += 2
        else:
            if position == waySection.originalSection.s:
                self.outWays.append( OutgoingWay(waySection,True))
            else:
                self.outWays.append( OutgoingWay(waySection,False))
            self.order += 1

    def sortSections(self,position):
    # Sort the outgoing ways in <self.outWays> first by their angle in counter-clockwise order
    # around their start positions and then by the distance of their
    # end-point from this position.
        def sort_key(outway):
            vec = outway.polyline[1] - position
            length = vec.length
            angle = atan2(vec.y,vec.x)
            if angle < 0:
                return 2*pi+angle, length
            else:
                return angle, length
        self.outWays = sorted(self.outWays, key = sort_key)

    def cleanShortWays(self,debug=False):
        # A way is defined as short way, when at least one of its left or right ends of its 
        # border is within the area of its neighbor way. Such ways are removed from the
        # intersection, but returned for eventual further use.
        shortWays = []

        # If outways of intersection form loops, don' clean anything
        if any(way.isLoop for way in self.outWays):
            return []

        if self.order < 3:
            return []

        for left,centre,right in cycleTriples(self.outWays):
            fragments = []
            endR = centre.polyline.offsetPointAt(len(centre.polyline)-1.,centre.rightW)
            endL = centre.polyline.offsetPointAt(len(centre.polyline)-1.,centre.leftW)
            if pointInPolygon(right.polygon, endR):
                shortWays.append(centre)
            elif pointInPolygon(right.polygon, endL):
                clipper = LinePolygonClipper(right.polygon)
                border = centre.polyline.parallelOffset(centre.leftW)
                fragments, _, _  = clipper.clipLine(border)
                if len(fragments) <= 1:
                    shortWays.append(centre)
            if pointInPolygon(left.polygon, endR):
                clipper = LinePolygonClipper(left.polygon)
                border = centre.polyline.parallelOffset(centre.rightW)
                fragments, _, _  = clipper.clipLine(border)
                if len(fragments) <= 1:
                    shortWays.append(centre)
            elif pointInPolygon(left.polygon, endL):
                shortWays.append(centre)

            # if debug and len(fragments)>1:
            #     for way in self.outWays:
            #         plotWay(way.polyline,way.leftW,way.rightW,'k',1)
            #     plotNode(endR, 'r', 30)
            #     plotNode(endL, 'b', 30)
            #     plotPolygon(right.polygon,False,'r','r',2,True)
            #     plotPolygon(left.polygon,False,'b','b',2,True)
            #     plotEnd()

        # Remove short ways from Intersection
        for way in shortWays:
            if way in self.outWays:
                self.outWays.remove(way)
                self.order -= 1

        return shortWays

    def intersectionPoly(self,debug=False):

        wayConnectors = dict()
        area = []
        for rightWay,centerWay,leftWay in cycleTriples(self.outWays):
            # if debug:
            #     for i,way in enumerate(self.outWays):
            #         way.polyline.plot('k')
            #         q = way.polyline[-1]
            #         plt.text(q[0],q[1]," "+str(i))
            #     plotWay(rightWay.polyline,rightWay.leftW,-rightWay.rightW,'b',1)
            #     plotWay(centerWay.polyline,centerWay.leftW,-centerWay.rightW,'b',1)
            #     plotWay(leftWay.polyline,leftWay.leftW,-leftWay.rightW,'b',1)
            #     # plotPureNetwork(self.network,False)
            #     rightWay.polyline.plot('r',2)
            #     centerWay.polyline.plot('b:',2)
            #     leftWay.polyline.plot('g',2)

                # if abs(centerWay.leftW) != abs(centerWay.rightW): 
                #     print(centerWay.leftW,centerWay.rightW,centerWay.leftW+centerWay.rightW)
                #     color= 'ro'
                # else:
                #     color = 'go'
                # plt.plot(centerWay.polyline[-1][0],centerWay.polyline[-1][1],color,markersize=10)
 
            # Intersection at the right side of center-way
            # outIsects = 0
            p1, type = offsetPolylineIntersection(rightWay.polyline,centerWay.polyline,rightWay.leftW,-centerWay.rightW,True,0.1)
            if type == 'valid':
                _,tP1 = centerWay.polyline.orthoProj(p1)
                if tP1 < 0.:
                    tP1 = 0.
                    p1 = centerWay.polyline.offsetPointAt(tP1,centerWay.rightW)
                elif isinstance(tP1,int):   # inter buffer point, reproject
                    p1 = centerWay.polyline.offsetPointAt(tP1,centerWay.rightW)
            elif type == 'parallel':
                transWidth = min(centerWay.polyline.length()*0.5,max(1.,abs(rightWay.leftW+centerWay.rightW)/transitionSlope) )
                tP1 = centerWay.polyline.d2t(transWidth)
                p1 = centerWay.polyline.offsetPointAt(tP1,centerWay.rightW)
            else: # out
                # rightWay.polyline.plot('r',4)
                # centerWay.polyline.plot('b:',2)
                # leftWay.polyline.plot('g',2)
                # # plotEnd()
                # outIsect += 1
                print('out')
                continue
            # if debug:
            #     plt.plot(p1[0],p1[1],'kx')
            #     plt.text(p1[0],p1[1],'   p1')

            # Intersection at the left side of center-way
            p3, type = offsetPolylineIntersection(centerWay.polyline,leftWay.polyline,centerWay.leftW,-leftWay.rightW,True,0.1)
            # if debug:
            #     plt.plot(p3[0],p3[1],'rx',markersize=4)
            #     plt.text(p3[0],p3[1],'   p3_first')
            if type == 'valid':
                _,tP3 = centerWay.polyline.orthoProj(p3)
                if tP3 < 0.:
                    tP3 = 0.
                    p3 = centerWay.polyline.offsetPointAt(tP3,centerWay.leftW)
                elif isinstance(tP3,int):   # inter buffer point, reproject
                    p3 = centerWay.polyline.offsetPointAt(tP3,centerWay.leftW)
            elif type == 'parallel':
                # transWidth =max(1.,abs(centerWay.leftW+leftWay.rightW)/transitionSlope)
                transWidth = min(centerWay.polyline.length()*0.5,max(1.,abs(centerWay.leftW+leftWay.rightW)/transitionSlope) )
                tP3 = centerWay.polyline.d2t(transWidth)
                p3 = centerWay.polyline.offsetPointAt(tP3,centerWay.leftW)
            else: # out
                # rightWay.polyline.plot('r',4)
                # centerWay.polyline.plot('b:',2)
                # leftWay.polyline.plot('g',2)
                # # plotEnd()
                # outIsect += 1
                print('out')
                continue
            # if debug:
            #     plt.plot(p3[0],p3[1],'kx')
            #     plt.text(p3[0],p3[1],'   p3')

            # Project these onto the centerline of the out-way and create intermediate
            # polygon point <p2>.
            # _,tP1 = centerWay.polyline.orthoProj(p1)
            # _,tP3 = centerWay.polyline.orthoProj(p3)
            Id = centerWay.section.id  if centerWay.fwd else -centerWay.section.id
            t0 = 0.
            if tP3 > tP1:
                p2 = centerWay.polyline.offsetPointAt(tP3,centerWay.rightW)
                t0 = tP3
                wayConnectors[Id] = len(area)+1
            else:
                p2 = centerWay.polyline.offsetPointAt(tP1,centerWay.leftW)
                t0 = tP1
                wayConnectors[Id] = len(area)
            # if debug:
            #     plt.plot(p2[0],p2[1],'kx')
            #     plt.text(p2[0],p2[1],'   p2')

            if centerWay.fwd:
                centerWay.section.trimS = max(centerWay.section.trimS, t0)
            else:
                t = len(centerWay.section.polyline)-1 - t0
                centerWay.section.trimT = min(centerWay.section.trimT, t)            
            # else:   # no intersections on both sides, set points to start of the way.
            #     if centerWay.fwd:
            #         p1 = p2 = centerWay.polyline.offsetPointAt(0.,centerWay.leftW)
            #         p3 = centerWay.polyline.offsetPointAt(0.,centerWay.rightW)
            #     else:
            #         t = len(centerWay.section.polyline)-1
            #         p1 = p2 = centerWay.polyline.offsetPointAt(t,centerWay.leftW)
            #         p3 = centerWay.polyline.offsetPointAt(t,centerWay.rightW)

            if area and (p1-area[-1]).length < 0.01:
                area = area[:-1]
                wayConnectors[Id] -= 1
            if p1==p2 or p2==p3: # way is perpendicular
                area.extend([p1,p3])
            else:
                area.extend([p1,p2,p3])
            # if debug:
            #     plotLine([p1,p2,p3] if p1!=p2 and p2!=p3 else [p1,p3],True,'m')
            #     plotPolygon([p1,p2,p3],True,'r')
            #     plotEnd()
        area = area[:-1] if (area[0]-area[-1]).length < 0.001 else area
        # if debug:
        #     plotPolygon(area,True,'r')
        #     for i,c in enumerate(wayConnectors.values()):
        #         p = area[c]
        #         plt.text(p[0],p[1],str(i),color = 'r',fontsize = 14)
        #     for way in self.outWays:
        #         way.polyline.plot('k')
        #     plotEnd()
        return area, wayConnectors



    # def intersectionPoly_noFillet(self):
    #     tmax = [0.]*self.order  # List of maximum line parameters of the projections of the intersection
    #                             # points onto the center-lines of the ways.
    #     isectPoints = [None]*self.order

    #     for (indx1, indx2), (way1,way2) in zip(cyclePair(range(self.order)),cyclePair(self.outWays)):
    #         # <way1> and <way2> are consecutive outgoing polylines in counter-clockwise order of the
    #         # intersection. These are the centerlines of the way-sections, <way1> is to the right
    #         # (clockwise) of <way2>.

    #         # Compute the left border <borderL> of the way-section given by way1 and the 
    #         # right border <borderR> of the way-section given by way2.
    #         # For way-loops, the forward and backward polyline is in expected consecutive order
    #         # in <self.outWayLines>. For way-loops, a parallel offset can't be computed
    #         # correctly, therefore, only their first segments are used.
    #         if way1.section.isLoop and way2.section.isLoop:
    #             trimmed1 = PolyLine(way1.polyline[:2])
    #             trimmed2 = PolyLine(way2.polyline[:2])
    #             borderL = trimmed1.parallelOffset(way1.leftW)
    #             borderR = trimmed2.parallelOffset(way2.rightW)
    #         else:
    #             borderL = way1.polyline.parallelOffset(way1.leftW)
    #             borderR = way2.polyline.parallelOffset(way2.rightW)

    #         # Find the intersection between the left border <borderL> of <way1> 
    #         # and the right border <borderR> of <way2>.
    #         # In <isectParams>, <iP> is the intersection point, <tL> and <tR> are the
    #         # line parameters of the intersection point (see PolyLine.py) for the 
    #         # left and the right border. <uVL> and <uVR> are the unit vectors of the intersecting
    #         # segments.
    #         isectParams = PolyLine.intersection(borderL,borderR)
    #         if isectParams is None:
    #             # No intersection found
    #             continue
    #         iP,_,_,_,_ = isectParams

    #         # The line parameters of the intersection may be different from the line parameter of
    #         # the center-line. To get consistent and equal vertices for the way-polygon and the
    #         # intersection area, the projection of the intersection onto the center-line is used.
    #         _,tPL = way1.polyline.orthoProj(iP)
    #         _,tPR = way2.polyline.orthoProj(iP)

    #         # Find the maximal line parameter for every way
    #         tmax[indx1] = max(tmax[indx1],tPL)
    #         tmax[indx2] = max(tmax[indx2],tPR)
    #         isectPoints[indx1] = way1.polyline.offsetPointAt(tPL,way1.leftW)

    #     polygon = []
    #     connectors = dict()
    #     # Construct the intersection area polygon
    #     for (indx1, indx2), (way1,way2) in zip(cyclePair(range(self.order)),cyclePair(self.outWays)):
    #         # <way1> and <way2> are consecutive outgoing polylines in counter-clockwise order of the
    #         # intersection. These are the centerlines of the way-sections, <way1> is to the right
    #         # (clockwise) of <way2>.

    #         isectP =  isectPoints[indx1]
    #         t1 = tmax[indx1]  # Line parameters of trim points.
    #         t2 = tmax[indx2]

    #         # The first way section will have to be trimmed to the length not occupied by
    #         # the polygon of the intersection area.
    #         way1.setTrim(t1)

    #         # Starting with the left border of the centerline <way1>.
    #         # Given the line parameter of the trim point on the centerline, we can 
    #         # compute the point <pL> perpendicularly offset to the left.
    #         pL = way1.polyline.offsetPointAt(t1,way1.leftW)
    #         if isectP:
    #             # If we had an intersection between the borders, by a simple check of
    #             # the coordinate difference to the trim point we may decide if
    #             # <pL> is the same as this intersection.
    #             dL = sum(pL-isectP)
    #             if abs(dL) > 1.e-4:
    #                 # If not on the intersection, add to polygon
    #                 polygon.append(pL)
    #                 # then extend polygon by the intersection.
    #             polygon.append(isectP)
    #         else:
    #             # Use only the trim point.
    #             polygon.append(pL)

    #         # The same procedure is then done for the right border of the centerline <way2>
    #         pR = way2.polyline.offsetPointAt(t2,way2.rightW)
    #         if isectP:
    #             dR = sum(pR-isectP)
    #             if abs(dR) > 1.e-4:
    #                 polygon.append(pR)
    #         else:
    #             polygon.append(pR)
    #         Id = way2.section.id if way2.fwd else -way2.section.id
    #         connectors[Id] = len(polygon)-2

    #     polygon = polygon[1:] + polygon[:1]
    #     return polygon, connectors

    # def intersectionPoly(self):
    #     tmax = [0.]*self.order  # List of maximum line parameters of the projections of the intersection
    #                             # points onto the center-lines of the ways.
    #     isectPoints = [None]*self.order
    #     filletVertsList = [None]*self.order

    #     for (indx1, indx2), (way1,way2) in zip(cyclePair(range(self.order)),cyclePair(self.outWays)):
    #         # <way1> and <way2> are consecutive outgoing polylines in counter-clockwise order of the
    #         # intersection. These are the centerlines of the way-sections, <way1> is to the right
    #         # (clockwise) of <way2>.

    #         # Compute the left border <borderL> of the way-section given by way1 and the 
    #         # right border <borderR> of the way-section given by way2.
    #         # For way-loops, the forward and backward polyline is in expected consecutive order
    #         # in <self.outWayLines>. For way-loops, a parallel offset can't be computed
    #         # correctly, therefore, only their first segments are used.
    #         if way1.section.isLoop and way2.section.isLoop:
    #             trimmed1 = PolyLine(way1.polyline[:2])
    #             trimmed2 = PolyLine(way2.polyline[:2])
    #             borderL = trimmed1.parallelOffset(way1.leftW)
    #             borderR = trimmed2.parallelOffset(way2.rightW)
    #         else:
    #             borderL = way1.polyline.parallelOffset(way1.leftW)
    #             borderR = way2.polyline.parallelOffset(way2.rightW)

    #         # Find the intersection between the left border <borderL> of <way1> 
    #         # and the right border <borderR> of <way2>.
    #         # In <isectParams>, <iP> is the intersection point, <tL> and <tR> are the
    #         # line parameters of the intersection point (see PolyLine.py) for the 
    #         # left and the right border. <uVL> and <uVR> are the unit vectors of the intersecting
    #         # segments.
    #         isectParams = PolyLine.intersection(borderL,borderR)
    #         if isectParams is None:
    #             # No intersection found
    #             continue
    #         iP,tL,tR,uVL,uVR = isectParams

    #         # Now, the fillet of this intersection gets constructed.
    #         # Get the fillet radius between these ways based on the ways' categories.
    #         # The smaller of them is used.
    #         radius1 = estFilletRadius(way1.section.originalSection.category,way1.section.originalSection.tags)
    #         radius2 = estFilletRadius(way2.section.originalSection.category,way2.section.originalSection.tags)
    #         radius = min(radius1,radius2)

    #         # For small angles, reduce the fillet radius continuously
    #         cosAngle = uVL.dot(uVR)
    #         if cosAngle > 0.3:  # ~28°
    #             radius *= cosAngle

    #         # <lengthL> and <lengthR> are the lengths on the intersected segments between
    #         # their start point and the intersection
    #         lengthL = borderL.lengthOfSegment(floor(tL))
    #         lengthR = borderR.lengthOfSegment(floor(tR))

    #         # Starting with the radius found above, a fillet is created within the intersected
    #         # segments. Its legs must not be longer than the way-segments. If required,
    #         # the raidus is decreased iteratively in the follwoing loop.
    #         while True:
    #             # Try to find an arc with radius <radius> and using the segments starting at <iP>
    #             # and suing the unit vectors <uVL> and <uVR> as tangents. In the result, <origin>
    #             # is the circle origin of the arc, <pLegEndL> and <pLegEndR> are the tangent vertices
    #             # and <legLength> is the length of the legs between the intersection point and the
    #             # tangent vertices.
    #             origin, pLegEndL, pLegEndR, legLength = filletArc(iP, uVL, uVR, radius)
    #             if origin is None: # No arc possible, when segments are almost parallel.
    #                 break

    #             # To check whether the leg ends are on the intersecting segments, the line
    #             # parameters of the tangent vertices are computed (see PolyLine.py). For a 
    #             # valid fillet, they must be both less than 1.
    #             tLegL = tL%1. + legLength / lengthL
    #             tLegR = tR%1. + legLength / lengthR
    #             if tLegL < 1. and tLegR < 1.:
    #                 # OK, fillet fits between these segments using this radius
    #                 break
    #             # if not, reduce radius
    #             radius = 0.9 * radius

    #         # If a fillet arc with legs that fit onto the segments has been found.
    #         if origin:
    #             # Create the vertices for the fillet.
    #             filletVerts = filletLine(origin, pLegEndL, pLegEndR, radius)

    #             # The line parameters of the intersection may be different from the line parameter of
    #             # the center-line. To get consistent and equal vertices for the way-polygon and the
    #             # intersection area, the projection of the intersection onto the center-line is used.
    #             _,tPL = way1.polyline.orthoProj(pLegEndL)
    #             _,tPR = way2.polyline.orthoProj(pLegEndR)

    #             # Find the maximal line parameter for every way
    #             tmax[indx1] = max(tmax[indx1],tPL)
    #             tmax[indx2] = max(tmax[indx2],tPR)
    #             isectPoints[indx1] = way1.polyline.offsetPointAt(tPL,way1.leftW)

    #             # Store the fillet vertices for <way1>.
    #             filletVertsList[indx1] = filletVerts

    #         else:
    #             # No possible fillet found
    #             print('No possible fillet found.')

    #     polygon = []
    #     connectors = dict()
    #     # Construct the intersection area polygon
    #     for (indx1, indx2), (way1,way2) in zip(cyclePair(range(self.order)),cyclePair(self.outWays)):
    #         # <way1> and <way2> are consecutive outgoing polylines in counter-clockwise order of the
    #         # intersection. These are the centerlines of the way-sections, <way1> is to the right
    #         # (clockwise) of <way2>.

    #         filletVerts = filletVertsList[indx1]
    #         isectP =  isectPoints[indx1]
    #         t1 = tmax[indx1]  # Line parameters of trim points.
    #         t2 = tmax[indx2]

    #         # The first way section will have to be trimmed to the length not occupied by
    #         # the polygon of the intersection area.
    #         way1.setTrim(t1)

    #         # Starting with the left border of the centerline <way1>.
    #         # Given the line parameter of the trim point on the centerline, we can 
    #         # compute the point <pL> perpendicularly offset to the left.
    #         pL = way1.polyline.offsetPointAt(t1,way1.leftW)
    #         if filletVerts:
    #             # If we had an fillet between the borders, by a simple check of
    #             # the coordinate difference of the first vertex of the fillet 
    #             # to the trim point we may decide if <pL> is the same as this vertex.
    #             dL = sum(pL-filletVerts[0])
    #             if abs(dL) > 1.e-4:
    #                 # If not on the fillet, add to polygon
    #                 polygon.append(pL)
    #             # then extend polygon by fillet verts.
    #             polygon.extend(filletVerts)
    #         else:
    #             # Use only the trim point.
    #             polygon.append(pL)

    #         # The same procedure is then done for the right border of the centerline <way2>
    #         pR = way2.polyline.offsetPointAt(t2,way2.rightW)
    #         if filletVerts:
    #             dR = sum(pR-filletVerts[-1])
    #             if abs(dR) > 1.e-4:
    #                 polygon.append(pR)
    #         else:
    #             polygon.append(pR)
    #         Id = way2.section.id if way2.fwd else -way2.section.id
    #         connectors[Id] = len(polygon)-2

    #     polygon = polygon[1:] + polygon[:1]
    #     return polygon, connectors

    # def findTransitionPoly(self):
    #     # Transitions are intersections of order==2 and have an incoming and
    #     # an outgoing way. They have to be processed before the other intersections,
    #     # because way widths may be altered due to turning lanes.
    #     way1, way2 = self.outWays

    #     # Check angle between these ways. If it is larger than about 20°, just create common
    #     # intersection area. The vectors of the first segments are used.
    #     vec1 = way1.section.sV if way1.fwd else way1.section.tV
    #     vec1 /= vec1.length # make unit vector
    #     vec2 = way2.section.sV if way2.fwd else way2.section.tV
    #     vec2 /= vec2.length # make unit vector

    #     if vec1.dot(vec2) > 0.5:
    #         polygon = None
    #         connectors = None
    #         try:
    #             polygon, connectors = self.intersectionPoly_noFillet()
    #         except:
    #             print('Problem in findTransitionPoly)')
    #         return polygon, connectors

    #     outWay, inWay = (way1, way2) if way1.fwd else (way2, way1)
    #     inTags, outTags = inWay.section.originalSection.tags, outWay.section.originalSection.tags

    #     # Do we have turn lanes? They are only possible in the outLine.
    #     # ******* turn lanes curently switche off
    #     if False:# 'turn:lanes' in outTags:
    #         # There is no transition polygon required. The outgoing way section
    #         # becomes eventually a turning lane.
    #         laneDescs = outTags['turn:lanes'].split('|')
    #         leftTurns = ['left','slight_left','sharp_left']
    #         rightTurns = ['right','slight_right','sharp_right']
    #         leftTurnLanes = sum(1 for tag in laneDescs if any(x in tag for x in leftTurns) )
    #         rightTurnLanes = sum(1 for tag in laneDescs if any(x in tag for x in rightTurns) )
    #         if leftTurnLanes or rightTurnLanes:
    #             leftWidthDifference = outWay.section.leftWidth - inWay.section.leftWidth
    #             rightWidthDifference = outWay.section.rightWidth - inWay.section.rightWidth
    #             if leftWidthDifference or rightWidthDifference:
    #                 outWay.section.turnParams = [leftWidthDifference,rightWidthDifference]

    #     # Prepare transition polygon 
    #     if outWay.section.turnParams:
    #         transitionWidth = 0.5
    #     else:
    #         inWidth = inWay.section.leftWidth + inWay.section.leftWidth
    #         outWidth = outWay.section.leftWidth + outWay.section.leftWidth
    #         widthDiff = abs(inWidth-outWidth)
    #         transitionWidth = min(10.,max(1.,2 * widthDiff))
    #     inT = inWay.polyline.d2t(transitionWidth)
    #     outT = outWay.polyline.d2t(transitionWidth)
    #     if inT is None or outT is None:
    #         pass
    #     else:
    #         assert inT is not None
    #         assert outT is not None

    #         # compute cornerpoints
    #         inLeft = inWay.polyline.offsetPointAt(inT,inWay.section.rightWidth)
    #         inRight = inWay.polyline.offsetPointAt(inT,-inWay.section.leftWidth)
    #         outLeftWidth = inWay.section.leftWidth if outWay.section.turnParams else outWay.section.leftWidth
    #         outRightWidth = inWay.section.rightWidth if outWay.section.turnParams else outWay.section.rightWidth
    #         outLeft = outWay.polyline.offsetPointAt(outT,outLeftWidth)
    #         outRight = outWay.polyline.offsetPointAt(outT,-outRightWidth)

    #         poly = []
    #         import numpy as np
    #         nrOfSplineVerts = 5
    #         t0 = np.linspace(0,1,nrOfSplineVerts)
    #         p0, p1 = inRight, outLeft
    #         v0, v1 = p0 - inWay.polyline.unitEndVec(False)*transitionWidth, p1 - outWay.polyline.unitEndVec(False)*transitionWidth
    #         for t in t0:
    #             sp = spline_4p( t, v0, p0, p1, v1 )
    #             poly.append( sp )
    #         p0, p1 = outRight, inLeft
    #         v0, v1 = p0 - outWay.polyline.unitEndVec(False)*transitionWidth, p1 - inWay.polyline.unitEndVec(False)*transitionWidth
    #         for t in t0:
    #             sp = spline_4p( t, v0, p0, p1, v1 )
    #             poly.append( sp )


    #         if inWay.fwd:
    #             inWay.section.trimS = inT
    #         else:
    #             inWay.section.trimT = len(inWay.polyline)-1 - inT
    #         if outWay.fwd:
    #             outWay.section.trimS = outT
    #         else:
    #             outWay.section.trimT = len(outWay.polyline)-1 - outT

    #         poly = [Vector(v) for v in map(tuple, poly)]
    #         poly = poly[1:] + poly[:1]
    #         area = sum( (p2[0]-p1[0])*(p2[1]+p1[1]) for p1,p2 in zip(poly,poly[1:]+[poly[0]]))
    #         if area > 0:
    #             poly.reverse()
    #         connectors = dict()
    #         Id = inWay.section.id if inWay.fwd else -inWay.section.id
    #         connectors[Id] = 0
    #         Id = outWay.section.id if outWay.fwd else -outWay.section.id
    #         connectors[Id] = nrOfSplineVerts
    #         return poly, connectors


def filletArc(p, uv1, uv2, radius):
    # p: corner point (class Vector)
    # uv1, uv2: unit vectors from p to legs direction
    # cos(a) for angle a between uv1 and uv2
    cos_a = uv1.x*uv2.x + uv1.y*uv2.y # dot product
    if abs(cos_a) >= 1.-1.e-4:
        return None, None, None, None
    # tan(a/2) = sqrt((1 - cos(a)) / (1 + cos(a))
    tan_a2 = sqrt( (1 - cos_a) / (1 + cos_a) )
    # length of legs to tangent points
    length = radius / tan_a2
    # tangent points
    tp1 = p + uv1 * length
    tp2 = p + uv2 * length
    # origin of circle
    o = tp2 + Vector((uv2.y,-uv2.x)) * radius
    # plt.plot(tp1.x,tp1.y,'b.')
    # plt.plot(tp2.x,tp2.y,'r.')
    # plt.plot(o.x,o.y,'g.')

    return  o, tp1, tp2, length

def filletLine(o, tp1, tp2, radius):
    vertList = [tp2]
    a2 = atan2(tp1.y-o.y, tp1.x-o.x)
    a1 = atan2(tp2.y-o.y, tp2.x-o.x)

    if a1 > a2:
        a2 += 2*pi
    atot = a2 - a1

    QUADRANT_SEGMENTS = 8
    filletAngleQuantum = pi / 2.0 / QUADRANT_SEGMENTS
    nSegs = int(atot / filletAngleQuantum + 0.5)

    # no segments because angle is less than increment-nothing to do!
    if nSegs < 1:
        return []

    # choose angle increment so that each segment has equal length
    ainc = atot / nSegs

    acurr = ainc
    # pt = Coordinate()
    while acurr < atot:
        a = a1 + acurr
        pt = o + Vector((radius * cos(a),radius * sin(a)))
        vertList.append(pt)
        acurr += ainc

    if (tp1-vertList[-1]).length > 0.5:
        vertList.append(tp1)
    vertList.reverse()
    return vertList


class IntersectionCluster():
    def __init__(self, positions, clusterPoly, network, waySections):
        self.positions = positions
        self.clusterPoly = clusterPoly[::-1]
        self.outWays = []
        self.order = 0

        acceptedPositions = []
        for position in self.positions:
            for net_section in network.iterOutSegments(position):
                if net_section.category != 'scene_border':
                    if waySections[net_section.sectionId].isValid:
                        section = waySections[net_section.sectionId]
                        # One endpoint of the polyline must be inside the cluster
                        # polygon and one outside.
                        inside1 = pointInPolygon(self.clusterPoly,section.polyline[0]) 
                        inside2 = pointInPolygon(self.clusterPoly,section.polyline[-1]) 
                        if xor(inside1,inside2):
                            if net_section.s not in acceptedPositions and net_section.t not in acceptedPositions:
                                self.addSection( position, waySections[net_section.sectionId] )
                                acceptedPositions.append(position)
                            else:
                                waySections[net_section.sectionId].isValid = False
        self.sortSections()

    def addSection(self, position, waySection):
        # The polyline of the outgoing way of the intersection in <self.outWays> starts
        # at the intersection <position>. The instance of the class OutgoingWay controls the
        # relation to the original way-section given in <waySection>. <self.order> is the
        # number of outgoing ways and determines the type of the intersection. Loop ways have
        # their two ends at the intersection position and are marked in <waySection> with 
        # waySection.isLoop == True.
        if waySection.isLoop:
            self.outWays.append( OutgoingWay(waySection,True))
            self.outWays.append( OutgoingWay(waySection,False))
            self.order += 2
        else:
            if position == waySection.originalSection.s:
                self.outWays.append( OutgoingWay(waySection,True))
            else:
                self.outWays.append( OutgoingWay(waySection,False))
            self.order += 1

    def sortSections(self):
    # Sort the outgoing ways in <self.outWays> first by their angle in counter-clockwise order
    # around the cener of gravity of their start positions and then by the distance of their
    # end-point from this center.
        center = sum( (position for position in self.positions),Vector((0,0)))/len(self.positions)
        def sort_key(outway):
            vec = outway.polyline[1] - center
            length = vec.length
            angle = atan2(vec.y,vec.x)
            if angle < 0:
                return 2*pi+angle, -length
            else:
                return angle, -length
        self.outWays = sorted(self.outWays, key = sort_key)
        # for i,outway in enumerate(self.outWays):
        #     p = outway.polyline[-1]
        #     plt.text(p[0],p[1],str(i),fontsize=12)

    def intersectionPoly_noFillet(self):

        def intersectLineWithClusterPoly(polyline):
            for i, p in zip(range(len(self.clusterPoly)),cyclePair(self.clusterPoly)):
                v = polyline.intersectSegment(*p)
                if v:
                    return *v,i
            return None

        if not self.outWays:
            return []

        tmax = [0.]*self.order  # List of maximum line parameters of the projections of the intersection
        isectPointsL = [None]*self.order
        isectPointsR = [None]*self.order
        edgesL = [-1]*self.order
        edgesR = [-1]*self.order

        for indx, way in enumerate(self.outWays):
            borderL = way.polyline.parallelOffset(way.leftW)
            borderR = way.polyline.parallelOffset(way.rightW)
            # way.polyline.plot('k')

            isectL = intersectLineWithClusterPoly(borderL)
            isectR = intersectLineWithClusterPoly(borderR)

            if isectL:
                iPL,tPL = way.polyline.orthoProj(isectL[0])
                tmax[indx] = max(tmax[indx],tPL)
                isectPointsL[indx] = way.polyline.offsetPointAt(tPL,way.leftW)
                edgesL[indx] = isectL[2]

            if isectR:
                iPL,tPR = way.polyline.orthoProj(isectR[0])
                tmax[indx] = max(tmax[indx],tPR)
                isectPointsR[indx] = way.polyline.offsetPointAt(tPR,way.rightW)
                edgesR[indx] = isectR[2]

        polygon = []
        # Construct the intersection area polygon
        # def isBetween(a, b, c):
        #     crossproduct = (c.y - a.y) * (b.x - a.x) - (c.x - a.x) * (b.y - a.y)

        #     # compare versus epsilon for floating point values, or != 0 if using integers
        #     if abs(crossproduct) > epsilon:
        #         return False

        #     dotproduct = (c.x - a.x) * (b.x - a.x) + (c.y - a.y)*(b.y - a.y)
        #     if dotproduct < 0:
        #         return False

        #     squaredlengthba = (b.x - a.x)*(b.x - a.x) + (b.y - a.y)*(b.y - a.y)
        #     if dotproduct > squaredlengthba:
        #         return False

        #     return True
        for (indx1, indx2), (way1,way2) in zip(cyclePair(range(self.order)),cyclePair(self.outWays)):
            # <way1> and <way2> are consecutive outgoing polylines in counter-clockwise order of the
            # intersection. These are the centerlines of the way-sections, <way1> is to the right
            # (clockwise) of <way2>.
            # polygon = []
            # plt.close()
            # x = [n[0] for n in self.clusterPoly] + [self.clusterPoly[0][0]]
            # y = [n[1] for n in self.clusterPoly] + [self.clusterPoly[0][1]]
            # plt.plot(x,y,'k:',linewidth=1,zorder=999)

            isectPL =  isectPointsL[indx1]
            isectPR =  isectPointsR[indx2]
            # plt.plot(isectPL[0],isectPL[1],'go',markersize=6)
            # plt.plot(isectPR[0],isectPR[1],'ro',markersize=6)

            t1 = tmax[indx1]  # Line parameters of trim points.
            t2 = tmax[indx2]

            # The first way section will have to be trimmed to the length not occupied by
            # the polygon of the intersection area.
            way1.setTrim(t1)

            # Starting with the left border of the centerline <way1>.
            # Given the line parameter of the trim point on the centerline, we can 
            # compute the point <pL> perpendicularly offset to the left.
            pL = way1.polyline.offsetPointAt(t1,way1.leftW)
            # plt.plot(pL[0],pL[1],'gx',markersize=12)
            if isectPL:
                # If we had an intersection between the borders, by a simple check of
                # the coordinate difference to the trim point we may decide if
                # <pL> is the same as this intersection.
                dL = sum(pL-isectPL)
                if abs(dL) > 1.e-4:
                    # If not on the intersection, add to polygon
                    polygon.append(pL)
                    # then extend polygon by the intersection.
                polygon.append(isectPL)
            else:
                # Use only the trim point.
                polygon.append(pL)

            # Add eventual corner vertices
            diff = edgesR[indx1]-edgesR[indx2]
            if diff < 0:
                diff = (edgesR[indx1]+4-edgesR[indx2])
            for i in range(diff):
                indx = (edgesR[indx1]-i-1)%4
                polygon.append(self.clusterPoly[indx])
            #     p = self.clusterPoly[edgesR[indx2]]
            #     # plt.plot(p[0],p[1],'mo',markersize=12)
            #     plt.text(p[0],p[1],' '+str(indx1)+' '+str(indx2)+' '+str(edgesL[indx1]))

            # The same procedure is then done for the right border of the centerline <way2>
            pR = way2.polyline.offsetPointAt(t2,way2.rightW)
            # plt.plot(pR[0],pR[1],'gx',markersize=12)
            if isectPR:
                polygon.append(isectPR)
                dR = sum(pR-isectPR)
                if abs(dR) > 1.e-4:
                    polygon.append(pR)
            else:
                polygon.append(pR)

            # x = [n[0] for n in polygon]
            # y = [n[1] for n in polygon]
            # plt.plot(x,y,'c:',linewidth=5,zorder=999)
            # plt.gca().axis('equal')
            # plt.show()

        return polygon
