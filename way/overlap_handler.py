from itertools import permutations, tee,islice,cycle
from mathutils import Vector
from math import sin,cos,pi

from lib.CompGeom.PolyLine import PolyLine
from lib.CompGeom.centerline import pointInPolygon
from lib.CompGeom.BoolPolyOps import boolPolyOp
from lib.CompGeom.offset_intersection import offsetPolylineIntersection
from defs.way_cluster_params import transitionSlope

# helper functions -----------------------------------------------
def cyclePair(iterable):
    # iterable -> (p0,p1), (p1,p2), (p2, p3), ..., (pn, p0)
    prevs, nexts = tee(iterable)
    prevs = islice(cycle(prevs), len(iterable) - 1, None)
    return zip(prevs,nexts)

def pairs(iterable):
    # iterable -> (p0,p1), (p1,p2), (p2, p3), ...
    p1, p2 = tee(iterable)
    next(p2, None)
    return zip(p1,p2)

# Adapted from https://stackoverflow.com/questions/16542042/fastest-way-to-sort-vectors-by-angle-without-actually-computing-that-angle
# Input:  d: vector.
# Output: a number from the range [0 .. 4] which is monotonic
#         in the angle this vector makes against the x axis.
def pseudoangle(d):
    p = d[0]/(abs(d[0])+abs(d[1])) # -1 .. 1 increasing with x
    return 3 + p if d[1] < 0 else 1 - p 
# ----------------------------------------------------------------


# This class holds all information to create the intersection area and end
# of a way-subcluster.
class EndWayHandler():
    ID = 0
    def __init__(self,cls,subCluster,clustType):
        self.id = EndWayHandler.ID
        EndWayHandler.ID += 1
        self.cls = cls
        self.subCluster = subCluster
        self.clustType = clustType
        self.clustConnectors = dict()
        self.wayConnectors = dict()
        self.area = []
        self.nodeWays = []
        self.hasValidWays = False

        # The area will be constructed in counter-clockwise order. Arange all data
        # in this order and so, that all ways (including the way-cluster) are seen
        # as outgoing ways of the area.
        self.centerline = subCluster.centerline if self.clustType=='start' else PolyLine(subCluster.centerline[::-1])
        self.widthL = subCluster.outWL() if self.clustType=='start' else subCluster.outWR()
        self.widthR = subCluster.outWR() if self.clustType=='start' else subCluster.outWL()
        split = subCluster.startSplit if self.clustType=='start' else subCluster.endSplit
        self.nodes = split.posW if self.clustType=='start' else split.posW[::-1]       # from right to left
        self.wIDs = split.currWIds if self.clustType=='start' else split.currWIds[::-1]   # from right to left

        # For every intersection (node) at the end of the cluster, a bundle of leaving ways
        # is collected and preprocessed by the class NodeWays. <self.nodeWays> collects
        # them from right to left.
        n = len(self.nodes)-1
        for indx,(node,inID) in enumerate(zip(self.nodes,self.wIDs)):
            wayPos = 'right' if indx==0 else 'left' if indx==n else 'mid'
            self.nodeWays.append( NodeWays(cls,self,node,inID,wayPos) )

        # During the preprocessing, some ways may have been invalidated by one node,
        # while they have been accepted before by another one. This was like altering
        # an iterable during the iteration in a for loop. We have to clean them here.
        invalidatedWayIDs = []
        for nodeWay in self.nodeWays:
            for outWay in nodeWay.outWays:
                if not cls.waySections[outWay.wayId].isValid:
                    invalidatedWayIDs.append(outWay.wayId)
        if invalidatedWayIDs:
            for nodeWay in self.nodeWays:
                for outWay in nodeWay.outWays:
                    if outWay.wayId in invalidatedWayIDs:
                        outWay.valid = False

        self.hasValidWays = any(outWay.valid for nodeWay in self.nodeWays for outWay in nodeWay.outWays)

        if self.hasValidWays:
            self.createArea()

    def createArea(self):
        connectors = []
        borders = []
        nextIndex = 0
        for nodeWay in self.nodeWays:
            if nodeWay.hasWays:
                border, conns = nodeWay.createBorder()
                for conn in conns:
                    conn[0] += nextIndex
                borders.extend(border)
                connectors.extend(conns)
                nextIndex = len(borders)

        outCluster = OutCluster(self.cls,self.subCluster,self.clustType)
        t_max = 0.
        for p in borders:
            _,t = outCluster.polyline.orthoProj(p)
            t_max = max(t_max, t)
        outCluster.trim_t = t_max
        outCluster.trim()

        # Create the final connectors for the ways.
        for conn in connectors:
            p = conn[1]
            way = conn[2]
            Id = self.cls.waySections[way.wayId].id
            Id = Id if way.fwd else -Id
            self.wayConnectors[Id] = conn[0]

        # Create the connector for the way-cluster
        cluster = outCluster.cluster
        areaTrimDist = outCluster.polyline.t2d(outCluster.trim_t)
        transWidth = cluster.transitionWidth if self.clustType == cluster.transitionPos else 0.
        transitionLength = transWidth/transitionSlope + areaTrimDist + 0.1
        dTrans = transitionLength
        tTrans = outCluster.polyline.d2t(dTrans)
        outCluster.trim_t = max(outCluster.trim_t,tTrans)
        outCluster.trim()
        p1 = outCluster.polyline.offsetPointAt(outCluster.trim_t,-outCluster.widthR)
        p2 = outCluster.polyline.offsetPointAt(outCluster.trim_t,outCluster.widthL)
        cID = cluster.id if self.clustType=='start' else -cluster.id
        self.clustConnectors[cID] = len(borders)
        borders.append(p1)
        borders.append(p2)

        self.area = borders

# This class holds the bundle of out-ways of a node (at the end of a way-cluster).
class NodeWays():
    ID = 0
    def __init__(self,cls,handler,node,inID,wayPos):
        self.id = NodeWays.ID
        NodeWays.ID += 1
        self.cls = cls
        self.wayPos = wayPos
        self.node = node
        self.inID = inID
        self.handler = handler
        self.outWays = []
        self.hasWays = False

        # Get the IDs of all way-sections that leave this node.
        node.freeze()
        outSectionIDs = self.getOutSectionIds(node,inID)

        # If it was an end-node, there are no out-ways. Invalidate this instance.
        if not outSectionIDs:
            return

        # A filtering is required to removw ways that pint into the cluster, wherever
        # they came from. First we need to construct vectors that point to the right
        # an to the left, perpendicularly to the way in the cluster.
        outVInID = self.outVector(node,inID) # Vector of the way into the cluster.
        perpVL = Vector((outVInID[1],-outVInID[0]))/outVInID.length
        perpVR = Vector((-outVInID[1],outVInID[0]))/outVInID.length
        alpha = 10./180.*pi
        perpVL = Vector(( perpVL[0]*cos(alpha)-perpVL[1]*sin(alpha),perpVL[0]*sin(alpha)+perpVL[1]*cos(alpha) ))
        perpVR = Vector(( perpVR[0]*cos(-alpha)-perpVR[1]*sin(-alpha),perpVR[0]*sin(-alpha)+perpVR[1]*cos(-alpha) ))

        # These vectors are inserted as first elements into a list of vectors of all
        # ways leaving this node. The indices <Id> in <allVectors> are as follows:
        # 0: cluster-way
        # 1: perp to the left
        # 2: perp to the right
        # 3: first vector, computed from outSectionIDs[0]
        # :
        # Note that these are not the indices of waySections. These can be computed
        # to as wayID = outSectionIDs[Id-3]
        outVectors = [self.outVector(node, Id) for Id in outSectionIDs]
        allVectors = [self.outVector(node, inID),perpVL,perpVR] + outVectors

        # An argsort of the angles delivers a circular list of the above indices,
        # where the angles of the vectors around the node are eorted counter-clockwise,
        # and where the special vectors with indices 0 .. 2 are easily to find.
        sortedIDs = sorted( range(len(allVectors)), key=lambda indx: pseudoangle(allVectors[indx]))

        # Depending on the position of the node at the cluster end, only ways within an angle range
        # are allowed. 
        if self.wayPos == 'right':
            # All ways between the cluster-way and the left perp are valid.
            start = sortedIDs.index(0)  # index of cluster-way
            shifted = sortedIDs[start:] + sortedIDs[:start]  # shift this index as first element
            end = shifted.index(1)  # Index of left perp
            filteredIDs = [Id for Id in shifted[0:end] if Id not in [0,1,2]]
        elif self.wayPos == 'left':
            # All ways between the right perp and the cluster-way are valid.
            start = sortedIDs.index(2)  # index of right perp
            shifted = sortedIDs[start:] + sortedIDs[:start]  # shift this index as first element
            end = shifted.index(0)  # Index of lcluster-way
            filteredIDs = [Id for Id in shifted[0:end] if Id not in [0,1,2]]
        elif self.wayPos == 'mid': 
            # All ways between the right perp and left perp are valid.
            start = sortedIDs.index(2)  # index of right perp
            shifted = sortedIDs[start:] + sortedIDs[:start]  # shift this index as first element
            end = shifted.index(1)  # Index of left perp
            filteredIDs = [Id for Id in shifted[0:end] if Id not in [0,1,2]]

        # Invalidate rejected ways.
        invalidIds = [outSectionIDs[Id-3] for Id in shifted if Id not in filteredIDs + [0,1,2]]
        for Id in invalidIds:
            self.cls.waySections[Id].isValid = False

        for Id in filteredIDs:
            self.outWays.append( OutWay(self.cls, self.node, outSectionIDs[Id-3]) )
        self.hasWays = True

        # Handle eventual overlaps between the way-sections.
        self.handleOverlaps()

    def outVector(self,node, wayID):
        s = self.cls.waySections
        outV = s[wayID].polyline[1]-s[wayID].polyline[0] if s[wayID].polyline[0] == node else s[wayID].polyline[-2]-s[wayID].polyline[-1]
        outV /= outV.length
        return outV

    # Find the IDs of all way-sections that leave the <node>, but are not inside the way-cluster.
    def getOutSectionIds(self,node,inID):
        outSectionIDs = []
        if node in self.cls.sectionNetwork:
            for net_section in self.cls.sectionNetwork.iterOutSegments(node):
                if net_section.category != 'scene_border':
                    if net_section.sectionId != inID:
                        # If one of the nodes is outside of the clusters ...
                        if not (net_section.s in self.handler.nodes and net_section.t in self.handler.nodes):                               
                            outSectionIDs.append(net_section.sectionId)
                        else: # ... else we have to check if this section is inside or outside of the way-cluster.
                            nonNodeVerts = net_section.path[1:-1]
                            if len(nonNodeVerts): # There are vertices beside the nodes at the ends.
                                # Check, if any of these vertices is outside of the way-cluster
                                clusterPoly = self.handler.subCluster.centerline.buffer(self.handler.subCluster.endWidth/2.,self.handler.subCluster.endWidth/2.)
                                if any( pointInPolygon(clusterPoly,vert) == 'OUT' for vert in nonNodeVerts ):
                                    outSectionIDs.append(net_section.sectionId) # There are, so keep this way ...
                                    continue
                                # ... else invalidate this way
                                self.cls.waySections[net_section.sectionId].isValid = False
        return outSectionIDs    

    # Ceck for simple overlaps and iteratively handle them.
    def handleOverlaps(self):
        if len(self.outWays) == 1:
            return

        while True: # Do this process until no more overlaps have been detected.
            noMoreOverlaps = True
            # The overlap detection is done by two tests. 
            # 1. When one of the corners at the end of one way is inside the area of
            #    the carriageway of the other way, we have eventually an overlap.
            # 2. Executed only when the first test is True. When the union of the
            # carriageway areas of both ways has a hole, then it's not an overlap.
            # Brute force, but there are only few ways!
            perms = permutations(range(len(self.outWays)),2)
            for i0,i1 in perms:
                # Ways in outWays may be invalifated furing this process.
                if self.outWays[i0].valid and self.outWays[i1].valid:
                    # Test 1
                    poly0 = self.outWays[i0].poly
                    outWay = self.outWays[i1]
                    if pointInPolygon(poly0,outWay.cornerL) == 'IN' or \
                        pointInPolygon(poly0,outWay.cornerR) == 'IN':
                        # Test 2
                        poly1 = self.outWays[i1].poly
                        union = boolPolyOp(poly0, poly1, 'union')
                        if len(union) == 1:
                            self.handleOverlappingWay(outWay)
                            noMoreOverlaps = False
            if noMoreOverlaps:
                break

    def handleOverlappingWay(self,outWay):
        # The endpoint of outWay may be an intersection.
        endP = outWay.polyline[-1]
        # Find ways that leave this intersection.
        leavingIDs = self.getOutSectionIds(endP,outWay.wayId)
        # If there aren't leaving ways, this conflcitin way is invalidated.
        if not leavingIDs:
            outWay.section.isValid = False
            outWay.valid = False
            return

        self.cls.processedNodes.add(endP.freeze())
        # Create instances of <OutWay> for leaving ways.
        leavingWays = []
        for Id in leavingIDs:
            leavingWays.append( OutWay(self.cls, endP, Id) )

        # The leaving way who's first segment best fits to the last segment
        # of the conflicting out-way <outWay> is selected to extend <outWay>.
        bestFitID = min( (i for i in range(len(leavingWays))), key=lambda x: abs(leavingWays[x].vIn.cross(outWay.vOut)) )
        outWay.extendBy(leavingIDs[bestFitID])

        # The remaining leaving ways are added to the outWays.
        for i in range(len(leavingWays)):
            if i != bestFitID:
                self.outWays.append(leavingWays[i])

    def createBorder(self):
        borderWays = self.outWays
        # Add way that is within cluster
        clusterWay =  OutWay(self.cls, self.node, self.inID)
        borderWays.append( clusterWay )
        borderWays = sorted(borderWays,key=lambda x: pseudoangle(x.polyline[1]-x.polyline[0]) )

        # Roll clusterWay to the start of the list.
        clusterWayIndx = [i for i,w in enumerate(borderWays) if w.id==clusterWay.id][0]
        borderWays = borderWays[clusterWayIndx:] + borderWays[:clusterWayIndx]

        area, connectors = self.createArea(borderWays)

        border = area[1:] + [area[0]]
        connectors = [[Id-1,pos,object] for (Id,pos,object) in connectors[1:]]

        # for way in borderWays:
        #     if way.valid:
        #         plotPolygon(way.poly,'b',1,False,False,True)
        # plotPoint(self.node,'r')
        # clusterWay.polyline.plot('g',3)
        # plotLine(border,'r',1,False,True)
        # for connector in connectors:
        #     p = connector[1]
        #     plt.text(p[0],p[1],'     '+str(connector[2].id),fontsize=18,color='r')
        # plotEnd()

        return border, connectors

 
    def createArea(self, borderWays):
        # Find the intersections of the way borders.
        for way1,way2 in cyclePair(borderWays):
            p, type = offsetPolylineIntersection(way1.polyline,way2.polyline,way1.widthL,way2.widthR)
            if type == 'valid':
                _,t1 = way1.polyline.orthoProj(p)
                _,t2 = way2.polyline.orthoProj(p)
                way1.trim_t = max(way1.trim_t, t1)
                way2.trim_t = max(way2.trim_t, t2)
            elif type == 'parallel':
                way1.trim_t = max(way1.trim_t, way1.polyline.d2t(0.))
                way2.trim_t = max(way2.trim_t, way2.polyline.d2t(0.))
            else:       # 'out'
                pass    # do nothing

        # Create the intersection area and the connectors
        area = []
        connectors = []
        for way in borderWays:
            wayType = way.__class__.__name__ 
            lp = way.polyline.offsetPointAt(way.trim_t,way.widthL)
            rp = way.polyline.offsetPointAt(way.trim_t,-way.widthR)
            if not area or rp != area[-1]:
                area.extend([rp,lp])
                if wayType in ['OutWay','OutCluster']:
                    connectors.append( [len(area)-2,rp, way] )
            else:
                if wayType in ['OutWay','OutCluster']:
                    connectors.append( [len(area)-1,rp, way] )
                area.append(lp)
        if area[0] == area[-1]:
            area = area[:-1]

        # # Transfer the trim values to the objects
        for outWay in self.outWays:
            outWay.trim()

        return area, connectors

# This class holds the data of a way leaving an end node
class OutWay():
    ID = 0
    def __init__(self,cls,node,wayId):
        self.id = OutWay.ID
        OutWay.ID += 1
        self.cls = cls
        self.node = node
        self.wayId = wayId
        self.valid = True

        self.trimOrigin = 0.
        self.trim_t = 0

        self.section = cls.waySections[wayId]
        self.fwd = self.section.polyline[0] == node
        self.polyline = self.section.polyline if self.fwd else PolyLine(self.section.polyline[::-1])
        self.widthL = self.section.leftWidth if self.fwd else self.section.rightWidth
        self.widthR = self.section.rightWidth if self.fwd else self.section.leftWidth

    def trim(self):
            t0 = self.trim_t - self.trimOrigin
            if self.fwd:
                self.section.trimS = max(self.section.trimS, t0)
            else:
                t = len(self.section.polyline)-1 - t0
                self.section.trimT = min(self.section.trimT, t)

    @property
    # Polygon of the carriage way.
    def poly(self):
        return self.polyline.buffer(self.widthL,self.widthR)

    @property
    # Corner position of the left end of the carriage way.
    def cornerL(self):
        return self.polyline.offsetPointAt(len(self.polyline)-1,self.widthL)

    @property
    # Corner position of the right end of the carriage way.
    def cornerR(self):
        return self.polyline.offsetPointAt(len(self.polyline)-1,-self.widthR)

    @property
    # Unit vector of the first segment, pointing inside.
    def vIn(self):
        v = self.polyline[1] - self.polyline[0]
        # plotLine([self.polyline[1],self.polyline[0]],False,'c',4,)
        # p = self.polyline[1]
        # plt.plot(p[0],p[1],'co',markersize=8)
        return v/v.length

    @property
    # Unit vector of the last segment, pointing outside.
    def vOut(self):
        v = self.polyline[-1] - self.polyline[-2]
        # plotLine([self.polyline[-1],self.polyline[-2]],False,'c',4)
        # p = self.polyline[-1]
        # plt.plot(p[0],p[1],'co',markersize=8)
        return v/v.length

    def extendBy(self,leavingID):
        ext_section = self.cls.waySections[leavingID]
        ext_fwd = ext_section.polyline[0] == self.section.polyline[-1]
        extPolyline = ext_section.polyline if ext_fwd else PolyLine(ext_section.polyline[::-1])
        ext_widthL = ext_section.leftWidth if ext_fwd else ext_section.rightWidth
        ext_widthR = ext_section.rightWidth if ext_fwd else ext_section.leftWidth

        self.section.isValid = False

        self.wayId = leavingID
        self.section = ext_section
        self.trimOrigin = len(self.polyline)-1
        self.polyline = self.polyline + extPolyline
        self.widthL = ext_widthL
        self.widthR = ext_widthR

class OutCluster():
    ID = 0
    def __init__(self,cls,cluster,clustType):
        self.id = OutWay.ID
        OutWay.ID += 1
        self.cls = cls

        self.cluster = cluster
        self.fwd = clustType == 'start'
        self.polyline = cluster.centerline if self.fwd else PolyLine(cluster.centerline[::-1])
        self.widthL = cluster.outWL() if self.fwd else cluster.outWR()
        self.widthR = cluster.outWR() if self.fwd else cluster.outWL()
        self.trim_t = 0.

    def trim(self):
        if self.fwd:
            self.cluster.trimS = max(self.cluster.trimS, self.trim_t)
        else:
            t = len(self.cluster.centerline)-1 - self.trim_t
            self.cluster.trimT = min(self.cluster.trimT, t)
