from itertools import permutations, tee,islice,cycle
from mathutils import Vector

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

from debugPlot import *

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
        self.hadOverlaps = False
        self.clustConnectors = dict()
        self.wayConnectors = dict()
        self.area = []

        # The area will be constructed in counter-clockwise order. Arange all data
        # in this order and so, that all ways (including the way-cluster) are seen
        # as outgoing ways of the area.
        self.centerline = subCluster.centerline if self.clustType=='start' else PolyLine(subCluster.centerline[::-1])
        self.widthL = subCluster.outWL() if self.clustType=='start' else subCluster.outWR()
        self.widthR = subCluster.outWR() if self.clustType=='start' else subCluster.outWL()
        split = subCluster.startSplit if self.clustType=='start' else subCluster.endSplit
        self.nodes = split.posW if self.clustType=='start' else split.posW[::-1]       # from right to left
        self.wIDs = split.currWIds if self.clustType=='start' else split.currWIds[::-1]   # from right to left

        nodeWays = []
        n = len(self.nodes)-1
        for indx,(node,inID) in enumerate(zip(self.nodes,self.wIDs)):
            wayPos = 'right' if indx==0 else 'left' if indx==n else 'mid'
            nodeWays.append( NodeWays(cls,self,node,inID,wayPos) )
        pass

# This class holds the bundle of out-ways of a node (at the end of a way-cluster).
class NodeWays():
    ID = 0
    def __init__(self,cls,handler,node,inID,wayPos):
        self.id = NodeWays.ID
        NodeWays.ID += 1
        self.cls = cls
        self.wayPos = wayPos
        self.node = node
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
        outVInID = self.outVector(node,inID) # Vector of the way into hte cluster.
        perpVL = Vector((outVInID[1],-outVInID[0]))/outVInID.length
        perpVR = Vector((-outVInID[1],outVInID[0]))/outVInID.length

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

        # TODO: Shall the rejected ways be invalidated?

        for Id in filteredIDs:
            self.outWays.append( OutWay(self.cls, self.node, outSectionIDs[Id-3]) )
        self.hasWays = True

        plotNodeWays(self)
        plotEnd()
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

    # Recursively check for simple overlaps and handle them.
    def handleOverlaps(self):
        if len(self.outWays) == 1:
            return

        # The overlap detection is done by two tests. 
        # 1. When one of the corners at the end of one way is inside the area of
        #    the carriageway of the other way, we have eventually an overlap.
        # 2. Executed only when the first test is True. When the union of the
        # carriageway areas of both ways has a hole, then its not an overlap.
        # Brute force, but there are only few ways!
        perms = permutations(range(len(self.outWays)),2)
        for i0,i1 in perms:
            # Test 1
            poly0 = self.outWays[i0].poly
            outWay = self.outWays[i1]
            if pointInPolygon(poly0,outWay.cornerL) == 'IN' or \
                pointInPolygon(poly0,outWay.cornerR) == 'IN':
                # Test 2
                poly1 = self.outWays[i1].poly
                union = boolPolyOp(poly0, poly1, 'union')
                print(len(union))
                if len(union) == 1:
                    self.handleOverlappingWay(outWay)
                # # Eventually one more overlap? Check recursively.
                # self.handleOverlaps()


# This class holds the data of a way leaving an end node
class OutWay():
    ID = 0
    def __init__(self,cls,node,wayId):
        self.id = OutWay.ID
        OutWay.ID += 1
        self.cls = cls
        self.node = node
        self.wayId = wayId

        self.section = cls.waySections[wayId]
        self.fwd = self.section.polyline[0] == node
        self.polyline = self.section.polyline if self.fwd else PolyLine(self.section.polyline[::-1])
        self.widthL = self.section.leftWidth if self.fwd else self.section.rightWidth
        self.widthR = self.section.rightWidth if self.fwd else self.section.leftWidth

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