from itertools import permutations, tee,islice,cycle
from mathutils import Vector

from lib.CompGeom.PolyLine import PolyLine
from lib.CompGeom.centerline import pointInPolygon
from lib.CompGeom.offset_intersection import offsetPolylineIntersection
from defs.way_cluster_params import transitionSlope

from way.overlap_handler_ext import areaFromEndClusters

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

# from osmPlot import *
from debugPlot import *

# This class holds all information to create the intersection area at one
# of the ends of a way-subcluster when its outways overlap. It is then able to
# create the intersection area and the corresponding clusters.
class OverlapHandler():
    ID = 0
    def __init__(self,cls,subCluster,clustType):
        self.id = OverlapHandler.ID
        OverlapHandler.ID += 1
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
        self.outIsects = split.posW if self.clustType=='start' else split.posW[::-1]       # from right to left
        self.wIDs = split.currWIds if self.clustType=='start' else split.currWIds[::-1]   # from right to left

        area = areaFromEndClusters(cls,self)
        return
        # for now, only the rightmost and leftmost sub-cluster is constructed.
        # TODO: Eventually add processing of intermediate sub-clusters?
        endSubClusterR = EndWaySubCluster(cls,self,self.outIsects[0],self.wIDs[0])
        if endSubClusterR.valid:
            if endSubClusterR.hasOverlaps:
                self.hadOverlaps = True

        endSubClusterL = EndWaySubCluster(cls,self,self.outIsects[-1],self.wIDs[-1])
        if endSubClusterL.valid:
            if endSubClusterL.hasOverlaps:
                self.hadOverlaps = True

        # No overlaps: Let it be processed by createClusterIntersections() in generate_streets.py
        if not self.hadOverlaps:
            return

        # plotWay(subCluster.centerline,subCluster.outWL(),subCluster.outWR(),'g')
        # if endSubClusterR.valid: endSubClusterR.plot()
        # if endSubClusterL.valid: endSubClusterL.plot()
        # plt.title('after overlap test')
        # plotEnd()

        # Construct intersection area and connectors.
        connectors = []
        # Borders are different depending on the number of clusters
        if endSubClusterR.valid and endSubClusterL.valid:
            border, conns = endSubClusterR.createBorder('right')
            self.area.extend(border)
            connectors.extend(conns)
            nextIndex = len(self.area)+1
            border, conns = endSubClusterL.createBorder('left')
            for conn in conns:
                conn[0] += nextIndex
            self.area.extend(border[:-2])
            connectors.extend(conns[:-1])
        elif endSubClusterR.valid:
            border, conns = endSubClusterR.createBorder('right')
            self.area.extend(border)
            connectors.extend(conns)
        elif endSubClusterL.valid:
            border, conns = endSubClusterL.createBorder('left')
            self.area.extend(border)
            connectors.extend(conns)

        # plotWay(subCluster.centerline,subCluster.outWL(),subCluster.outWR(),'g')
        # if endSubClusterR.valid: endSubClusterR.plot()
        # if endSubClusterL.valid: endSubClusterL.plot()
        # plotPolygon(self.area,False,'r','r',1,True,0.3)
        # for conn in connectors:
        #     object = conn[2]
        #     Id = ''
        #     if isinstance(object,OutWay):
        #         Id = str( object.lastExtendingSection.id if object.isExtended else object.section.id )
        #     p = conn[1]
        #     wayType = conn[2].__class__.__name__
        #     # plt.text(p[0],p[1],'Test',fontsize=12,color='k', rotation = -30,horizontalalignment='left',verticalalignment='top')
        #     plt.text(p[0],p[1],' '+str(conn[0])+' '+wayType+' '+Id,fontsize=12,color='k', rotation = -45,horizontalalignment='left',verticalalignment='top')
        # plt.title('area and connectors')
        # plotEnd()

        # Create the final connectors.
        conOffset = 0
        for conn in connectors:
            object = conn[2]
            if isinstance(object,OutWay):  # It's a way.
                Id = object.lastExtendingSection.id if object.isExtended else object.section.id
                Id = Id if object.fwd else -Id
                self.wayConnectors[Id] = conn[0]+conOffset
            else:   # It's a cluster. clustType
                cluster = object.cluster
                transWidth = cluster.transitionWidth if self.clustType == cluster.transitionPos else 0.1
                if self.clustType == 'start':
                    transitionLength = transWidth/transitionSlope
                    dTrans = transitionLength
                    tTrans = cluster.centerline.d2t(dTrans)
                    if tTrans > cluster.trimS:
                        p1 = cluster.centerline.offsetPointAt(tTrans,-cluster.outWR())
                        p2 = cluster.centerline.offsetPointAt(tTrans,cluster.outWL())
                        cluster.trimS = max(cluster.trimS,tTrans)
                        self.area.insert(conn[0]+conOffset+1,p1)
                        self.clustConnectors[cluster.id] = conn[0]+conOffset+1
                        self.area.insert(conn[0]+conOffset+2,p2)
                        conOffset += 2
                    else:
                        self.clustConnectors[cluster.id] = conn[0]+conOffset
                else:   # clustType == 'end'
                    transitionLength = transWidth/transitionSlope
                    dTrans = cluster.centerline.length() - transitionLength
                    tTrans = cluster.centerline.d2t(dTrans)
                    if tTrans < cluster.trimT:
                        p1 = cluster.centerline.offsetPointAt(tTrans,cluster.outWL())
                        p2 = cluster.centerline.offsetPointAt(tTrans,-cluster.outWR())
                        cluster.trimT = min(cluster.trimT,tTrans)
                        self.area.insert(conn[0]+conOffset+1,p1)
                        self.clustConnectors[-cluster.id] = conn[0]+conOffset+1
                        self.area.insert(conn[0]+conOffset+2,p2)
                        conOffset += 2
                    else:
                        self.clustConnectors[-cluster.id] = conn[0]+conOffset

# This class holds the out-way(s) from an end intersection of the cluster. It checks eventual overlaps,
# adds more ways when overlaps are present and creates its sub-area border and their connectors. 
class EndWaySubCluster():
    ID = 0
    def __init__(self,cls,endCluster,isect,inID):
        self.id = EndWaySubCluster.ID
        EndWaySubCluster.ID += 1
        self.cls = cls
        self.endCluster = endCluster
        self.valid = False
        self.hasOverlaps = False

        # Get the IDs of all way-sections that leave this node.
        self.node = isect.freeze()  # Called node, because it is a node in the section network
        outIds = self.getOutSectionIds(self.node,inID)

        # If it was an end-node, there are no out-ways. Invalidate this instance.
        if not outIds:
            self.valid = False
            return

        # Create an instance of <OutWay> for every leaving way-section
        self.outWays = []
        for Id in outIds:
            self.outWays.append( OutWay(self.cls, self.node, Id) )

        # Handle eventual overlaps between the way-sections.
        self.handleOverlaps()
        self.valid = True


    def createBorder(self,position):
        # Add cluster as out-way
        self.outWays.append(OutCluster(self.cls,self.endCluster.subCluster))

        # Create unit vector perpendicular to the last segment of the clsuters centerline,
        # pointing to the right.
        cluster = self.endCluster.subCluster
        fwd = cluster.startSplit.type == 'both'
        centerline = cluster.centerline if fwd else PolyLine(cluster.centerline[::-1])
        cV = centerline[0] - centerline[1]
        self.perpCenterline = Vector((cV[1],-cV[0]))/cV.length

        # Add dummy out-ways of given length from the node to the right and/or left.
        dummyR = OutDummy(self.cls,centerline[0],centerline[0]+self.perpCenterline*100)
        dummyL = OutDummy(self.cls,centerline[0],centerline[0]-self.perpCenterline*100)
        if position == 'right':
            self.outWays.append(dummyR)
        else:
            self.outWays.append(dummyL)

        self.sortOutways()

        # Find the out-way that is the cluster-way.
        clusterIndx = [i for i,w in enumerate(self.outWays) if isinstance(w,OutCluster)][0]
        if position == 'right':
            # Make the cluster-way to be the first way.
            self.outWays = self.outWays[clusterIndx:] + self.outWays[:clusterIndx]
        else:
            # Make the cluster-way to be the last way.
            self.outWays = self.outWays[clusterIndx+1:] + self.outWays[:clusterIndx+1]

        # plt.close()
        # for i,way in enumerate(self.outWays):
        #     plotWay(way.polyline,way.widthL,way.widthR,'k')
        #     p = way.polyline[-1]
        #     plt.text(p[0],p[1],'  '+str(i))
        # plotEnd()

        area, connectors = self.createArea()

        # Make area transition along end of cluster
        if position == 'right':
            transistionLIne = dummyR.polyline.parallelOffset(1.)
            p = transistionLIne.orthoProj(area[-1])[0]
            area.append(p)
        else:
            transistionLIne = dummyR.polyline.parallelOffset(1.)
            p = transistionLIne.orthoProj(area[0])[0]
            area = [p] + area


        # plotLine(area,True,'r',1)
        # for conn in connectors:
        #     p = conn[1]
        #     wayType = conn[2].__class__.__name__
        #     plt.text(p[0],p[1],'    '+str(conn[0])+' '+wayType,fontsize=16,color='r')
        # plotEnd()
        return area, connectors

    # Find the IDs of all way-sections that leave the <node>, but are not inside the way-cluster.
    def getOutSectionIds(self,node,inID):
        outSectionIDs = []
        if node in self.cls.sectionNetwork:
            for net_section in self.cls.sectionNetwork.iterOutSegments(node):
                if net_section.category != 'scene_border':
                    if net_section.sectionId != inID:
                        # If one of the nodes is outside of the clusters ...
                        if not (net_section.s in self.endCluster.outIsects and net_section.t in self.endCluster.outIsects):                               
                            outSectionIDs.append(net_section.sectionId)
                        else: # ... else we have to check if this section is inside or outside of the way-cluster.
                            nonNodeVerts = net_section.path[1:-1]
                            if len(nonNodeVerts): # There are vertices beside the nodes at the ends.
                                # Check, if any of these vertices is outside of the way-cluster
                                clusterPoly = self.endCluster.subCluster.centerline.buffer(self.endCluster.subCluster.endWidth/2.,self.endCluster.subCluster.endWidth/2.)
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
        # Brute force, but there are only few ways in an
        perms = permutations(range(len(self.outWays)),2)
        for i0,i1 in perms:
            poly = self.outWays[i0].poly
            outWay = self.outWays[i1]
            if pointInPolygon(poly,outWay.cornerL) == 'IN' or \
                pointInPolygon(poly,outWay.cornerR) == 'IN':
                self.hasOverlaps = True
                self.handleOverlappingWay(outWay)
                # Eventually one more overlap? Check recursively.
                self.handleOverlaps()

    def handleOverlappingWay(self,outWay):
        # The endpoint of outWay is an intersection. Find its outgoing ways.
        endP = outWay.polyline[-1]
        if outWay.isExtended:
            outWay.lastExtendingSection.isValid = False
            extendedIDs = self.getOutSectionIds(endP,outWay.lastExtendingID)
        else:
            outWay.section.isValid = False
            extendedIDs = self.getOutSectionIds(endP,outWay.wayId)
        if not extendedIDs:
            return
        self.cls.processedNodes.add(endP)
        # Create instances of <OutWay> for them.
        extOutWays = []
        for Id in extendedIDs:
            extOutWays.append( OutWay(self.cls, endP, Id) )

        # The way that best fits in angle is used to extend <outWay>.
        bestID = min( (i for i in range(len(extOutWays))), key=lambda x: abs(extOutWays[x].vIn.cross(outWay.vOut)) )
        # If it was already extended, invalidate the last extending section
        if outWay.isExtended:
            self.cls.waySections[outWay.lastExtendingID].isValid = False

        outWay.isExtended = True
        outWay.lastExtendingSection = self.cls.waySections[extendedIDs[bestID]]
        outWay.lastExtendingID = extendedIDs[bestID]
        outWay.extendedLength = len(outWay.polyline)-1
        outWay.polyline = outWay.polyline + extOutWays[bestID].polyline
        extWidthL = extOutWays[bestID].section.leftWidth if extOutWays[bestID].fwd else extOutWays[bestID].section.rightWidth
        extWidthR = extOutWays[bestID].section.rightWidth if extOutWays[bestID].fwd else extOutWays[bestID].section.leftWidth
        outWay.widthL = max(outWay.widthL,extWidthL)
        outWay.widthR =max(outWay.widthR,extWidthR)

        # outWay.polyline.plot('r',3)
        # The remaining extending outways are added to the end-way subcluster
        for i in range(len(extOutWays)):
            if i != bestID:
                self.outWays.append(extOutWays[i])
        # self.plot()
        # plt.title('found overlaps')
        # plotEnd()


    def sortOutways(self):
        self.outWays = sorted(self.outWays,key=lambda x: pseudoangle(x.polyline[1]-x.polyline[0]) )

    def createArea(self):
        # Find the intersections of the way borders.
        for way1,way2 in pairs(self.outWays):
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
        for way in self.outWays:
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

        # Transfer the trim values to the objects
        for outWay in self.outWays:
            outWay.trim()

        return area, connectors

    # def plot(self):
    #     for way in self.outWays:
    #         if isinstance(way,OutDummy):
    #             continue
    #         if way.isExtended:
    #             plotWay(way.polyline,way.widthL,way.widthR,'b')
    #             # way.polyline.plot('r',3,True)
    #         else:
    #             plotWay(way.polyline,way.widthL,way.widthR,'b')
    #             # way.polyline.plot('k',3,True)



# This class holds all data of a way leaving an intersection.
class OutWay():
    ID = 0
    def __init__(self,cls,node,wayId):
        self.id = OutWay.ID
        OutWay.ID += 1
        self.cls = cls
        self.wayId = wayId

        self.section = cls.waySections[wayId]
        self.fwd = self.section.polyline[0] == node
        self.polyline = self.section.polyline if self.fwd else PolyLine(self.section.polyline[::-1])
        self.widthL = self.section.leftWidth if self.fwd else self.section.rightWidth
        self.widthR = self.section.rightWidth if self.fwd else self.section.leftWidth
        self.trim_t = 0.

        self.isExtended = False
        self.lastExtendingSection = None
        self.lastExtendingID = None
        self.extendedLength = 0

    def trim(self):
        if self.isExtended:
            t = self.trim_t - self.extendedLength
            if self.fwd:
                self.lastExtendingSection.trimS = max(self.lastExtendingSection.trimS, t)
            else:
                t = len(self.lastExtendingSection.polyline)-1 - t
                self.lastExtendingSection.trimT = min(self.lastExtendingSection.trimT, t)
        else:
            if self.fwd:
                self.section.trimS = max(self.section.trimS, self.trim_t)
            else:
                t = len(self.section.polyline)-1 - self.trim_t
                self.section.trimT = min(self.section.trimT, t)

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

class OutDummy():
    ID = 0
    def __init__(self,cls,p0,p1):
        self.id = OutWay.ID
        OutWay.ID += 1
        self.cls = cls

        self.polyline = PolyLine([p0,p1])
        self.widthL = 0.001
        self.widthR = 0.001
        self.trim_t = 0.

    def trim(self):
        pass

class OutCluster():
    ID = 0
    def __init__(self,cls,cluster):
        self.id = OutWay.ID
        OutWay.ID += 1
        self.cls = cls

        self.cluster = cluster
        self.fwd = cluster.startSplit.type == 'both'
        self.polyline = cluster.centerline if self.fwd else PolyLine(cluster.centerline[::-1])
        self.widthL = cluster.outWL() if self.fwd else cluster.outWR()
        self.widthR = cluster.outWR() if self.fwd else cluster.outWL()
        self.trim_t = 0.

        self.isExtended = False
        self.lastExtendingSection = None
        self.lastExtendingID = None

    def trim(self):
        if self.fwd:
            self.cluster.trimS = max(self.cluster.trimS, self.trim_t)
        else:
            t = len(self.cluster.centerline)-1 - self.trim_t
            self.cluster.trimT = min(self.cluster.trimT, t)
