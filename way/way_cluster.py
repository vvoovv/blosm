from itertools import tee

from defs.way_cluster_params import transitionLimit,transitionSlope
from way.way_properties import estimateWayWidth
from way.intersection_cluster import IntersectionCluster
from way.way_section import WaySection
from lib.CompGeom.PolyLine import PolyLine
from lib.CompGeom.offset_intersection import offsetPolylineIntersection

# from osmPlot import *

# helper functions -----------------------------------------------
def pairs(iterable):
    # iterable -> (p0,p1), (p1,p2), (p2, p3), ...
    p1, p2 = tee(iterable)
    next(p2, None)
    return zip(p1,p2)

def widthOrthoCenterline(centerline,left,right,p):
    pc,_ = centerline.orthoProj(p)
    _,dL = left.distTo(pc)
    _,dR = right.distTo(pc)
    return dL + dR
# ----------------------------------------------------------------
class SplitDescriptor():
    ID = 0
    def __init__(self,t,type,posL,posR,currWIds):
        self.id = SplitDescriptor.ID
        SplitDescriptor.ID += 1
        self.t = t          # Parameter <t> on centerline
        self.type = type    # Type of split ('left','right','both','end','clipped',...)
        self.posL = posL.freeze()   # Position of vertex on left way
        self.posR = posR.freeze()   # Position of vertex on right way
        self.posW = None            # List of positions of all ways
        self.currWIds = currWIds    # List of current way-IDs of way-sections
                                    # at split position in order left to right

class LongClusterWay():
    ID = 0
    def __init__(self,cls):
        self.id = LongClusterWay.ID
        LongClusterWay.ID += 1
        # Reference to the class StreetGenerator
        self.cls = cls
        # List of section IDs of the individual ways
        self.sectionIDs = []
        # List of the intersection IDs in the individual ways
        self.intersections = []
        # List of the polylines of the individual ways
        self.polylines = []
        # The centerline of the long cluster
        self.centerline = None
        # Clip status at end: 'clipL' or 'clipR' in one line , 'clipB'
        # when both lines clipped, None, when not clipped.
        self.clipped = None
        # List of subcluster of class <ClusterWay>
        self.subClusters = []

    def split(self):
        # The long cluster has now to be split into subclusters. Reasons for split positons
        # are intersections on the outermost ways left and right and possble start and 
        # end points of short inner ways. Reference line is the cluster's centerline. 

        # Find intersections on outermost (left and right) ways along cluster and create
        # their descriptors.
        clusterSplits = []
        left, right =  self.polylines[0], self.polylines[-1]
        # wIdsL, wIdsR = self.sectionIDs[0], self.sectionIDs[-1]
        nrOfWays = len(self.polylines)
        wIs = [0]*nrOfWays  # Curent indices of way-sections self.sectionIDs along line
        # Split descriptor for start of subcluster
        clusterSplits.append( SplitDescriptor(0.,'both',left[0],right[0],[self.sectionIDs[k][i] for k,i in enumerate(wIs)]) )

        # Position <p> and parameter <t> of all intersections on all lines, sorted by <t>.
        for i in range(len(self.intersections)):
            self.intersections[i] = list(dict.fromkeys(self.intersections[i]))
        splits = [(lineNr,sectNr,p,*self.centerline.orthoProj(p)) for lineNr,isects in enumerate(self.intersections) for sectNr,p in enumerate(isects)]
        splits.sort(key=lambda x: x[4]) # Sort by <t>
        # for p,t in splits:
        for split in splits:
            lineNr,sectNr,isect, pos, t = split
            posC,_ = self.polylines[lineNr].intersectWithLine(isect,pos)
            wIs[lineNr] = wIs[lineNr]+1
            if lineNr == 0: # left              
                clusterSplits.append( SplitDescriptor(t,'left',isect,posC,[self.sectionIDs[k][i] for k,i in enumerate(wIs)]) )
            elif lineNr == nrOfWays-1: # right
                clusterSplits.append( SplitDescriptor(t,'right',posC,isect,[self.sectionIDs[k][i] for k,i in enumerate(wIs)]) )
            else:   # inner
                clusterSplits.append( SplitDescriptor(t,'inner',isect,posC,[self.sectionIDs[k][i] for k,i in enumerate(wIs)]) )

        # Finally, the split descriptor for the end of the long cluster has to be constructed.
        # At the end, the ways may have been clipped
        tEnd = len(self.centerline)
        if self.clipped:
            eL = self.sectionIDs[0][-1]
            eR = self.sectionIDs[-1][-1]
            clusterSplits.append( SplitDescriptor(tEnd,self.clipped,left[-1],right[-1],[[eL,],[eR,]]) )
        else:
            clusterSplits.append( SplitDescriptor(tEnd,'both',left[-1],right[-1],[ids[-1] for ids in self.sectionIDs]) )
        clusterSplits.sort(key=lambda x: x.t)

        # Create clusters, split them when there are intermdiate intersections
        for cs0,cs1 in pairs(clusterSplits):
            subCluster = SubCluster(self.centerline.trimmed(cs0.t,cs1.t))
            subCluster.length = subCluster.centerline.length()
            subCluster.longCluster = self
            subCluster.wayIDs = cs0.currWIds
            subCluster.startWidth = widthOrthoCenterline(subCluster.centerline,left,right,cs0.posL)
            subCluster.endWidth   = widthOrthoCenterline(subCluster.centerline,left,right,cs1.posL)
         
            # Prepare positions of all ways at splits
            cs0.posW = [cs0.posL]
            for line in self.polylines[1:-1]:
                posC = line.orthoProj(subCluster.centerline[0])[0]
                cs0.posW.append(posC)
            cs0.posW.append(cs0.posR)
            subCluster.startSplit = cs0

            cs1.posW = [cs1.posL]
            for line in self.polylines[1:-1]:
                posC = line.orthoProj(subCluster.centerline[-1])[0]
                cs1.posW.append(posC)
            cs1.posW.append(cs1.posR)
            subCluster.endSplit = cs1

            # prepare data eventually required for width transition
            widthDifference = abs(subCluster.startWidth - subCluster.endWidth)
            if widthDifference < transitionLimit:
                subCluster.width = (subCluster.startWidth + subCluster.endWidth)/2.
                subCluster.transitionWidth = 0.
                subCluster.transitionPos = None
            else:
                subCluster.width = min(subCluster.startWidth, subCluster.endWidth)
                subCluster.transitionWidth = widthDifference
                subCluster.transitionPos = 'start' if subCluster.startWidth > subCluster.endWidth else 'end'

            for Id in subCluster.wayIDs:
                category = self.cls.waySections[Id].originalSection.category
                tags = self.cls.waySections[Id].originalSection.tags
                subCluster.wayWidths.append(estimateWayWidth(category,tags))

            self.subClusters.append(subCluster)

class SubCluster():
    ID = 0
    def __init__(self,centerline):
        self.id = SubCluster.ID
        SubCluster.ID += 1
        self.longCluster = None
        self.centerline = centerline
        self.width = None
        self.startWidth = None
        self.endWidth = None
        self.startSplit = None
        self.endSplit = None
        self.wayIDs = None
        self.wayWidths = []
        self.transitionWidth = None
        self.transitionPos = None
        self.trimS = 0.                      # trim factor for start
        self.trimT = len(self.centerline)-1  # trim factor for target
        self.length = None
        self.valid = True

    # def transitionBecauseWidth(self):
    #     widthDiff = self.endWidth - self.startWidth
    #     if abs(widthDiff) > widthThresh:
    #         w = self.endWidth
    #         d = self.centerline.length() - 
    #         t = len(self.centerline)
    #         pL = self.centerline.offsetPointAt(t,abs(w/2))
    #         pR = self.centerline.offsetPointAt(t,-abs(w/2))
    #         end = ('trans',pL,pR,self.leftWay,self.rightWay)
    #         segCluster = ClusterWay(self.centerline)
    #         segCluster.startPoints = self.startPoints
    #         segCluster.endPoints = end
    #         segCluster.leftWay = self.leftWay
    #         segCluster.rightWay = self.rightWay
    #         segCluster.startWidth = self.startWidth

    def outW(self):
        return self.width + (self.wayWidths[0]+self.wayWidths[-1])/2.
    def outWL(self):
        return (self.width + self.wayWidths[0])/2.
    def outWR(self):
        return (self.width + self.wayWidths[-1])/2.
    def inWL(self):
        return (self.width - self.wayWidths[0])/2.
    def inWR(self):
        return (self.width - self.wayWidths[-1])/2.

def computeTransitionWidths(cluster1,cluster2):
    # Inter-cluster transition width. Clusters have different widths. Positive if <cluster2 is
    # thicker. We correct it on the smaller cluster.
    interTransWidth = cluster2.outW()-cluster1.outW()

    # Required transitionwidth for cluster1
    transWidth1 = interTransWidth if interTransWidth > 0. else 0.
    transWidth1 += cluster1.transitionWidth if cluster1.transitionPos == 'end' else 0.
    if abs(transWidth1) < transitionLimit: transWidth1 = 0.
    
    # Required transitionwidth for cluster2
    transWidth2 = -interTransWidth if interTransWidth < 0. else 0.
    transWidth2 += cluster2.transitionWidth if cluster2.transitionPos == 'start' else 0.
    if abs(transWidth2) < transitionLimit: transWidth2 = 0.

    return max(transWidth1,0.1), max(transWidth2,0.1)

def widthAndLineFromID(cls,node,ID):
    section = cls.waySections[ID]
    fwd = section.originalSection.s == node
    category = section.originalSection.category
    tags = section.originalSection.tags
    width = estimateWayWidth(category,tags)
    line = PolyLine(section.fwd() if fwd else section.rev())
    return line,width

def createLeftTransition(cls, cluster1, cluster2):
    area = []
    transWidth1, transWidth2 = computeTransitionWidths(cluster1,cluster2)

    # Construct area through cluster1 (counter-clockwise order)
    transitionLength = transWidth1/transitionSlope
    dTrans = cluster1.centerline.length() - transitionLength
    tTrans = cluster1.centerline.d2t(dTrans)
    cluster1.trimT = min(cluster1.trimT,tTrans)
    # Two additional points due to transition
    p = cluster1.centerline.offsetPointAt(cluster1.trimT,cluster1.outWL())
    area.append(p)
    if cls.isectShape == 'common':
        p = cluster1.centerline.offsetPointAt(cluster1.trimT,-cluster1.outWR())
    elif cls.isectShape == 'separated':
        p = cluster1.centerline.offsetPointAt(cluster1.trimT,cluster1.inWL())
    area.append(p)

    # Construct area through cluster2 (counter-clockwise order)
    # if transWidth2:
    transitionLength = transWidth2/transitionSlope
    dTrans = transitionLength
    tTrans = cluster2.centerline.d2t(dTrans)
    cluster2.trimS = max(cluster2.trimS,tTrans)
    # Two additional points due to transition
    if cls.isectShape == 'common':
        p = cluster2.centerline.offsetPointAt(cluster2.trimS,-cluster2.outWR())
    elif cls.isectShape == 'separated':
        p = cluster2.centerline.offsetPointAt(cluster2.trimS,cluster2.inWL())
    area.append(p)
    p = cluster2.centerline.offsetPointAt(cluster2.trimS,cluster2.outWL())
    area.append(p)

    clustConnectors = dict()
    clustConnectors[cluster1.id] = ( 0, 'E', -1 if cls.isectShape == 'common' else 0)
    clustConnectors[cluster2.id] = ( 2, 'S', -1 if cls.isectShape == 'common' else 0)
    return area, clustConnectors

def createRightTransition(cls, cluster1, cluster2):
    area = []
    transWidth1, transWidth2 = computeTransitionWidths(cluster1,cluster2)

    # Construct area through cluster2 (counter-clockwise order)
    transitionLength = transWidth2/transitionSlope
    dTrans = transitionLength
    tTrans = cluster2.centerline.d2t(dTrans)
    cluster2.trimS = max(cluster2.trimS,tTrans)
    # Two additional points due to transition
    if cls.isectShape == 'common':
        p = cluster2.centerline.offsetPointAt(cluster2.trimS,-cluster2.outWR())
    elif cls.isectShape == 'separated':
        p = cluster2.centerline.offsetPointAt(cluster2.trimS,cluster2.inWL())
    area.append(p)
    p = cluster2.centerline.offsetPointAt(cluster2.trimS,cluster2.outWL())
    area.append(p)

    # Construct area through cluster1 (counter-clockwise order)
    transitionLength = transWidth1/transitionSlope
    dTrans = cluster1.centerline.length() - transitionLength
    tTrans = cluster1.centerline.d2t(dTrans)
    cluster1.trimT = min(cluster1.trimT,tTrans)
    # Two additional points due to transition
    p = cluster1.centerline.offsetPointAt(cluster1.trimT,cluster1.outWL())
    area.append(p)
    if cls.isectShape == 'common':
        p = cluster1.centerline.offsetPointAt(cluster1.trimT,-cluster1.outWR())
    elif cls.isectShape == 'separated':
        p = cluster1.centerline.offsetPointAt(cluster1.trimT,-cluster1.inWR())
    area.append(p)

    clustConnectors = dict()
    clustConnectors[cluster2.id] = ( 0, 'S', -1 if cls.isectShape == 'common' else 1)
    clustConnectors[cluster1.id] = ( 2, 'E', -1 if cls.isectShape == 'common' else 1)
    return area, clustConnectors

def createLeftIntersection(cls, cluster1, cluster2, node,blockedWayIDs):
    cL = cluster1.centerline + cluster2.centerline
    cLr = PolyLine(cL[::-1])
    wayConnectors = dict()
    clustConnectors = dict()
    # find width and line of outgoing way
    outWayID = [w.sectionId for w in cls.sectionNetwork.iterOutSegments(node) if w.sectionId not in blockedWayIDs \
        and cls.waySections[w.sectionId].isValid][0]
    outLine, outWidth = widthAndLineFromID(cls,node,outWayID)

    # On the left side of the cluster, for an intersection area in counter-clockwise
    # order, the clusters are in reversed order.

    # Find intersections <p1> and <p3> of way borders left and right of the out-way
    # with the left border of cluster 2.
    outW = max(cluster1.outWL(),cluster2.outWL())
    # p1, valid = offsetPolylineIntersection(cluster2.centerline,outLine,outW,outWidth/2.)
    p1, valid = offsetPolylineIntersection(cL,outLine,outW,outWidth/2.)
    # if valid!='valid':
    #     plotPureNetwork(cls.sectionNetwork)
    #     cluster1.centerline.plot('g:',3)
    #     cluster2.centerline.plot('r',3)
    #     plt.plot(p1[0],p1[1],'ro')
    #     outLine.plot('c:',3)
    #     plt.title('createLeftIntersection, p1')
    #     plotEnd()
    assert valid=='valid'
    reverseCenterline = PolyLine(cluster1.centerline[::-1])
    # p3, valid = offsetPolylineIntersection(outLine,reverseCenterline,outWidth/2.,outW)
    p3, valid = offsetPolylineIntersection(outLine,cLr,outWidth/2.,outW)
    # if valid!='valid':
    #     plotPureNetwork(cls.sectionNetwork)
    #     cluster1.centerline.plot('g:',3)
    #     cluster2.centerline.plot('r',3)
    #     plt.plot(p3[0],p3[1],'ro')
    #     outLine.plot('c:',3)
    #     plt.title('createLeftIntersection, p3')
    #     plotEnd()
    assert valid=='valid'

    # Project these onto cluster centerlines to get initial trim values
    _,t = cluster1.centerline.orthoProj(p3)
    cluster1.trimT = min(cluster1.trimT, t)
    _,t = cluster2.centerline.orthoProj(p1)
    cluster2.trimS = max(cluster2.trimS, t)

    # Project these onto the centerline of the out-way and create intermediate
    # polygon point <p2>.
    _,tP1 = outLine.orthoProj(p1)
    _,tP3 = outLine.orthoProj(p3)
    section = cls.waySections[outWayID]
    fwd = section.originalSection.s == node
    if tP3 > tP1:
        p2 = outLine.offsetPointAt(tP3,-outWidth/2.)
        tP = tP3
        wayConnectors[section.id] = ( 1, 'S' if fwd else 'E')
    else:
        p2 = outLine.offsetPointAt(tP1,outWidth/2.)
        tP = tP1
        wayConnectors[section.id] = ( 0, 'S' if fwd else 'E')
    if fwd:
        section.trimS = max(section.trimS,tP)
    else:
        section.trimT = min(section.trimT,len(outLine)-1-tP)

    # start area in counter-clockwise order, continue at end of cluster1
    area = [p1,p2,p3] if p1!=p2 and p2!=p3 else [p1,p3] 

    # cluster1.centerline.plot('g:',3)
    # cluster2.centerline.plot('r:',3)
    # plt.plot(p1[0],p1[1],'ro')
    # plt.text(p1[0],p1[1],'p1')
    # plt.plot(p2[0],p2[1],'ro')
    # plt.text(p2[0],p2[1],'p2')
    # plt.plot(p3[0],p3[1],'ro')
    # plt.text(p3[0],p3[1],'p3')
    # outLine.plot('c',4)


    transWidth1, transWidth2 = computeTransitionWidths(cluster1,cluster2)

    # Construct area through cluster1 (counter-clockwise order). On the left side,
    # we start with the first cluster.
    if transWidth1:
        transitionLength = transWidth1/transitionSlope
        dTrans = cluster1.centerline.length() - transitionLength
        tTrans = cluster1.centerline.d2t(dTrans)
        cluster1.trimT = min(cluster1.trimT,tTrans)
        # Two additional points due to transition, <p4> on the left and <p5> on 
        # the right of the cluster.
        p4 = cluster1.centerline.offsetPointAt(cluster1.trimT,cluster1.outWL())
        area.append(p4)
        clustConnectors[cluster1.id] = ( len(area)-1, 'E', -1 if cls.isectShape == 'common' else 0)
        if cls.isectShape == 'common':
            p5 = cluster1.centerline.offsetPointAt(cluster1.trimT,-cluster1.outWR())
        elif cls.isectShape == 'separated':
            p5 = cluster1.centerline.offsetPointAt(cluster1.trimT,cluster1.inWL())
        area.append(p5)
    else:
        # No transistion, we construct directly <p5> on the right of the cluster.
        clustConnectors[cluster1.id] = ( len(area)-1, 'E', -1 if cls.isectShape == 'common' else 0)
        _,tP = cluster1.centerline.orthoProj(p3)
        cluster1.trimT = min(cluster1.trimT,tP)
        if cls.isectShape == 'common':
            p5 = cluster1.centerline.offsetPointAt(tP,-cluster1.outWR())
        elif cls.isectShape == 'separated':
            p5 = cluster1.centerline.offsetPointAt(tP,cluster1.inWL())
        area.append(p5)

    # Construct area through cluster2 (counter-clockwise order)
    if transWidth2:
        transitionLength = transWidth2/transitionSlope
        dTrans = transitionLength
        tTrans = cluster2.centerline.d2t(dTrans)
        cluster2.trimS = max(cluster2.trimS,tTrans)
        # Two additional points due to transition
        if cls.isectShape == 'common':
            p6 = cluster2.centerline.offsetPointAt(cluster2.trimS,-cluster2.outWR())
        elif cls.isectShape == 'separated':
            p6 = cluster2.centerline.offsetPointAt(cluster2.trimS,cluster2.inWL())
        area.append(p6)
        clustConnectors[cluster2.id] = ( len(area)-1, 'S', -1 if cls.isectShape == 'common' else 0)
        p7 = cluster2.centerline.offsetPointAt(cluster2.trimS,cluster2.outWL())
        area.append(p7)
    else:
        _,tP = cluster2.centerline.orthoProj(p1)
        cluster2.trimS = max(cluster2.trimS,tP)
        if cls.isectShape == 'common':
            p7 = cluster2.centerline.offsetPointAt(cluster2.trimS,-cluster2.outWR())
        elif cls.isectShape == 'separated':
            p7 = cluster2.centerline.offsetPointAt(cluster2.trimS,cluster2.inWL())
        area.append(p7)
        clustConnectors[cluster2.id] = ( len(area)-1, 'S', -1 if cls.isectShape == 'common' else 0)
 
    return area, clustConnectors, wayConnectors

def createRightIntersection(cls, cluster1, cluster2, node, blockedWayIDs):
    cL = cluster1.centerline + cluster2.centerline
    cLr = PolyLine(cL[::-1])
    wayConnectors = dict()
    clustConnectors = dict()
    # find width and line of outgoing way
    outWayID = [w.sectionId for w in cls.sectionNetwork.iterOutSegments(node) if w.sectionId not in blockedWayIDs \
                           and cls.waySections[w.sectionId].isValid][0]
    outLine, outWidth = widthAndLineFromID(cls,node,outWayID)

    # On the right side of the cluster, for an intersection area in counter-clockwise
    # order, the clusters are in correct order.

    # Find intersections <p1> and <p3> of way borders left and right of the out-way
    # with the right border of cluster 2.
    reverseCenterline = PolyLine(cluster1.centerline[::-1])
    outW = max(cluster1.outWR(),cluster2.outWR())
    # p1, valid = offsetPolylineIntersection(reverseCenterline,outLine,outW,outWidth/2.)
    p1, valid = offsetPolylineIntersection(cLr,outLine,outW,outWidth/2.)
    # if valid!='valid':
    #     plotPureNetwork(cls.sectionNetwork)
    #     cluster1.centerline.plot('g:',3)
    #     cluster2.centerline.plot('r',3)
    #     plt.plot(p1[0],p1[1],'ro')
    #     outLine.plot('c:',3)
    #     plt.title('createRightIntersection, p1')
    #     plotEnd()

    assert valid=='valid'
    # p3, valid = offsetPolylineIntersection(outLine,cluster2.centerline,outWidth/2.,outW)
    p3, valid = offsetPolylineIntersection(outLine,cL,outWidth/2.,outW)
    # if valid!='valid':
    #     plotPureNetwork(cls.sectionNetwork)
    #     cluster1.centerline.plot('g:',3)
    #     cluster2.centerline.plot('r',3)
    #     plt.plot(p3[0],p3[1],'ro')
    #     outLine.plot('c:',3)
    #     plt.title('createRightIntersection, p3')
    #     plotEnd()

    assert valid=='valid'

    # Project these onto cluster centerlines to get initial trim values.
    _,t = cluster1.centerline.orthoProj(p1)
    cluster1.trimT = min(cluster1.trimT, t)
    _,t = cluster2.centerline.orthoProj(p3)
    cluster2.trimS = max(cluster2.trimS, t)

    # Project these onto the centerline of the out-way and create intermediate
    # polygon point <p2>.
    _,tP1 = outLine.orthoProj(p1)
    _,tP3 = outLine.orthoProj(p3)
    section = cls.waySections[outWayID]
    fwd = section.originalSection.s == node
    if tP3 > tP1:
        p2 = outLine.offsetPointAt(tP3,-outWidth/2.)
        tP = tP3
        wayConnectors[section.id] = ( 0, 'S' if fwd else 'E')
    else:
        p2 = outLine.offsetPointAt(tP1,outWidth/2.)
        tP = tP1
        wayConnectors[section.id] = ( 1, 'S' if fwd else 'E')
    if fwd:
        section.trimS = max(section.trimS,tP)
    else:
        section.trimT = min(section.trimT,len(outLine)-1-tP)

    # Start area in counter-clockwise order, <p1> is on cluster1
    area = [p1,p2,p3] if p1!=p2 and p2!=p3 else [p1,p3]

    # cluster1.centerline.plot('g:',3)
    # cluster2.centerline.plot('r:',3)
    # plt.plot(p1[0],p1[1],'ro')
    # plt.text(p1[0],p1[1],'p1')
    # plt.plot(p2[0],p2[1],'ro')
    # plt.text(p2[0],p2[1],'p2')
    # plt.plot(p3[0],p3[1],'ro')
    # plt.text(p3[0],p3[1],'p3')
    # outLine.plot('c',4)

    transWidth1, transWidth2 = computeTransitionWidths(cluster1,cluster2)

    # Construct area through cluster2 (counter-clockwise order). On the right side,
    # we start with the second cluster.
    if transWidth2: # If we need a width transition
        transitionLength = transWidth2/transitionSlope
        dTrans = transitionLength
        tTrans = cluster2.centerline.d2t(dTrans)
        cluster2.trimS = max(cluster2.trimS,tTrans)
        # Two additional points due to transition, <p4> on the right and <p5> on 
        # the left of the cluster.
        p4 = cluster2.centerline.offsetPointAt(cluster2.trimS,-cluster2.outWR())
        area.append(p4)
        clustConnectors[cluster2.id] = ( len(area)-1, 'S', -1 if cls.isectShape == 'common' else 1)
        if cls.isectShape == 'common':
            p5 = cluster2.centerline.offsetPointAt(cluster2.trimS,cluster2.outWL())
        elif cls.isectShape == 'separated':
            p5 = cluster2.centerline.offsetPointAt(cluster2.trimS,-cluster2.inWR())
        area.append(p5)
    else:
        # No transistion, we construct directly <p5> on the left of the cluster.
        clustConnectors[cluster1.id] = ( len(area)-1, 'S', -1 if cls.isectShape == 'common' else 1)
        _,tP = cluster2.centerline.orthoProj(p3)
        cluster2.trimS = max(cluster2.trimS,tP)
        if cls.isectShape == 'common':
            p5 = cluster2.centerline.offsetPointAt(tP,cluster2.outWL())
        elif cls.isectShape == 'separated':
            p5 = cluster2.centerline.offsetPointAt(tP,-cluster2.inWR())
        area.append(p5)

    # Construct area through cluster1 (counter-clockwise order)
    if transWidth1:
        transitionLength = transWidth1/transitionSlope
        dTrans = cluster1.centerline.length() - transitionLength
        tTrans = cluster1.centerline.d2t(dTrans)
        cluster1.trimT = min(cluster1.trimT,tTrans)
        # Two additional points due to transition, <p6> on the left and <p7> on 
        # the right of the cluster.
        if cls.isectShape == 'common':
            p6 = cluster1.centerline.offsetPointAt(cluster1.trimT,cluster1.outWL())
        elif cls.isectShape == 'separated':
            p6 = cluster1.centerline.offsetPointAt(cluster1.trimT,-cluster1.inWR())
        area.append(p6)
        clustConnectors[cluster1.id] = ( len(area)-1, 'E', -1 if cls.isectShape == 'common' else 1)
        p7 = cluster1.centerline.offsetPointAt(cluster1.trimT,-cluster1.outWR())
        area.append(p7)
    else:
        # No transistion, we construct directly <p7> on the left of the cluster.
        _,tP = cluster1.centerline.orthoProj(p1)
        cluster1.trimT = min(cluster1.trimT,tP)
        if cls.isectShape == 'common':
            p7 = cluster1.centerline.offsetPointAt(cluster1.trimT,cluster1.outWL())
        elif cls.isectShape == 'separated':
            p7 = cluster1.centerline.offsetPointAt(cluster1.trimT,-cluster1.inWR())
        area.append(p7)
        clustConnectors[cluster1.id] = ( len(area)-1, 'E', -1 if cls.isectShape == 'common' else 1)
    return area, clustConnectors, wayConnectors

def createClippedEndArea(cls,longCluster,endsubCluster):
    def findUnclippedOutWay(unClippedNode,unclippedIDs):
        for net_section in cls.sectionNetwork.iterOutSegments(unClippedNode):
            if net_section.category != 'scene_border':
                if net_section.sectionId != unclippedIDs:
                    section = cls.waySections[net_section.sectionId]
                    if section.originalSection.s == unClippedNode:
                        return section,True   # forward = True
                    else:
                        return section, False # forward = False
        return None, None

    def findPolyline(section, forward):
        # Trim centerline of way to the clipped position (end of cluster)
        _,t = section.polyline.orthoProj(endsubCluster.centerline[-1])
        if forward: 
            section.trimS = max(section.trimS,t)
            polyline = section.polyline.trimmed(section.trimS,section.trimT)
            dims = (section.leftWidth,section.rightWidth)
        else:
            section.trimT = min(section.trimT,t)
            polyline = PolyLine(section.polyline.trimmed(section.trimS,section.trimT)[::-1])
            dims = (section.rightWidth,section.leftWidth)
        return polyline,dims

    type = endsubCluster.endSplit.type
    doClip = [False]*len(endsubCluster.endSplit.currWIds)
    if type == 'clipL':
        doClip[0] = True
    if type == 'clipR':
        doClip[-1] = True
    if type == 'clipB':
        doClip = [True]*len(endsubCluster.endSplit.currWIds)

    # Find outgoing polylines
    polylines = []
    dims = []
    fwds = []
    sections = []
    for indx,wID in enumerate(endsubCluster.endSplit.currWIds):
        section = cls.waySections[wID[-1]]
        if doClip[indx]:    # clipped section
            forward = section.polyline[0] in longCluster.polylines[indx]
            polyline,dim = findPolyline(section,forward)
            section.isClipped = True
        else:               # unclipped section
            node = section.polyline[0]
            section, forward = findUnclippedOutWay(node,wID[-1])
            if not section:
                continue
            polyline,dim = findPolyline(section,forward)
        section.isValid = True
        polylines.append(polyline)
        dims.append(dim)
        fwds.append(forward)
        sections.append(section)

    # The area will be constructed along a line perpendicular to the centerline of the cluster.
    # First, we need a single segment polyline perpendicular to the centerline of the cluster,
    # either left-to-right or reversed.
    pL = endsubCluster.centerline.offsetPointAt(endsubCluster.trimT,endsubCluster.endWidth)
    pR = endsubCluster.centerline.offsetPointAt(endsubCluster.trimT,-endsubCluster.endWidth)
    perpPoly2right = PolyLine([pL,pR])
    perpPoly2left  = PolyLine([pR,pL])

    # We start at the right to get an area in counter-clockwise order.
    wIDs = [wID[-1] for wID in endsubCluster.endSplit.currWIds]
    area = []
    wayConnectors = dict()
    clustConnectors = dict()
    for line,dim,fwd,wID,section in zip(polylines[::-1],dims[::-1],fwds[::-1],wIDs[::-1],sections[::-1]):
        # Find intersections <p1> and <p3> of line borders given in dim with the
        # perpendicular line.
        width = (dim[0]+dim[1])/2.
        p1, valid = offsetPolylineIntersection(perpPoly2right,line,7.,width)
        assert valid=='valid'
        p3, valid = offsetPolylineIntersection(line,perpPoly2left,width,7.)
        assert valid=='valid'

        # Project these onto the centerline of the out-way line and create intermediate
        # polygon point <p2>.
        _,tP1 = line.orthoProj(p1)
        _,tP3 = line.orthoProj(p3)
        if tP3 > tP1:
            p2 = line.offsetPointAt(tP3,-width)
            _,tS = section.polyline.orthoProj(p3)
            wayConnectors[section.id] = ( len(area)+1, 'S' if fwd else 'E')
        else:
            p2 = line.offsetPointAt(tP1,width)
            _,tS = section.polyline.orthoProj(p1)
            wayConnectors[section.id] = ( len(area), 'S' if fwd else 'E')
        if fwd:
            section.trimS = max(section.trimS,tS)
        else:
            section.trimT = min(section.trimT,tS)
        area.extend([p1,p2,p3])

    pL = endsubCluster.centerline.offsetPointAt(endsubCluster.trimT,endsubCluster.outWL())
    pR = endsubCluster.centerline.offsetPointAt(endsubCluster.trimT,-endsubCluster.outWR())
    area.extend([pL,pR])
    clustConnectors[endsubCluster.id] = ( len(area)-2, 'E', -1)

    return area, clustConnectors, wayConnectors

def createShortClusterIntersection(cls, cluster1, shortCluster, cluster2,blockedWayIDs):
    # Replace the short cluster by an intersection area.
    shortCluster.valid = False

    # Create an instance of <IntersectionCluster>, which will form the
    # cluster area and its connectors.
    isectCluster = IntersectionCluster()

    # Collect all intersection nodes of the way-sections at the cluster.
    nodes = set()
    nodes.update( [cluster1.endSplit.posL,cluster1.endSplit.posR] )
    nodes.update( [cluster2.startSplit.posL,cluster2.startSplit.posR] )

    # Insert centerlines and widths of clusters to <IntersectionCluster>. Keep a map
    # <ID2Object> for the ID in <IntersectionCluster> to the inserted object.
    ID2Object = dict()

    # End of cluster before short cluster
    Id = isectCluster.addWay(PolyLine(cluster1.centerline[::-1]),cluster1.outWR(),cluster1.outWL())
    ID2Object[Id] = (cluster1.centerline[-1], cluster1, 'end', cluster1.longCluster.id)
    # Start of cluster after  short cluster
    Id = isectCluster.addWay(cluster2.centerline,cluster2.outWL(),cluster2.outWR())
    ID2Object[Id] = (cluster2.centerline[0], cluster2, 'start', cluster2.longCluster.id)

    # Find all outgoing way-sections that are connected to cluster nodes, but exlude
    # those that belong to the clusters, using <wayIDs>. Insert them together with their
    # widths into <IntersectionCluster>. Extend the map <ID2Object> for the ID in
    # <IntersectionCluster> to the inserted object.
    for node in nodes:
        for net_section in cls.sectionNetwork.iterOutSegments(node):
            if net_section.category != 'scene_border':
                if net_section.sectionId not in blockedWayIDs and cls.waySections[net_section.sectionId].isValid:
                    section = cls.waySections[net_section.sectionId]
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
        # # Create the intersection area and their connectors5 
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

    # plotPolygon(area,False,'r','r',2)
    return area, wayConnectors, clustConnectors

