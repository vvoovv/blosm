from itertools import tee

from defs.way_cluster_params import transitionLimit,transitionSlope
from way.way_properties import estimateWayWidth
from lib.CompGeom.offset_intersection import offsetPolylineIntersection
from lib.CompGeom.PolyLine import PolyLine


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

class LongClusterWay():
    def __init__(self,cls):
        self.cls = cls
        self.leftSectionIDs = None
        self.rightSectionIDs = None
        self.intersectionsAll = None
        self.left = None
        self.right = None
        self.centerline = None

        self.clusterWays = []
        
    def split(self):
        debug = []
        # Find intersections along cluster, given as line parameter t of centerline.
        # Sort them so that they are aligned along the cluster centerline.
        leftWIndx = 0
        rightWIndx = 0
        intermediateIsects = [(0.,'both',self.left[0],self.right[0],self.leftSectionIDs[leftWIndx], self.rightSectionIDs[rightWIndx])]
        for intersectionsThis in self.intersectionsAll:
            for isect in intersectionsThis:
                type = 'left' if isect in self.left else 'right'
                p,t = self.centerline.orthoProj(isect)
                if type == 'left':
                    rightP,_ = self.right.intersectWithLine(isect,p)
                    leftWIndx += 1
                    intermediateIsects.append((t,'left',isect,rightP,self.leftSectionIDs[leftWIndx], self.rightSectionIDs[rightWIndx]))
                else:
                    leftP,_ = self.left.intersectWithLine(isect,p)
                    rightWIndx += 1
                    intermediateIsects.append((t,'right',leftP,isect,self.leftSectionIDs[leftWIndx], self.rightSectionIDs[rightWIndx]))
        intermediateIsects.append((len(self.centerline),'both',self.left[-1],self.right[-1],None,None))
        intermediateIsects.sort(key=lambda x: x[0])

        # Create clusters, split them when there are intermdiate intersections
        for c0,c1 in pairs(intermediateIsects):
            clusterWay = ClusterWay(self.centerline.trimmed(c0[0],c1[0]))
            clusterWay.leftWayID = c0[4]
            clusterWay.rightWayID = c0[5]
            clusterWay.startPoints = c0[1:]
            clusterWay.endPoints = c1[1:]
            clusterWay.startWidth = widthOrthoCenterline(clusterWay.centerline,self.left,self.right,clusterWay.startPoints[1])
            clusterWay.endWidth = widthOrthoCenterline(clusterWay.centerline,self.left,self.right,clusterWay.endPoints[1])

            # prepare data eventually required for width transition
            widthDifference = abs(clusterWay.startWidth - clusterWay.endWidth)
            if widthDifference < transitionLimit:
                clusterWay.clusterWidth = (clusterWay.startWidth + clusterWay.endWidth)/2.
                clusterWay.transitionWidth = 0.
                clusterWay.transitionPos = None
            else:
                clusterWay.clusterWidth = min(clusterWay.startWidth, clusterWay.endWidth)
                clusterWay.transitionWidth = widthDifference
                clusterWay.transitionPos = 'start' if clusterWay.startWidth > clusterWay.endWidth else 'end'

            # Get widths of left and right ways
            categoryL = self.cls.waySections[clusterWay.leftWayID].originalSection.category
            tagsL = self.cls.waySections[clusterWay.leftWayID].originalSection.tags
            clusterWay.leftWayWidth = estimateWayWidth(categoryL,tagsL)
            categoryR = self.cls.waySections[clusterWay.rightWayID].originalSection.category
            tagsR = self.cls.waySections[clusterWay.rightWayID].originalSection.tags
            clusterWay.rightWayWidth = estimateWayWidth(categoryR,tagsR)
            self.clusterWays.append(clusterWay)

class ClusterWay():
    ID = 0
    def __init__(self,centerline):
        self.id = ClusterWay.ID
        ClusterWay.ID += 1
        self.centerline = centerline
        self.clusterWidth = None
        self.transitionWidth = None
        self.transitionPos = None
        self.trimS = 0.                      # trim factor for start
        self.trimT = len(self.centerline)-1  # trim factor for target
        self.leftWayID = None
        self.leftWayWidth = None
        self.rightWayID = None
        self.rightWayWidth = None
        self.startPoints = None
        self.endPoints = None

        self.startConnected = True
        self.endConnected = True

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
        return self.clusterWidth + (self.leftWayWidth+self.rightWayWidth)/2.
    def outWL(self):
        return (self.clusterWidth + self.leftWayWidth)/2.
    def outWR(self):
        return (self.clusterWidth + self.rightWayWidth)/2.
    def inWL(self):
        return (self.clusterWidth - self.leftWayWidth)/2.
    def inWR(self):
        return (self.clusterWidth - self.rightWayWidth)/2.

    # def splitBecauseWidth(self):
    #     clusters = []
    #     widthDiff = self.endWidth - self.startWidth
    #     if abs(widthDiff) > widthThresh:
    #         nrOfSegs = ceil(abs(widthDiff) / widthThresh)
    #         totalLength = self.centerline.length()
    #         lengthStep = totalLength/nrOfSegs
    #         widthStep = widthDiff/(nrOfSegs-1)

    #         lastEndPoints = self.startPoints
    #         last_t = 0.
    #         for segNr in range(nrOfSegs-1):
    #             d = lengthStep * (segNr+1)
    #             w = self.startWidth + widthStep * segNr
    #             t = self.centerline.d2t(d)
    #             pL = self.centerline.offsetPointAt(t,abs(w/2))
    #             pR = self.centerline.offsetPointAt(t,-abs(w/2))
    #             end = ('width',pL,pR,self.leftWayID,self.rightWayID)

    #             segCluster = ClusterWay(self.centerline.trimmed(last_t,t))
    #             segCluster.startPoints = lastEndPoints
    #             segCluster.endPoints = end
    #             segCluster.leftWayID = self.leftWayID
    #             segCluster.rightWayID = self.rightWayID
    #             segCluster.startWidth = w

    #             clusters.append(segCluster)

    #             last_t = t
    #             lastEndPoints = segCluster.endPoints

    #         segCluster = ClusterWay(self.centerline.trimmed(last_t,len(self.centerline)))
    #         segCluster.startPoints = lastEndPoints
    #         segCluster.endPoints = self.endPoints
    #         segCluster.leftWayID = self.leftWayID
    #         segCluster.rightWayID = self.rightWayID
    #         segCluster.startWidth = self.endWidth
    #         clusters.append(segCluster)


    #         return clusters
    #     else:
    #         return [self]

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

    return transWidth1, transWidth2

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

    # Construct area through cluster1 (counter-clockwise order)
    transitionLength = transWidth1/transitionSlope
    dTrans = cluster1.centerline.length() - transitionLength
    tTrans = cluster1.centerline.d2t(dTrans)
    cluster1.trimT = min(cluster1.trimT,tTrans)
    # Two additional points due to transition
    p = cluster1.centerline.offsetPointAt(cluster1.trimT,-cluster1.outWR())
    area.append(p)
    if cls.isectShape == 'common':
        p = cluster1.centerline.offsetPointAt(cluster1.trimT,cluster1.outWL())
    elif cls.isectShape == 'separated':
        p = cluster1.centerline.offsetPointAt(cluster1.trimT,-cluster1.inWR())
    area.append(p)

    # Construct area through cluster2 (counter-clockwise order)
    transitionLength = transWidth2/transitionSlope
    dTrans = transitionLength
    tTrans = cluster2.centerline.d2t(dTrans)
    cluster2.trimS = max(cluster2.trimS,tTrans)
    # Two additional points due to transition
    if cls.isectShape == 'common':
        p = cluster2.centerline.offsetPointAt(cluster2.trimS,cluster2.outWL())
    elif cls.isectShape == 'separated':
        p = cluster2.centerline.offsetPointAt(cluster2.trimS,-cluster2.inWR())
    area.append(p)
    p = cluster2.centerline.offsetPointAt(cluster2.trimS,-cluster2.outWR())
    area.append(p)

    clustConnectors = dict()
    clustConnectors[cluster1.id] = ( 1, 'E', -1 if cls.isectShape == 'common' else 1)
    clustConnectors[cluster2.id] = ( 3, 'S', -1 if cls.isectShape == 'common' else 1)
    return area, clustConnectors

def createLeftIntersection(cls, cluster1, cluster2, node):
    wayConnectors = dict()
    clustConnectors = dict()
    # find width of outgoing way
    id1, id2 = cluster1.leftWayID, cluster2.leftWayID
    outWayID = [w.sectionId for w in cls.sectionNetwork.iterOutSegments(node) if w.sectionId not in (id1,id2)][0]

    # print(id1,id2,outWayID)
    # outways = [w for w in cls.sectionNetwork.iterOutSegments(node)]
    # for w in outways:
    #     p = w.path[-1]
    #     plt.text(p[0],p[1],str(w.sectionId))
    #     plotLine(w.path,False,'c',2)
    # cls.waySections[id1].polyline.plot('m',4)
    # # cls.waySections[id2].polyline.plot('k')
    # cls.waySections[outWayID].polyline.plot('r',3)
    # plotEnd()

    outLine, outWidth = widthAndLineFromID(cls,node,outWayID)

    # Find intersections <p1> and <p3> of way borders left and right of the out-way
    outW = max(cluster1.outWL(),cluster2.outWL())
    p1, valid = offsetPolylineIntersection(cluster2.centerline,outLine,outW,outWidth/2.)
    assert valid=='valid'
    reverseCenterline = PolyLine(cluster1.centerline[::-1])
    p3, valid = offsetPolylineIntersection(outLine,reverseCenterline,outWidth/2.,outW)
    assert valid=='valid'

    # Project these onto cluster centerlines to get initial trim values
    _,t = cluster1.centerline.orthoProj(p3)
    cluster1.trimT = min(cluster1.trimT, t)
    _,t = cluster2.centerline.orthoProj(p1)
    cluster2.trimS = max(cluster2.trimS, t)

    # Project these onto the centerline of out-way and create intermediate polygon point <p1>
    _,tP1 = outLine.orthoProj(p1)
    _,tP3 = outLine.orthoProj(p3)
    section = cls.waySections[outWayID]
    fwd = section.originalSection.s == node
    if tP3 > tP1:
        p2 = outLine.offsetPointAt(tP3,-outWidth/2.)
        tP = tP3
        wayConnectors[outWayID] = ( 1, 'S' if fwd else 'E')
    else:
        p2 = outLine.offsetPointAt(tP1,outWidth/2.)
        tP = tP1
        wayConnectors[outWayID] = ( 0, 'S' if fwd else 'E')
    if fwd:
        section.trimS = tP
    else:
        section.trimT = tP

    # start area in counter-clockwise order, continue at end of cluster1
    area = [p1,p2,p3]

    transWidth1, transWidth2 = computeTransitionWidths(cluster1,cluster2)

    # Construct area through cluster1 (counter-clockwise order)
    if transWidth1:
        transitionLength = transWidth1/transitionSlope
        dTrans = cluster1.centerline.length() - transitionLength
        tTrans = cluster1.centerline.d2t(dTrans)
        cluster1.trimT = min(cluster1.trimT,tTrans)
        # Two additional points due to transition
        p = cluster1.centerline.offsetPointAt(cluster1.trimT,cluster1.outWL())
        area.append(p)
        clustConnectors[cluster1.id] = ( len(area)-1, 'E', -1 if cls.isectShape == 'common' else 0)
        if cls.isectShape == 'common':
            p = cluster1.centerline.offsetPointAt(cluster1.trimT,-cluster1.outWR())
        elif cls.isectShape == 'separated':
            p = cluster1.centerline.offsetPointAt(cluster1.trimT,cluster1.inWL())
        area.append(p)
    else:
        clustConnectors[cluster1.id] = ( len(area)-1, 'E', -1 if cls.isectShape == 'common' else 0)
        _,tP = cluster1.centerline.orthoProj(p3)
        cluster1.trimT = min(cluster1.trimT,tP)
        if cls.isectShape == 'common':
            p = cluster1.centerline.offsetPointAt(tP,-cluster1.outWR())
        elif cls.isectShape == 'separated':
            p = cluster1.centerline.offsetPointAt(tP,cluster1.inWL())
        area.append(p)

    # Construct area through cluster2 (counter-clockwise order)
    if transWidth2:
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
        clustConnectors[cluster2.id] = ( len(area)-1, 'S', -1 if cls.isectShape == 'common' else 0)
        p = cluster2.centerline.offsetPointAt(cluster2.trimS,cluster2.outWL())
        area.append(p)
    else:
        _,tP = cluster2.centerline.orthoProj(p1)
        cluster2.trimS = max(cluster2.trimS,tP)
        if cls.isectShape == 'common':
            p = cluster2.centerline.offsetPointAt(cluster2.trimS,-cluster2.outWR())
        elif cls.isectShape == 'separated':
            p = cluster2.centerline.offsetPointAt(cluster2.trimS,cluster2.inWL())
        area.append(p)
        clustConnectors[cluster2.id] = ( len(area)-1, 'S', -1 if cls.isectShape == 'common' else 0)
 
    return area, clustConnectors, wayConnectors

def createRightIntersection(cls, cluster1, cluster2, node):
    wayConnectors = dict()
    clustConnectors = dict()
    # find width of outgoing way
    id1, id2 = cluster1.rightWayID, cluster2.rightWayID
    outWayID = [w.sectionId for w in cls.sectionNetwork.iterOutSegments(node) if w.sectionId not in (id1,id2)][0]

    # print(id1,id2,outWayID)
    # plt.subplot(1,2,1)
    # outways = [w for w in cls.sectionNetwork.iterOutSegments(node)]
    # for w in outways:
    #     p = w.path[-1]
    #     plt.text(p[0],p[1],str(w.sectionId))
    #     plotLine(w.path,False,'c',2)
    # plt.gca().axis('equal')
    # plt.subplot(1,2,2)
    # cls.waySections[id1].polyline.plot('c')
    # cls.waySections[id2].polyline.plot('c')
    # cls.waySections[outWayID].polyline.plot('r',3)
    # cluster2.centerline.plot('k',1,True)
    # plotEnd()

    outLine, outWidth = widthAndLineFromID(cls,node,outWayID)

    # Find intersections <p1> and <p3> of way borders left and right of the out-way
    reverseCenterline = PolyLine(cluster1.centerline[::-1])
    outW = max(cluster1.outWL(),cluster2.outWL())
    p1, valid = offsetPolylineIntersection(reverseCenterline,outLine,outWidth/2.,outW)
    assert valid=='valid'
    p3, valid = offsetPolylineIntersection(outLine,cluster2.centerline,outW,outWidth/2.)
    assert valid=='valid'

    # Project these onto cluster centerlines to get initial trim values
    _,t = cluster2.centerline.orthoProj(p3)
    cluster2.trimT = min(cluster2.trimS, t)
    _,t = cluster1.centerline.orthoProj(p1)
    cluster1.trimS = max(cluster1.trimT, t)

    # Project these onto the centerline of out-way and create intermediate polygon point <p1>
    _,tP1 = outLine.orthoProj(p1)
    _,tP3 = outLine.orthoProj(p3)
    section = cls.waySections[outWayID]
    fwd = section.originalSection.s == node
    if tP3 > tP1:
        p2 = outLine.offsetPointAt(tP3,outWidth/2.)
        tP = tP3
        wayConnectors[outWayID] = ( 1, 'S' if fwd else 'E')
    else:
        p2 = outLine.offsetPointAt(tP1,-outWidth/2.)
        tP = tP1
        wayConnectors[outWayID] = ( 0, 'S' if fwd else 'E')
    if fwd:
        section.trimS = tP
    else:
        section.trimT = tP

    # start area in counter-clockwise order, end is on clutser1
    area = [p1,p2,p3]

    transWidth1, transWidth2 = computeTransitionWidths(cluster1,cluster2)

    # Construct area through cluster2 (counter-clockwise order)
    if transWidth2:
        transitionLength = transWidth2/transitionSlope
        dTrans = cluster2.centerline.length() - transitionLength
        tTrans = cluster2.centerline.d2t(dTrans)
        cluster2.trimS = max(cluster2.trimS,tTrans)
        # Two additional points due to transition
        p = cluster2.centerline.offsetPointAt(cluster2.trimS,cluster1.outWL())
        area.append(p)
        if cls.isectShape == 'common':
            p = cluster2.centerline.offsetPointAt(cluster2.trimS,-cluster2.outWR())
        elif cls.isectShape == 'separated':
            p = cluster2.centerline.offsetPointAt(cluster2.trimS,cluster1.inWL())
        area.append(p)
        clustConnectors[cluster2.id] = ( len(area)-1, 'E', -1 if cls.isectShape == 'common' else 1)
    else:
        clustConnectors[cluster1.id] = ( len(area)-1, 'E', -1 if cls.isectShape == 'common' else 1)
        _,tP = cluster2.centerline.orthoProj(p3)
        cluster1.trimS = min(cluster1.trimS,tP)
        if cls.isectShape == 'common':
            p = cluster2.centerline.offsetPointAt(tP,-cluster2.outWR())
        elif cls.isectShape == 'separated':
            p = cluster2.centerline.offsetPointAt(tP,cluster2.inWL())
        area.append(p)

    # Construct area through cluster1 (counter-clockwise order)
    if transWidth1:
        transitionLength = transWidth1/transitionSlope
        dTrans = transitionLength
        tTrans = cluster1.centerline.d2t(dTrans)
        cluster1.trimT = max(cluster1.trimT,tTrans)
        # Two additional points due to transition
        if cls.isectShape == 'common':
            p = cluster1.centerline.offsetPointAt(cluster1.trimT,-cluster1.outWR())
        elif cls.isectShape == 'separated':
            p = cluster1.centerline.offsetPointAt(cluster1.trimT,cluster1.inWL())
        area.append(p)
        p = cluster1.centerline.offsetPointAt(cluster1.trimT,cluster1.outWL())
        area.append(p)
        clustConnectors[cluster1.id] = ( len(area)-1, 'S', -1 if cls.isectShape == 'common' else 1)
    else:
        clustConnectors[cluster1.id] = ( len(area)-1, 'S', -1 if cls.isectShape == 'common' else 1)
        _,tP = cluster1.centerline.orthoProj(p1)
        cluster1.trimS = max(cluster1.trimT,tP)
        if cls.isectShape == 'common':
            p = cluster1.centerline.offsetPointAt(cluster1.trimT,-cluster1.outWR())
        elif cls.isectShape == 'separated':
            p = cluster1.centerline.offsetPointAt(cluster1.trimT,cluster1.inWL())
        area.append(p)

    return area, clustConnectors, wayConnectors
