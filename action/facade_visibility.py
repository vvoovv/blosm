from math import sqrt
import numpy as np
from bisect import bisect_left
from operator import itemgetter
from defs.facade_classification import searchRange, FacadeClass, WayLevel, CrossedFacades, FrontFacadeSight



class PriorityQueue():
    def __init__(self):
        self.keys = []
        self.queue = []
        self.ids = []
    
    def push(self, key, item):
        i = bisect_left(self.keys, key) # Determine where to insert item.
        self.keys.insert(i, key)        # Insert key of item to keys list.
        self.queue.insert(i, item)      # Insert the item itself in the corresponding place.
        self.ids.insert(i, id(item))
    
    def pop(self):
        self.keys.pop(0)
        self.ids.pop(0)
        return self.queue.pop(0)
    
    def remove(self, item):
        itemIndex = self.ids.index(id(item))
        del self.ids[itemIndex]
        del self.queue[itemIndex]
        del self.keys[itemIndex]
    
    def empty(self):
        return not self.queue
    
    def cleanup(self):
        self.keys.clear()
        self.queue.clear()
        self.ids.clear()


class FacadeVisibility:
    
    def __init__(self):
        self.app = None
        self.kdTree = None
        self.bldgVerts = None
        self.searchWidthMargin = searchRange[0]
        self.searchHeight = searchRange[1]
        self.searchHeight_2 = searchRange[1]*searchRange[1]
        self.vertIndexToBldgIndex = []
        self.priorityQueue = PriorityQueue()
        self.posEvents = []
        self.negEvents = []
    
    def do(self, manager):
        # check if have a way manager
        if not self.app.managersById["ways"]:
            return
        
        buildings = manager.buildings
        
        # the total number of vertices
        totalNumVerts = sum(building.polygon.numEdges for building in buildings if building.polygon)
        # create mapping between the index of the vertex and index of the building in <buildings>
        self.vertIndexToBldgIndex.extend(
            bldgIndex for bldgIndex, building in enumerate(buildings) if building.polygon for _ in range(building.polygon.numEdges)
        )
        
        # allocate the memory for an empty numpy array
        bldgVerts = np.zeros((totalNumVerts, 2))
        # fill in <self.bldgVerts>
        index = 0
        for building in buildings:
            if building.polygon:
                # store the index of the first vertex of <building> in <self.bldgVerts> 
                building.auxIndex = index
                for vert in building.polygon.verts:
                    bldgVerts[index] = vert
                    index += 1
        self.bldgVerts = bldgVerts
        
        self.createKdTree(buildings, totalNumVerts)
        
        self.calculateFacadeVisibility(manager)
    
    def cleanup(self):
        self.kdTree = None
        self.bldgVerts = None
        self.vertIndexToBldgIndex.clear()

    def calculateFacadeVisibility(self, manager):
        buildings = manager.buildings
        
        posEvents = self.posEvents
        negEvents = self.negEvents
        
        for way in self.app.managersById["ways"].getFacadeVisibilityWays():
            
            for segment in way.segments:
                segmentCenter, segmentUnitVector, segmentLength = \
                (segment.v1 + segment.v2)/2., segment.unitVector, segment.length
                
                posEvents.clear()
                negEvents.clear()
                
                halfSegmentWidth = segmentLength/2.
                searchWidth = halfSegmentWidth + self.searchWidthMargin
                # Get all building vertices for the buildings whose vertices are returned
                # by the query to the KD tree
                queryBldgIndices = set(
                    self.makeKdQuery(segmentCenter, sqrt(searchWidth*searchWidth + self.searchHeight_2))
                )
                queryBldgVertIndices = tuple(
                    vertIndex for bldgIndex in queryBldgIndices for vertIndex in range(buildings[bldgIndex].auxIndex, buildings[bldgIndex].auxIndex + buildings[bldgIndex].polygon.numEdges)
                )
                # slice <self.bldgVerts> using <queryBldgVertIndices>
                queryBldgVerts = self.bldgVerts[queryBldgVertIndices,:]
                
                # transform <queryBldgVerts> to the system of reference of <way>
                matrix = np.array(( (segmentUnitVector[0], -segmentUnitVector[1]), (segmentUnitVector[1], segmentUnitVector[0]) ))
                
                # shift origin to the segment center
                queryBldgVerts -= segmentCenter
                # rotate <queryBldgVerts>, output back to <queryBldgVerts>
                np.matmul(queryBldgVerts, matrix, out=queryBldgVerts)
                
                #
                # fill <posEvents> and <negEvents>
                #
                
                # index in <queryBldgVerts>
                firstVertIndex = 0
                for bldgIndex in queryBldgIndices:
                    building = buildings[bldgIndex]
                    building.resetVisInfoTmp()
                    building.resetCrossedEdges()
                    
                    for edge, edgeVert1, edgeVert2 in building.polygon.edgeInfo(queryBldgVerts, firstVertIndex, skipShared=False):
                        # check if there are edges crossed by this way-segment (the vertices have different signs of y-coord)
                        if np.diff( (np.sign(edgeVert1[1]),np.sign(edgeVert2[1])) ):
                            # find x-coord of intersection with way-segment axis (the x-axis in this coordinate system).
                            # divide its value by the half of the segment length to get a relative number.
                            # for absolute values between 0 and 1, the intersection is in the way-segment, else out of it.
                            dx = edgeVert2[0]- edgeVert1[0]
                            dy = edgeVert2[1]- edgeVert1[1]
                            x0 = edgeVert1[0]
                            y0 = edgeVert1[1]
                            intsectX = (x0+abs(y0/dy)*dx ) / halfSegmentWidth if dy else 0.
                            # store result in building for later analysis
                            building.addCrossedEdge(edge, intsectX)

                    intersections = building.crossedEdges
                    # if there are intersections, fill a dummy line in the interior of the building
                    # between the intersection. The edge index is negative, so that it can be 
                    # excluded in lines marked with "updateAuxVisibilityAdd".
                    if intersections:
                        # create list of intersections and remove duplicates 
                        intsectList = list(dict.fromkeys( x[1] for x in intersections ))
                        intsectList.sort()
                        for i in range(0, len(intsectList), 2):
                            x1 = intsectList[i] * halfSegmentWidth
                            x2 = intsectList[i+1] * halfSegmentWidth
                            edge = None  # None to mark it as a dummy edge
                            posEvents.append(
                                (edge, None, x1, np.array((x1,0.)), np.array((x2,0.)))
                            )
                            # Keep track of the related "edge starts" event,
                            # that's why <posEvents[-1]>
                            posEvents.append(
                                (edge, posEvents[-1], x2, np.array((x1,0.)), np.array((x2,0.)))
                            )
                            negEvents.append(
                                (edge, None, x1, np.array((x1,0.)), np.array((x2,0.)))
                            )
                            # Keep track of the related "edge starts" event,
                            # that's why <posEvents[-1]>
                            negEvents.append(
                                (edge, negEvents[-1], x2, np.array((x1,0.)), np.array((x2,0.)))
                            )

                    for edge, edgeVert1, edgeVert2 in building.polygon.edgeInfo(queryBldgVerts, firstVertIndex, skipShared=True):
                        if edgeVert1[0] > edgeVert2[0]:
                            # because the algorithm scans with increasing x, the first vertice has to get smaller x
                            edgeVert1, edgeVert2 = edgeVert2, edgeVert1
                        
                        if edgeVert1[1] > 0. or edgeVert2[1] > 0.:
                            posEvents.append(
                                (edge, None, edgeVert1[0], edgeVert1, edgeVert2)
                            )
                            # Keep track of the related "edge starts" event,
                            # that's why <posEvents[-1]>
                            posEvents.append(
                                (edge, posEvents[-1], edgeVert2[0], edgeVert1, edgeVert2)
                            )
                        if edgeVert1[1] < 0. or edgeVert2[1] < 0.:
                            negEvents.append(
                                (edge, None, edgeVert1[0], edgeVert1, edgeVert2)
                            )
                            # Keep track of the related "edge starts" event,
                            # that's why <negEvents[-1]>
                            negEvents.append(
                                (edge, negEvents[-1], edgeVert2[0], edgeVert1, edgeVert2)
                            )
                     
                    firstVertIndex += building.polygon.numEdges

                self.processEvents(posEvents, True)
                self.processEvents(negEvents, False)

                # index in <queryBldgVerts>
                firstVertIndex = 0
                for bldgIndex in queryBldgIndices:
                    building = buildings[bldgIndex]

                    # compute visibility ratio and select edges in range
                    for edge, edgeVert1, edgeVert2 in building.polygon.edgeInfo(queryBldgVerts, firstVertIndex, skipShared=True):
                        _visInfo = edge._visInfo
                        dx = abs(edgeVert2[0] - edgeVert1[0])
                        dy = abs(edgeVert2[1] - edgeVert1[1])
                        
                        if dx:
                            _visInfo.value /= dx    # normalization of visibility
                        else:
                            _visInfo.value = 0.     # edge perpendicular to way-segment
                        # setting attributes for <_visInfo>
                        _visInfo.set(
                            segment,
                            abs(edgeVert1[1] + edgeVert2[1]),
                            dx,
                            dy
                        )
                    
                    # check for intersections and process them
                    edgeIntersections = building.crossedEdges
                    # if there are intersections, process their edges
                    if edgeIntersections:
                        # for values between 0 and 1, the intersection is in the way-segment.
                        # these may also be tunnels, which are skipped in the WayManager currently.
                            wayIntersects =  tuple( isect[0] for isect in edgeIntersections if abs(isect[1]) <= 1.001 )
                            if wayIntersects:
                                # all intersections with the way-segment itself are passages
                                for edge, intsectX in edgeIntersections:
                                    if abs(intsectX) <= 1.001:
                                        edge.cl = FacadeClass.passage # facade with passage 
                                        edge._visInfo.value = 1.
                                        edge.visInfo.update(edge._visInfo)
                            else:
                                # process the nearest axis intersections. If there are on both sides, we assume a street within a
                                # courtyard. Else, the edge gets visible.

                                # largest index on negative (left) side
                                axisLeftEdge, isec = max( (isec for isec in edgeIntersections if isec[1]<=0), key=itemgetter(1), default=(None,None))
                                if axisLeftEdge:
                                    if isec > -2.*searchWidth/segmentLength and not axisLeftEdge.hasSharedBldgVectors():
                                        # can only get dead-end, if way-level higher than already detected
                                        if axisLeftEdge.visInfo.waySegment:
                                            if WayLevel[way.category] < WayLevel[axisLeftEdge.visInfo.waySegment.way.category]:
                                                axisLeftEdge.cl = FacadeClass.deadend # dead-end at a building 
                                                axisLeftEdge._visInfo.value = 0.
                                                axisLeftEdge.visInfo.update(axisLeftEdge._visInfo)
                                        else:
                                            axisLeftEdge.cl = FacadeClass.deadend # dead-end at a building 
                                            axisLeftEdge._visInfo.value = 0.
                                            axisLeftEdge.visInfo.update(axisLeftEdge._visInfo)
                                else:
                                    # smallest index on positive (right) side
                                    axisRightEdge, isec = min( (isec for isec in edgeIntersections if isec[1]>=0.), key=itemgetter(1), default=(None,None))
                                    if axisRightEdge:
                                        if isec < 2.*searchWidth/segmentLength and not axisRightEdge.hasSharedBldgVectors():
                                            # can only get dead-end, if way-level higher than already detected
                                            if axisRightEdge.visInfo.waySegment:
                                                if WayLevel[way.category] < WayLevel[axisRightEdge.visInfo.waySegment.way.category]:
                                                    axisRightEdge.cl = FacadeClass.deadend # dead-end at a building 
                                                    axisRightEdge._visInfo.value = 0.
                                                    axisRightEdge.visInfo.update(axisRightEdge._visInfo)
                                            else:
                                                axisRightEdge.cl = FacadeClass.deadend # dead-end at a building 
                                                axisRightEdge._visInfo.value = 0.
                                                axisRightEdge.visInfo.update(axisRightEdge._visInfo)

                    # check for range and angles
                    for edge, edgeVert1, edgeVert2 in building.polygon.edgeInfo(queryBldgVerts, firstVertIndex, skipShared=True):
                        # update edges that are within search range
                        if not self.insideRange(edgeVert1, edgeVert2, halfSegmentWidth, self.searchHeight):
                            edge._visInfo.value = 0.
                        
                        if edge._visInfo > edge.visInfo or edge.cl in CrossedFacades:
                            if edge.cl in CrossedFacades:
                                if WayLevel[way.category] <= WayLevel[edge.visInfo.waySegment.way.category]:
                                    if edge._visInfo.value > FrontFacadeSight:
                                        edge.cl = FacadeClass.unknown
                                        edge.visInfo.update(edge._visInfo)
                            else: 
                                edge.visInfo.update(edge._visInfo)

                    firstVertIndex += building.polygon.numEdges

        # compute weighted average distances of way-segments
        for building in manager.buildings:
            for vector in building.polygon.getVectors():
                edge = vector.edge
                visInfo = edge.visInfo
                if visInfo.value and visInfo.waySegment:
                    visInfo.waySegment.sumVisibility += visInfo.value
                    visInfo.waySegment.sumDistance += visInfo.distance * visInfo.value
        for way in self.app.managersById["ways"].getFacadeVisibilityWays():            
            for segment in way.segments:
                segment.avgDist = segment.sumDistance / segment.sumVisibility if segment.sumVisibility else 0.

    def insideRange( self, v1, v2, xRange, yRange):
        # Checks if an edge given by vertices v1 and v2 is within or intersects
        # a rectangle parallel to the axes of the coordinate system with a range
        # of +-xRange in x-direction and +-yRange in y-direction.
        if v1[0] < v2[0]:
            xInRange = v1[0] < xRange and v2[0] > -xRange
        else:
            xInRange = v1[0] > -xRange and v2[0] < xRange
        if v1[1] < v2[1]:
            yInRange = v1[1] < yRange and v2[1] > -yRange
        else:
            yInRange = v1[1] > -yRange and v2[1] < yRange
        return xInRange and yInRange


    def processEvents(self, events, positiveY):
        """
        Process <self.posEvents> or <self.negEvents>
        """
        queue = self.priorityQueue
        activeEvent = None
        activeX = 0.
        
        # sort events by increasing start x and then by increasing abs(end y)
        events.sort(
            key = (lambda x: (x[2], x[4][1])) if positiveY else (lambda x: (x[2], -x[4][1]))
        )

        queue.cleanup()
        
        #self.drawEvents(events)
        
        for event in events:
            _, relatedEdgeStartsEvent, eventX, edgeVert1, edgeVert2 = event
            if positiveY:
                startY = edgeVert1[1]
                endY = edgeVert2[1]
            else:
                startY = -edgeVert1[1]
                endY = -edgeVert2[1]
            if not activeEvent:
                activeEvent = event
                activeX = eventX
            elif not relatedEdgeStartsEvent: # an "edge starts" event here
                activeStartX = activeEvent[3][0]
                activeEndX = activeEvent[4][0]
                if positiveY:
                    activeStartY = activeEvent[3][1]
                    activeEndY = activeEvent[4][1]
                else:
                    activeStartY = -activeEvent[3][1]
                    activeEndY = -activeEvent[4][1]
                # if both edges start at the same x-coord, the new edge is in front,
                # when its endpoint is nearer to the way-segment.
                isInFront = False
                if eventX == activeStartX:
                    isInFront = endY < activeEndY 
                # potentially, if the new edge starts in front of the active edge's 
                # maximum y-coord, it could be in front of the active edge.
                elif startY < max(activeStartY,activeEndY):
                    # if the new edge starts in behind the active edge's 
                    # minimum y-coord, the decision has to be based on the true y-coord 
                    # of the active edge at this x-coord.
                    if startY > min(activeStartY,activeEndY):
                        dx = activeEndX - activeStartX
                        # dy = activeEndY - activeStartY
                        isInFront = startY < activeStartY + (eventX-activeStartX) * (activeEndY - activeStartY) / dx\
                            if dx else False
                    # else, the new edge is in front for sure
                    else:
                        isInFront = True

                if isInFront: # the new edges hides the active edge
                    # edge = activeEvent[0]
                    if activeEvent[0]: # exclude dummy edges of crossings updateAuxVisibilityAdd
                        activeEvent[0]._visInfo.value += eventX - activeX
                    queue.push(activeEndY, activeEvent)
                    activeEvent = event
                    activeX = eventX # the new edges is behind the active edge
                else: 
                    queue.push(endY, event)
            else:
                if activeEvent is relatedEdgeStartsEvent: # the active edge ends
                    # edge = activeEvent[0]
                    if activeEvent[0]:   # exclude dummy edges of crossings updateAuxVisibilityAdd
                        activeEvent[0]._visInfo.value += eventX - activeX
                    if not queue.empty(): # there is an hidden edge that already started                   
                        activeEvent = queue.pop()
                        activeX = eventX
                    else:
                        activeEvent = None
                        activeX = 0.
                else: # there must be an edge in the queue, that ended before
                    queue.remove(relatedEdgeStartsEvent)


class FacadeVisibilityBlender(FacadeVisibility):
    
    def createKdTree(self, buildings, totalNumVerts):
        from mathutils.kdtree import KDTree
        kdTree = KDTree(totalNumVerts)
        
        index = 0
        for building in buildings:
            if building.polygon:
                for vert in building.polygon.verts:
                    kdTree.insert(vert, index)
                    index += 1
        kdTree.balance()
        self.kdTree = kdTree
    
    def makeKdQuery(self, searchCenter, searchRadius):
        """
        Returns a generator of building indices in <manager.buldings>.
        The buildings indices aren't unique
        """
        return (
            self.vertIndexToBldgIndex[vertIndex] for _,vertIndex,_ in self.kdTree.find_range(searchCenter, searchRadius)
        )


class FacadeVisibilityOther(FacadeVisibility):
    
    def createKdTree(self, buildings, totalNumVerts):
        from scipy.spatial import KDTree
        
        self.kdTree = KDTree(self.bldgVerts)
    
    def makeKdQuery(self, searchCenter, searchRadius):
        """
        Returns a generator of building indices in <manager.buldings>.
        The buildings indices aren't unique
        """
        return (
            self.vertIndexToBldgIndex[vertIndex] for vertIndex in self.kdTree.query_ball_point(searchCenter, searchRadius)
        )