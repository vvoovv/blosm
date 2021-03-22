from math import sqrt
import numpy as np
from bisect import bisect_left
from operator import itemgetter

from mpl import Mpl


class PriorityQueue():
    def __init__(self):
        self.keys = []
        self.queue = []

    def push(self, key, item):
        i = bisect_left(self.keys, key) # Determine where to insert item.
        self.keys.insert(i, key)        # Insert key of item to keys list.
        self.queue.insert(i, item)      # Insert the item itself in the corresponding place.

    def pop(self):
        self.keys.pop(0)
        return self.queue.pop(0)

    def remove(self, item):
        itemIndex = self.queue.index(item)
        del self.queue[itemIndex]
        del self.keys[itemIndex]

    def empty(self):
        return not self.queue
    
    def cleanup(self):
        self.keys.clear()
        self.queue.clear()


class FacadeVisibility:
    
    def __init__(self, searchRange=(10., 100.)):
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
        
        # Create an instance of <util.polygon.Polygon> for each <building>,
        # remove straight angles for them and calculate the total number of vertices
        for building in buildings:
            if not building.polygon:
                building.initPolygon(manager.data)
                if not building.polygon:
                    continue
            if not building.visibility:
                building.initVisibility()
        
        # the total number of vertices
        totalNumVerts = sum(building.polygon.n for building in buildings if building.polygon)
        # create mapping between the index of the vertex and index of the building in <buildings>
        self.vertIndexToBldgIndex.extend(
            bldgIndex for bldgIndex, building in enumerate(buildings) if building.polygon for _ in range(building.polygon.n)
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
                    bldgVerts[index] = (vert[0], vert[1])
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
            #if not way.polyline:
            #    way.initPolyline()
            
            for segmentCenter, segmentUnitVector, segmentLength in way.segments(manager.data):
                posEvents.clear()
                negEvents.clear()
                
                searchWidth = segmentLength/2. + self.searchWidthMargin
                # Get all building vertices for the buildings whose vertices are returned
                # by the query to the KD tree
                queryBldgIndices = set(
                    self.makeKdQuery(segmentCenter, sqrt(searchWidth*searchWidth + self.searchHeight_2))
                )
                queryBldgVertIndices = tuple(
                    vertIndex for bldgIndex in queryBldgIndices for vertIndex in range(buildings[bldgIndex].auxIndex, buildings[bldgIndex].auxIndex + buildings[bldgIndex].polygon.n)
                )
                # slice <self.bldgVerts> using <queryBldgVertIndices>
                queryBldgVerts = self.bldgVerts[queryBldgVertIndices,:]
                
                # transform <queryBldgVerts> to the system of reference of <way>
                matrix = np.array(( segmentUnitVector, (segmentUnitVector[1], -segmentUnitVector[0]) ))
                
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
                    building.resetAuxVisibility()
                    
                    for edgeIndex, edgeVert1, edgeVert2 in building.edgeInfo(queryBldgVerts, firstVertIndex):
                        if edgeVert1[0] > edgeVert2[0]:
                            # because the algorithm scans with increasing x, the first vertice has to get smaller x
                            edgeVert1, edgeVert2 = edgeVert2, edgeVert1
                        
                        dx = edgeVert2[0] - edgeVert1[0]
                        dy = edgeVert2[1] - edgeVert1[1]
                        slope = dy/0.0000001 if dx == 0. else dy/dx

                        if edgeVert1[1] > 0. and edgeVert2[1] > 0.:
                            posEvents.append(
                                (building, edgeIndex, None, edgeVert1[0], edgeVert1[1], edgeVert2[1], slope)
                            )
                            # Keep track of the related "edge starts" event,
                            # that's why <posEvents[-1]>
                            posEvents.append(
                                (building, edgeIndex, posEvents[-1], edgeVert2[0], edgeVert1[1], edgeVert2[1], slope)
                            )
                        else:
                            negEvents.append(
                                (building, edgeIndex, None, edgeVert1[0], -edgeVert1[1], -edgeVert2[1], -slope)
                            )
                            # Keep track of the related "edge starts" event,
                            # that's why <negEvents[-1]>
                            negEvents.append(
                                (building, edgeIndex, negEvents[-1], edgeVert2[0], -edgeVert1[1], -edgeVert2[1], -slope)
                            )
                     
                    firstVertIndex += building.polygon.n
                
                self.processEvents(posEvents)
                self.processEvents(negEvents)
                
                # index in <queryBldgVerts>
                firstVertIndex = 0
                for bldgIndex in queryBldgIndices:
                    building = buildings[bldgIndex]
                    
                    for edgeIndex, edgeVert1, edgeVert2 in building.edgeInfo(queryBldgVerts, firstVertIndex):
                        dx = abs(edgeVert2[0] - edgeVert1[0])
                        dy = abs(edgeVert2[1] - edgeVert1[1])
                        # at least one vertice of the edge must be in rectangular search range
                        if (abs(edgeVert1[0]) < searchWidth and abs(edgeVert1[1]) < self.searchHeight) or\
                                (abs(edgeVert2[0]) < searchWidth and abs(edgeVert2[1]) < self.searchHeight):
                            building.updateAuxVisibilityDivide(edgeIndex, dx)
                        else:
                            building.updateAuxVisibilitySet(edgeIndex, 0.)
                        if dx > dy: # abs of angle to way-segment < 45°
                            building.updateVisibilityMax(edgeIndex)
                    firstVertIndex += building.polygon.n
    
    def processEvents(self, events):
        """
        Process <self.posEvents> or <self.negEvents>
        """
        queue = self.priorityQueue
        activeEvent = None
        activeX = 0.
        
        # sort events by increasing x and then by increasing y
        events.sort(key=itemgetter(3,5))
        
        queue.cleanup()
        
        #self.drawEvents(events)
        
        for event in events:
            _, _, relatedEdgeStartsEvent, eventX, startY, endY, slope = event
            if not activeEvent:
                activeEvent = event
                activeX = eventX
            elif not relatedEdgeStartsEvent: # an "edge starts" event here
                # if both edges start at a common vertice, use y of endpoints,
                # else compute true y of active edge: y = startY + differenceOfX * slope
                isInFront = (endY < activeEvent[5]) if eventX == activeEvent[3] \
                            else (startY < activeEvent[4]+(eventX-activeEvent[3])*activeEvent[6])
                if isInFront: # the new edges hides the active edge
                    activeEvent[0].updateAuxVisibilityAdd(activeEvent[1], eventX - activeX)
                    queue.push(activeEvent[4], activeEvent)
                    activeEvent = event
                    activeX = eventX # the new edges is behind the active edge
                else: 
                    queue.push(endY, event)
            else:
                if activeEvent is relatedEdgeStartsEvent: # the active edge ends
                    activeEvent[0].updateAuxVisibilityAdd(activeEvent[1], eventX - activeX)
                    if not queue.empty(): # there is an hidden edge that already started                   
                        activeEvent = queue.pop()
                        activeX = eventX
                    else:
                        activeEvent = None
                        activeX = 0.
                else: # there must be an edge in the queue, that ended before
                    queue.remove(relatedEdgeStartsEvent)
    
    def drawEvents(self, events):
        mpl = Mpl.getMpl()
        ax = mpl.ax
        
        for building, edgeIndex, edgeStarts, eventX, eventY in events:
            ax.plot(
                eventX, eventY, marker='+', color='green' if edgeStarts else 'red'
            )
        
        mpl.show()
        mpl.shown = False


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