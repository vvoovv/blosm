from collections import deque
import heapq
from itertools import combinations
from scipy.spatial import cKDTree 
from way.way_network import Segment, WayNetwork
import matplotlib.pyplot as plt

class PriorityQueue:
    def __init__(self):
        self.elements = []
    
    def empty(self):
        return not self.elements
    
    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))
    
    def get(self):
        return heapq.heappop(self.elements)[1]

# class Junction():
#     def __init__(self,junction):
#         self.junction = junction
#         x,y = self.getXY()
#         self.center = ( sum(x)/len(junction),sum(y)/len(junction))

#     def getXY(self):
#         x = [v[0] for v in self.junction]
#         y = [v[1] for v in self.junction]
#         return x,y

#     def iterNodes(self):
#         for node in self.junction:
#             yield node

#     def __lt__(self, other):
#         return len(other.junction) < len(self.junction)




def createSectionNetwork(network):
    sectionNetwork = WayNetwork()
    # Intialize a container with all nodes that are intersections or ends (degree != 2)
    # For each node <sectionStart> in this container:
    #     Find next neighbor of this node
    #     For each edge to these neighbors:
    #         Follow the neighbors of the way started by the segment from node to this neighbor,
    #             until an end-node <sectionEnd> is found:
    #             1) another intersection node is found
    #             2) an edge is encountered on the way that has a different category
    #             3) the start node is found => loop (remove completely)
    #             The edges from <sectionStart> to <sectionEnd> get merged to a new
    #             way-segment and added to the new graph
    startNodes = deque( [node for node in network.iterAllIntersectionNodes()])
    seenStartNodes = set(startNodes)
    foundEdges = []

    while len(startNodes) > 0:
        sectionStart = startNodes.popleft()
        for firstNeighbor in network.iterAdjacentNodes(sectionStart):

            # merge edges along path
            firstEdge = network.getSegment(sectionStart,firstNeighbor)
            sectionEnd = firstEdge.target
            edgesToMerge = [firstEdge]
            path = [sectionStart]
            for nextEdge in network.iterAlongWay(firstEdge):
                # 1) iteration ends if another intersection node or an end-node is found
                if nextEdge.target == sectionStart:
                    # 3) the start_node is found => loop (remove completely)
                    edgesToMerge = []
                    break
                path.append(nextEdge.target)
                if nextEdge.category != firstEdge.category:
                    # 2) an inter-category node is found
                    if nextEdge.source not in seenStartNodes:
                        # found a new possible start node
                        seenStartNodes.add(nextEdge.source)
                        startNodes.append(nextEdge.source)
                    break
                edgesToMerge.append(nextEdge)
                sectionEnd = nextEdge.target

            totalLength = sum([e.length for e in edgesToMerge])
            path.append(sectionEnd)
            if (sectionStart, sectionEnd) not in foundEdges:
                foundEdges.extend([(sectionStart, sectionEnd), (sectionEnd, sectionStart)])
                sectionNetwork.addSegment(sectionStart,sectionEnd,totalLength, firstEdge.category, path, False)
    
    return sectionNetwork

def mergePoints(points, dist):
    tree = cKDTree(points) 
    pairs = tree.query_pairs(dist) 
    neighbors = {} 
    for i,j in pairs: 
        neighbors.setdefault(i, set()).add(j)
        neighbors.setdefault(j, set()).add(i)
    processed = set()
    clusters = []
    for i in range(len(points)):
        if i not in processed:
            if i in neighbors:
                to_process = deque([i])
                seen = {i}
                while to_process:
                    j = to_process.popleft()
                    newNeighbors = neighbors[j]
                    to_process.extend(list(newNeighbors-seen))
                    seen.add(j)
                if len(seen) > 1:
                    processed |= seen
                    clusters.append({points[k] for k in seen})
                pass
            # else:
            #     clusters.append({points[i]})
            #     processed |= {i}
    return clusters

# def findWayJunctionsFor(graph, seed_crossings, categories, distance):
#         processed = set()
#         wayJunctions = []
#         for node in seed_crossings:
#             if node not in processed:
#                 to_process = deque([node])
#                 seen = {node}
#                 while to_process:
#                     i = to_process.popleft()
#                     outWays = [segment for segment in graph.iterOutSegments(i)]
#                     neighbors = {
#                         (w.source if w.target==i else w.target) for w in outWays \
#                             if w.length < distance and w.category in categories
#                     }
#                     to_process.extend(list(neighbors-seen))
#                     seen.add(i)
#                 if len(seen) > 1:
#                     processed |= seen
#                     wayJunctions.append(seen)
#         return wayJunctions

def findWayJunctionsFor(graph, seed_crossings, categories, distance):
    validNodes = []
    for node in seed_crossings:
        if any([segment.category in categories for segment in graph.iterOutSegments(node)]):
            validNodes.append(node)
    if validNodes:
        return mergePoints(validNodes, distance)
    else:
        return {}

def isStraight(inS , outS):
    from mathutils import Vector
    inV = Vector(inS.target) - Vector(inS.source)
    outV = Vector(outS.target) - Vector(outS.source)
    inUnitV = inV/inV.length
    outUnitV = outV/outV.length
    cosine = inUnitV.dot(outUnitV)
    return cosine > 0.5

def iterAlongRoad(lastSegment, network ):
    # follow way in the direction of the first segment until a crossing occurs
    current = lastSegment
    while True:
        # if len(network[current.target]) == 1: # end node
        #     break
        nextSegs = [s for s in network.iterOutSegments(current.target) if s.category==current.category \
            and isStraight(current,s) and s.target != current.source]
        if nextSegs:
            current = nextSegs[0]
            yield current
        else:
            break

def findWayClusters(network, junctions):
    from collections import deque
    junctionCenters = {}
    junctionNodes = set()
    for junction in junctions:
        x0 = sum([n[0] for n in junction])/len(junction)
        y0 = sum([n[1] for n in junction])/len(junction)
        for node in junction:
            junctionCenters[node] = (x0,y0)
            junctionNodes.add(node)

    sectionNetwork = WayNetwork()
    for jIndx,junction in enumerate(junctions):
        # print( jIndx, len(junction))
        # find outgoing segments of this junction
        outSegments = {}
        for node in junction:
            for outSeg in network.iterOutSegments(node):
                if outSeg.target not in junction:
                    outSegments[outSeg] = jIndx

        # these are the start segments for a way search to the next junction
        startSegments = deque( outSegments )
        foundSegments = []

        # iterate over out-segments
        while len(startSegments) > 0:
            firstSegment = startSegments.popleft()
            roadStart = firstSegment.source
            segmentsToMerge = [firstSegment]
            path = firstSegment.path
            nextJunctionFound = False
            # follow the road until new junctin is found
            for nextSegment in iterAlongRoad(firstSegment,network):
                roadEnd = nextSegment.target 
                if nextSegment.target == firstSegment.source:
                    # the start segment is found => loop (remove completely)
                    segmentsToMerge = []
                    break
                if nextSegment.target in junction:
                    # we are back in the starting juction => loop (remove completely)
                    segmentsToMerge = []
                    break
                if nextSegment.target in junctionNodes:
                    nextJunctionFound = True
                    segmentsToMerge.append(nextSegment)
                    path.extend(nextSegment.path)
                    break
                path.extend(nextSegment.path)
                segmentsToMerge.append(nextSegment)

            if nextJunctionFound:
                if (roadStart, roadEnd) not in foundSegments:
                    foundSegments.extend([(roadStart, roadEnd), (roadEnd, roadStart)])
                    totalLength = sum([e.length for e in segmentsToMerge])
                    # print(jIndx, junctionCenters[roadStart], junctionCenters[roadEnd])
                    sectionNetwork.addSegment(junctionCenters[roadStart],junctionCenters[roadEnd],totalLength, firstSegment.category, path, False)

    plotWayClusters(sectionNetwork)
    return sectionNetwork

def plotWayClusters(network):
    linewidth = 2
    import matplotlib.pyplot as plt
    import random
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    for source in network.iterNodes():
        for target in network[source]:
            c = random.choice(colors)
            ln = len(network[source][target])
            if ln < 1:
                continue
            for seg in network[source][target]:
            # seg = sectionNetwork[source][target][0]
                v1 = seg.source
                v2 = seg.target
                path = seg.path
                x = (v1[0]+v2[0])/2.
                y = (v1[1]+v2[1])/2.
                # plt.text(x,y,'  %d'%(ln), zorder=200)
                if seg.path:
                    last = path[0]
                    for n in path[1:]:
                        plt.plot((last[0], n[0]),(last[1], n[1]),c=c, zorder=200, linewidth=linewidth)
                        last = n
                    plt.plot((last[0], v2[0]),(last[1], v2[1]),c=c, zorder=200, linewidth=linewidth)
                else:
                    plt.plot((v1[0], v2[0]),(v1[1], v2[1]),c=c, zorder=200, linewidth=linewidth)


