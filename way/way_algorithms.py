from collections import deque
import heapq
from itertools import combinations
from scipy.spatial import cKDTree 
from way.way_network import WayNetwork, NetSegment
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

class Junction():
    def __init__(self):
        self.nodes = set()

    def addNode(self,node):
        self.nodes.add(node)

    def merge(self,other):
        self.nodes.a |= other.nodes

    def __iter__(self):
        return iter(self.nodes)

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
            firstSegment = network.getSegment(sectionStart,firstNeighbor)
            sectionEnd = firstSegment.t
            mergedSegment = NetSegment(firstSegment)
            path = [sectionStart]
            for nextSegment in network.iterAlongWay(firstSegment):
                # 1) iteration ends if another intersection node or an end-node is found
                if nextSegment.t == sectionStart:
                    # 3) the start_node is found => loop (remove completely)
                    mergedSegment = None
                    break
                path.append(nextSegment.t)
                if nextSegment.category != firstSegment.category:
                    # 2) an inter-category node is found
                    if nextSegment.s not in seenStartNodes:
                        # found a new possible start node
                        seenStartNodes.add(nextSegment.s)
                        startNodes.append(nextSegment.s)
                    break
                mergedSegment.join(nextSegment)
                sectionEnd = nextSegment.t

            # totalLength = sum([e.length for e in edgesToMerge])
            # path.append(sectionEnd)
            if (sectionStart, sectionEnd) not in foundEdges:
                foundEdges.extend([(sectionStart, sectionEnd), (sectionEnd, sectionStart)])
                sectionNetwork.addSegment(mergedSegment, True)
    
    return sectionNetwork

def findWayJunctionsFor(graph, seed_crossings, categories, distance):
    processed = set()
    wayJunctions = []
    for node in seed_crossings:
        # plt.plot(node[0],node[1],'b.',zorder=150)
        if node not in processed:
            to_process = deque([node])
            seen = {node}
            while to_process:
                i = to_process.popleft()
                outWays = [segment for segment in graph.iterOutSegments(i)]
                neighbors = {
                    (w.source if w.t==i else w.t) for w in outWays \
                        if w.length < distance and w.category in categories
                }
                if len(neighbors) > 2:
                    to_process.extend(list(neighbors-seen))
                    seen.add(i)
            if len(seen) > 1:
                processed |= seen
                wayJunctions.append(seen)
    return wayJunctions
