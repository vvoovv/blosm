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
    forbiddenStarts = []

    while len(startNodes) > 0:
        startNode = startNodes.popleft()
        # iterate over all out-segments of this start node
        for outSegment in network.iterOutSegments(startNode):
            # do not use return segment of undirected graph
            if (outSegment.s, outSegment.t) not in forbiddenStarts:
                segmentsToMerge = []
                # iterate until intersection or different category found
                for nextSegment in network.iterAlongWay(outSegment):
                    if nextSegment.t == startNode:
                        # we are back to the start node => loop (remove completely)
                        segmentsToMerge = []
                        break
                    else:
                        segmentsToMerge.append(nextSegment)

                if segmentsToMerge:
                    forbiddenStarts.append( (nextSegment.t, nextSegment.s))
                    mergedSegment = NetSegment(segmentsToMerge[0])
                    for seg in segmentsToMerge[1:]:
                        mergedSegment.join(seg)
                    sectionNetwork.addSegment(mergedSegment)
    
    return sectionNetwork


                
    
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
                if len(neighbors) > 1:
                    to_process.extend(list(neighbors-seen))
                    seen.add(i)
            if len(seen) > 1:
                processed |= seen
                wayJunctions.append(seen)
    return wayJunctions
