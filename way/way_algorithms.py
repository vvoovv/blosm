from collections import deque
import heapq
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
            path = []
            for nextEdge in network.iterAlongWay(firstEdge):
                # 1) iteration ends if another intersection node or an end-node is found
                if nextEdge.target == sectionStart:
                    # 3) the start_node is found => loop (remove completely)
                    edgesToMerge = []
                    break
                path.append(nextEdge.source)
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
            if (sectionStart, sectionEnd) not in foundEdges:
                foundEdges.extend([(sectionStart, sectionEnd), (sectionEnd, sectionStart)])
                sectionNetwork.addSegment(sectionStart,sectionEnd,totalLength, firstEdge.category, path, False)
    
    return sectionNetwork

def findWayJunctionsFor(graph, seed_crossings, categories, distance):
        processed = set()
        wayJunctions = []
        for node in seed_crossings:
            if node not in processed:
                to_process = deque([node])
                seen = {node}
                while to_process:
                    i = to_process.popleft()
                    outWays = [segment for segment in graph.iterOutSegments(i)]
                    neighbors = {
                        (w.source if w.target==i else w.target) for w in outWays \
                            if w.length < distance and w.category in categories
                    }
                    to_process.extend(list(neighbors-seen))
                    seen.add(i)
                if len(seen) > 1:
                    processed |= seen
                    wayJunctions.append(seen)
        return wayJunctions

from itertools import combinations
def findRoadClusters(graph, junctions):
    from math import sqrt
    for junction in junctions:
            x = [ v[0] for v in junction ]
            y = [ v[1] for v in junction ]
            dx = (max(x)-min(x))/2.
            dy = (max(y)-min(y))/2.
            r = 1.2*sqrt(dx*dx+dy*dy)
            plt.gca().set_aspect(1) 
            plt.gca().add_artist(plt.Circle(
                ( min(x)+dx, min(y)+dy ),
                r,
                alpha=0.3,
                color='red',
                zorder=100
            )) 
            # plt.plot(min(x)+dx,min(y)+dy, 'o', ms=15, markerfacecolor=color, alpha=0.3)#, markeredgecolor='red', markeredgewidth=5)
            for v in junction:
                x,y = v[0], v[1]
                plt.scatter(x, y, 30, color='red', zorder=100)

    comb = combinations(junctions,2)
    from itertools import cycle
    import matplotlib.colors as mcolors
    colornames = list(mcolors.CSS4_COLORS)
    cycol = cycle(colornames)

    count = 0
    for startJunction,goalJunction in comb:
        count += 1
        color = next(cycol)
        if startJunction != goalJunction:
            # outOfJunction = {segment.target for source in startJunction for segment in graph.iterOutSegments(source)}
            # outOfJunction -= startJunction
            for start in startJunction:
                path, cost = AstarSearchToJunction(graph,start,goalJunction)
                if path and cost:
                     if cost < 250:
                        p0 = path[0]
                        for i,node in enumerate(path):
                            if i>0:
                                plt.plot([p0[0],node[0]],[p0[1],node[1]],color)
                                p0 = node

    plt.gca().axis('equal')
    plt.show()



def heuristic(goal,target):
    return abs(goal[0]-target[0]) + abs(goal[1]-target[1])

def GreedyBestFirstSearchToJunction(graph,start,goalJunction):
    # plt.plot(start[0],start[1],'ro')
    # for n in goalJunction:
    #     plt.plot(n[0],n[1],'k.')
    x0 = [node[0] for node in goalJunction]
    y0 = [node[1] for node in goalJunction]
    junctionCenter = ( sum(x0)/len(goalJunction),sum(y0)/len(goalJunction))
    # plt.plot(junctionCenter[0],junctionCenter[1],'bo')

    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = dict()
    came_from[start] = None
    segments = []

    while not frontier.empty():
        current = frontier.get()

        if current in goalJunction or len(came_from)>20:
            if current in goalJunction:
                path = []
                while current != start: 
                    path.append(current)
                    current = came_from[current]
                path.append(start) # optional
                path.reverse()
                return path
                #     plt.text((seg.source[0]+seg.target[0])/2,(seg.source[1]+seg.target[1])/2,str(i))
                # plt.plot([seg.target[0],current[0]],[seg.target[1],current[1]],'k')
            break
        
        for nextSeg in graph.iterOutSegments(current):
            nextNode = nextSeg.target
            if nextNode not in came_from:
                priority = heuristic(junctionCenter, nextNode)
                frontier.put(nextNode, priority)
                came_from[nextNode] = current
                segments.append(nextSeg)

def AstarSearchToJunction(graph,start,goalJunction):
    from math import sqrt, inf
    x0 = [node[0] for node in goalJunction]
    y0 = [node[1] for node in goalJunction]
    junctionCenter = ( sum(x0)/len(goalJunction),sum(y0)/len(goalJunction))
    costThresh = 1.1*sqrt( (start[0]-junctionCenter[0])*(start[0]-junctionCenter[0])+(start[1]-junctionCenter[1])*(start[1]-junctionCenter[1]) )

    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = dict()
    cost_so_far = dict()
    segment = dict()
    came_from[start] = None
    cost_so_far[start] = 0
    segment[start] = None

    while not frontier.empty():
        current = frontier.get()

        if current in goalJunction:
            path = []
            cost = 0
            while current != start: 
                path.append(current)
                current = came_from[current]
                cost += segment[current].length if segment[current] else 0
                test=1
            path.append(start) # optional
            path.reverse()
            test = cost_so_far
            return path, cost
       
        for nextSeg in graph.iterOutSegments(current):
            nextNode = nextSeg.target
            new_cost = cost_so_far[current] + nextSeg.length
            if nextNode not in cost_so_far or new_cost < cost_so_far[nextNode]:
                if new_cost > costThresh:
                    return [], inf
                cost_so_far[nextNode] = new_cost
                priority = new_cost + heuristic(junctionCenter, nextNode)
                frontier.put(nextNode, priority)
                came_from[nextNode] = current
                segment[nextNode] = nextSeg