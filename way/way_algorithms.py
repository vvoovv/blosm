from collections import deque
from way.way_network import Segment, WayNetwork

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
