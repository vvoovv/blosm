from collections import deque
from way.way_network import Segment, WayNetwork

def createSectionNetwork(network):
    sectionNetwork = WayNetwork()
    # Intialize a container <startNodes> with all nodes that are intersections or ends (degree != 2)
    # For each node in this <startNodes>, given by its index <nodeID>:
    #     Find all neighbors as <neighbors>
    #     For each edge <neighbor> of these neighbors, starting by <startNodeId>:
    #         Follow the neighbors of the way started by <firstNeighbor>, until an <end_node> is found:
    #             1) another intersection node is found
    #             2) an edge is encountered on the way that has a different category
    #             3) the start_node is found => loop (remove completely)
    #             The edges from <start_node> to <end_node> get merged to a <way_section>
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
            for nextEdge in network.iterAlongWay(firstEdge):
                # 1) iteration ends if another intersection node or an end-node is found
                if nextEdge.target == sectionStart:
                    # 3) the start_node is found => loop (remove completely)
                    edgesToMerge = []
                    break
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
                sectionNetwork.addSegment(sectionStart,sectionEnd,totalLength, firstEdge.category)
    
    return sectionNetwork
