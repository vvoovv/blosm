from collections import deque
from way.way_network import WayNetwork
from way.way_algorithms import createSectionNetwork

main_roads =   (  
    "primary",
    # "primary_link",
    "secondary",
    # "secondary_link",
    "tertiary",
    "residential"
)

small_roads = (
    #"residential",
    "service",
    # "pedestrian",
    # "track",
    # "escape",
    "footway",
    # "bridleway",
    # "steps",
    # "path",
    "cycleway"
)

allWayCategories = (
    "other",
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "living_street",
    "service",
    "pedestrian",
    "track",
    "escape",
    "raceway",
    # "road", # other
    "footway",
    "bridleway",
    "steps",
    "path",
    "cycleway"
)


class WayClustering:
    
    def __init__(self):
        self.networkGraph = None
    
    def do(self, wayManager):
        # create full way network
        wayManager.networkGraph = WayNetwork()
        for way in wayManager.getAllWays():
            for segment in way.segments:
                wayManager.networkGraph.addSegment(tuple(segment.v1),tuple(segment.v2),segment.length,way.category)

        # create way-section network
        graph = wayManager.waySectionGraph = createSectionNetwork(wayManager.networkGraph)

        # find way-junctions for principal roads
        allCrossings = graph.getCrossingsThatContain(allWayCategories)
        mainJunctions = self.findWayJunctionsFor(graph, allCrossings, main_roads, 20.)

        # expand them with near crossings of small roads
        for cluster in mainJunctions:
            for side_cluster in self.findWayJunctionsFor(graph, cluster, small_roads, 15.):
                cluster |= side_cluster

        # remove these crossings from <allCrossings>
        remainingCrossings = list({crossing for crossing in allCrossings} -\
                        {crossing for cluster in mainJunctions for crossing in cluster })

        # find way-junctions for small roads in <remainingCrossings>
        smallJunctions = self.findWayJunctionsFor(graph, remainingCrossings, small_roads, 15.)
        test=1
        
        wayManager.junctions = (
            mainJunctions,
            smallJunctions
        )
    
    def cleanup(self):
        pass

    def findWayJunctionsFor(self, graph, seed_crossings, categories, distance):
            # seed_crossings = graph.get_crossings_that_contain(categories)
            # outWays = [segment[0] for segment in graph.iterAllSegments()]
            processed = set()
            wayJunctions = []
            for node in seed_crossings:
                if node not in processed:
                    to_process = deque([node])
                    seen = {node}
                    while to_process:
                        i = to_process.popleft()
                        outWays = [segment[0] for segment in graph.iterOutSegments(i)]
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


# from collections import deque
# def findCrossingsFor(graph, indx, outways, categories, processed, distance):
#         to_process = deque([indx])
#         seen = {indx}
#         while to_process:
#             i = to_process.popleft()
#             neighbors = { (e.s if e.t==i else e.t) for e in outways[i] if e.length<distance and e.category in categories }
#             to_process.extend(list(neighbors-seen))
#             seen.add(i)
#         if len(seen)>1:
#             processed |= seen
#         return seen if len(seen)>1 else set()