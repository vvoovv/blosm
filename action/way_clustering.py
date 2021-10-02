from collections import deque
from way.way_network import WayNetwork
from way.way_algorithms import createSectionNetwork, findWayJunctionsFor

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
        mainJunctions = findWayJunctionsFor(graph, allCrossings, main_roads, 20.)

        # expand them with near crossings of small roads
        for cluster in mainJunctions:
            for side_cluster in findWayJunctionsFor(graph, cluster, small_roads, 15.):
                cluster |= side_cluster

        # remove these crossings from <allCrossings>
        remainingCrossings = list({crossing for crossing in allCrossings} -\
                        {crossing for cluster in mainJunctions for crossing in cluster })

        # find way-junctions for small roads in <remainingCrossings>
        smallJunctions = findWayJunctionsFor(graph, remainingCrossings, small_roads, 15.)

        wayManager.junctions = (
            mainJunctions,
            smallJunctions
        )
    
    def cleanup(self):
        pass

