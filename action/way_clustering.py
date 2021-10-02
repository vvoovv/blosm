from collections import deque
from way.way_network import WayNetwork
from way.way_algorithms import createSectionNetwork, findWayJunctionsFor, findRoadClusters
from defs.way import allRoadwayCategories, mainRoads, smallRoads


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
        allCrossings = graph.getCrossingsThatContain(allRoadwayCategories)
        mainJunctions = findWayJunctionsFor(graph, allCrossings, mainRoads, 20.)

        # expand them with near crossings of small roads
        for cluster in mainJunctions:
            for side_cluster in findWayJunctionsFor(graph, cluster, smallRoads, 15.):
                cluster |= side_cluster

        # remove these crossings from <allCrossings>
        remainingCrossings = list({crossing for crossing in allCrossings} -\
                        {crossing for cluster in mainJunctions for crossing in cluster })

        # find way-junctions for small roads in <remainingCrossings>
        smallJunctions = findWayJunctionsFor(graph, remainingCrossings, smallRoads, 15.)

        # does not yet work
        # findRoadClusters(graph, mainJunctions)

        wayManager.junctions = (
            mainJunctions,
            smallJunctions
        )
    
    def cleanup(self):
        pass

