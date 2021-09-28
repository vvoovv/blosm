import numpy as np
from collections import deque
from way.way_network_graph import OSMWay, WayNetworkGraph, SectionGraphCreator
import  matplotlib.pyplot as plt

main_roads =   (  
    "primary",
    # "primary_link",
    "secondary",
    # "secondary_link",
    "tertiary",
)

small_roads = (
    "residential",
    "service",
    # "pedestrian",
    # "track",
    # "escape",
    # "footway",
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


class RoadClustering:
    
    def __init__(self):
        self.networkGraph = None
    
    def do(self, manager):
        # prepare data structures required for WayNetworkGraph
        nodes = {}
        ways = []
        for ID, way in enumerate( self.app.managersById["ways"].getAllWays() ):
            length = 0.
            for segment in way.segments:
                nodes[segment.id1] = segment.v1
                nodes[segment.id2] = segment.v2
                length += segment.length
            name = way.element.tags['name'] if 'name' in way.element.tags else ''
            ways.append(OSMWay(ID, name, way.category, way.element.nodes, length ))

        self.networkGraph = WayNetworkGraph(nodes, ways)
        # debugPlot(self.networkGraph, 'Full Network')
        sectionCreator = SectionGraphCreator(self.networkGraph)
        self.waySectionGraph = sectionCreator.createSectionNetwork()
        # debugPlot(self.waySectionGraph, 'Section Network')

        # find road-clusters for principal roads
        all_crossings = self.waySectionGraph.get_crossings_that_contain(allWayCategories)
        main_clusters = findRoadCrossingsFor(self.waySectionGraph, all_crossings, main_roads, 20.)

        # expand them with near crossings of small roads
        for cluster in main_clusters:
            for side_cluster in findRoadCrossingsFor(self.waySectionGraph, cluster, small_roads, 15.):
                cluster |= side_cluster

        # remove these crossings from <all_crossings>
        remaining_crossings = list({crossing for crossing in all_crossings} -\
                        {crossing for cluster in main_clusters for crossing in cluster })

        # find road-clusters for small roads in the remianing croosings
        small_clusters = findRoadCrossingsFor(self.waySectionGraph, remaining_crossings, small_roads, 15.)

        plotStart()
        plotRoads(self.networkGraph)
        plotRoadCrossings(self.waySectionGraph, main_clusters, 'red')
        plotRoadCrossings(self.waySectionGraph, small_clusters, 'limegreen')
        plotEnd()


        assert False, 'The experimental part ends here. No renderer has been implemented yet.'

        def cleanup(self):
            pass

def findRoadCrossingsFor(graph, seed_crossings, categories, distance):
        # seed_crossings = graph.get_crossings_that_contain(categories)
        outWays = graph.get_out_edges()
        processed = set()
        road_crossings = []
        for indx in seed_crossings:
            if indx not in processed:
                to_process = deque([indx])
                seen = {indx}
                while to_process:
                    i = to_process.popleft()
                    neighbors = { (e.s if e.t==i else e.t) for e in outWays[i] \
                                   if e.length<distance and e.category in categories }
                    to_process.extend(list(neighbors-seen))
                    seen.add(i)
                if len(seen)>1:
                    processed |= seen
                    road_crossings.append(seen)
        return road_crossings


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

        

    
def debugPlot( graph, title):
        edges = graph.edges
        v = graph.vertices
        n = graph.out_ways
        c = graph.way_categories
        for e in edges:
            v1 = v[e.s].data
            v2 = v[e.t].data
            plt.plot([v1[0],v2[0]],[v1[1],v2[1]],'k')
            x = (v1[0]+v2[0])/2.
            y = (v1[1]+v2[1])/2.
            plt.text(x,y,'  %4.1f'%(e.geom_dist))
            # plt.text(x,y,'  '+e.category+' '+str(e.iID))

        for indx, vert in enumerate(v):
            x,y = vert.data[0], vert.data[1]
            categories = c[indx]
            degree = len(n[indx])
            if degree == 1:
                plt.scatter(x,y,15,color='gray')
            elif degree == 2 and c[indx][0] != c[indx][1]:
                plt.scatter(x,y,30,color='limegreen')
                # plt.text(x,y,'  '+str(vert.iID) + ' ' + c[indx][0] + ' ' + c[indx][1] )
            elif degree == 3:
                plt.scatter(x,y,30,color='blue')
            elif degree > 3:
                plt.scatter(x,y,30,color='red')
                # plt.text(x,y,'  '+str(vert.iID))


def plotStart():
    fig = plt.figure()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

def plotEnd(title=''):
        plt.gca().axis('equal')
        # plt.xlim([-600,750])
        # plt.ylim([-400,400])
        plt.title(title)
        plt.show()

def plotRoads(graph):
    edges = graph.edges
    v = graph.vertices
    n = graph.out_ways
    c = graph.way_categories
    for e in edges:
            v1 = v[e.s].data
            v2 = v[e.t].data
            plt.plot([v1[0],v2[0]],[v1[1],v2[1]],'k')
            x = (v1[0]+v2[0])/2.
            y = (v1[1]+v2[1])/2.
            # plt.text(x,y,'  %4.1f'%(e.length))
            # plt.text(x,y,e.category)
            # plt.text(v1[0],v1[1],'  '+str(e.s))
            # plt.text(v2[0],v2[1],'  '+str(e.t))

def plotRoadCrossings(graph, crossings, color):
    edges = graph.edges
    v = graph.vertices
    n = graph.out_ways
    c = graph.way_categories

    from math import sqrt
    for crossing in crossings:
        x = [v[indx].data[0] for indx in crossing ]
        y = [v[indx].data[1] for indx in crossing ]
        dx = (max(x)-min(x))/2.
        dy = (max(y)-min(y))/2.
        r = sqrt(dx*dx+dy*dy)*1.2
        cc = plt.Circle(( min(x)+dx, min(y)+dy ), r , alpha=0.3, color=color)
        ax = plt.gca()
        ax.set_aspect( 1 ) 
        ax.add_artist( cc ) 
        # plt.plot(min(x)+dx,min(y)+dy, 'o', ms=15, markerfacecolor=color, alpha=0.3)#, markeredgecolor='red', markeredgewidth=5)
        for indx in crossing:
            x,y = v[indx].data[0], v[indx].data[1]
            plt.scatter(x,y,30,color=color)
