import numpy as np
from way.way_network_graph import OSMWay, WayNetworkGraph, SectionGraphCreator
import  matplotlib.pyplot as plt

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
        self.debugPlot(self.networkGraph, 'Full Network')
        sectionCreator = SectionGraphCreator(self.networkGraph)
        self.waySectionGraph = sectionCreator.createSectionNetwork()
        self.debugPlot(self.waySectionGraph, 'Section Network')
        assert False, 'The experimental part ends here. No renderer has been implemented yet.'
    
    def debugPlot(self, graph, title):
        fig = plt.figure()
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
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
            # plt.text(x,y,'  '+e.category+' '+str(e.iID))

        for indx, vert in enumerate(v):
            x,y = vert.data[0], vert.data[1]
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

        plt.gca().axis('equal')
        plt.xlim([-600,750])
        plt.ylim([-400,400])
        plt.title(title)
        plt.show()

    def cleanup(self):
        pass