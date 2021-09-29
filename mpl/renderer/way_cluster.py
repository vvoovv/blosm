from math import sqrt
from . import Renderer
import matplotlib.pyplot as plt


class WayClusterRenderer(Renderer):
    
    def render(self, wayManager, data):
        self.plotWay(wayManager.networkGraph)
        self.plotJunctions(wayManager.waySectionGraph, wayManager.junctions[0], 'red')
        self.plotJunctions(wayManager.waySectionGraph, wayManager.junctions[1], 'limegreen')

    def plotWay(self, graph):
        edges = graph.edges
        v = graph.vertices
        n = graph.out_ways
        c = graph.way_categories
        for e in edges:
            v1 = v[e.s].data
            v2 = v[e.t].data
            self.mpl.ax.plot(
                [v1[0],v2[0]],
                [v1[1],v2[1]],
                'k'
            )
            # x = (v1[0]+v2[0])/2.
            # y = (v1[1]+v2[1])/2.
            # plt.text(x,y,'  %4.1f'%(e.length))
            # plt.text(x,y,e.category)
            # plt.text(v1[0],v1[1],'  '+str(e.s))
            # plt.text(v2[0],v2[1],'  '+str(e.t))

    def plotJunctions(self, graph, crossings, color):
        ax = self.mpl.ax
        
        edges = graph.edges
        v = graph.vertices
        n = graph.out_ways
        c = graph.way_categories
        
        for crossing in crossings:
            x = [ v[indx].data[0] for indx in crossing ]
            y = [ v[indx].data[1] for indx in crossing ]
            dx = (max(x)-min(x))/2.
            dy = (max(y)-min(y))/2.
            r = 1.2*sqrt(dx*dx+dy*dy)
            ax.set_aspect(1) 
            ax.add_artist(plt.Circle(
                ( min(x)+dx, min(y)+dy ),
                r,
                alpha=0.3,
                color=color
            )) 
            # plt.plot(min(x)+dx,min(y)+dy, 'o', ms=15, markerfacecolor=color, alpha=0.3)#, markeredgecolor='red', markeredgewidth=5)
            for indx in crossing:
                x,y = v[indx].data[0], v[indx].data[1]
                ax.scatter(x, y, 30, color=color)
        
        
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