from math import sqrt
from . import Renderer
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


class RoadPolygonsRenderer(Renderer):
    
    styles = dict(
        motorway = dict(
            color="#e892a2", linewidth=7.,
            path_effects=[pe.Stroke(linewidth=7.4, foreground="#e15681"), pe.Normal()]
        ),
        motorway_link = dict(
            color="#e892a2", linewidth=7.,
            path_effects=[pe.Stroke(linewidth=7.4, foreground="#e15681"), pe.Normal()]
        ),
        trunk = dict(
            color="#e892a2", linewidth=7.,
            path_effects=[pe.Stroke(linewidth=7.4, foreground="#e15681"), pe.Normal()]
        ),
        trunk_link = dict(
            color="#e892a2", linewidth=7.,
            path_effects=[pe.Stroke(linewidth=7.4, foreground="#e15681"), pe.Normal()]
        ),
        primary = dict(
            color="#fcd6a4", linewidth=5.,
            path_effects=[pe.Stroke(linewidth=5.4, foreground="black"), pe.Normal()]
        ),
        primary_link = dict(
            color="#fcd6a4", linewidth=5.,
            path_effects=[pe.Stroke(linewidth=5.4, foreground="black"), pe.Normal()]
        ),
        secondary = dict(
            color="#f7fabf",
            linewidth=4.,
            path_effects=[pe.Stroke(linewidth=4.4, foreground="black"), pe.Normal()]
        ),
        secondary_link = dict(
            color="#f7fabf",
            linewidth=4.,
            path_effects=[pe.Stroke(linewidth=4.4, foreground="black"), pe.Normal()]
        ),
        tertiary = dict(
            color="#ffffff", linewidth=3.,
            path_effects=[pe.Stroke(linewidth=3.4, foreground="black"), pe.Normal()]
        ),
        tertiary_link = dict(
            color="#ffffff", linewidth=3.,
            path_effects=[pe.Stroke(linewidth=3.4, foreground="black"), pe.Normal()]
        ),
        residential = dict(
            color="#ffffff", linewidth=3.,
            path_effects=[pe.Stroke(linewidth=3.4, foreground="black"), pe.Normal()]
        ),
        unclassified = dict(
            color="#ffffff", linewidth=3.,
            path_effects=[pe.Stroke(linewidth=3.4, foreground="black"), pe.Normal()]
        ),
        pedestrian = dict(
            color="#dddde8", linewidth=3.,
            path_effects=[pe.Stroke(linewidth=3.4, foreground="black"), pe.Normal()]
        ),
        living_street = dict(
            color="#ededed", linewidth=3.,
            path_effects=[pe.Stroke(linewidth=3.4, foreground="black"), pe.Normal()]
        ),
        service = dict(
            color="#ffffff", linewidth=2.,
            path_effects=[pe.Stroke(linewidth=2.4, foreground="black"), pe.Normal()]
        ),
        raceway = dict(
            color="#ffffff", linewidth=3.,
            path_effects=[pe.Stroke(linewidth=3.4, foreground="black"), pe.Normal()]
        ),
        footway = dict(color="#fa8072", linewidth=1., linestyle="dashed"),
        steps = dict(color="#fa8072", linewidth=2., linestyle="dotted"),
        path = dict(color="#fa8072", linewidth=1., linestyle="dashed"),
        track = dict(color="#fa8072", linewidth=1., linestyle="dashed"),
        cycleway = dict(color="#0000ff", linewidth=1., linestyle="dashed"),
        # a special case
        scene_border = dict(color="#ff0000", linewidth=1., linestyle="solid"),
        # railways
        tram = dict(color="#000000", linewidth=2., linestyle="solid"),
        rail = dict(color="#000000", linewidth=3., linestyle="solid")
    )
    
    def render(self, wayManager, data):
        self.mpl.ax.set_facecolor("#f2efe9")
        self.plotWay(wayManager.networkGraph)
        self.plotJunctions(wayManager.waySectionGraph, wayManager.junctions[0], 'red')
        self.plotJunctions(wayManager.waySectionGraph, wayManager.junctions[1], 'limegreen')

    def plotWay(self, graph):
        segments = [segment for segment in graph.iterAllSegments()]
        # v = graph.vertices
        # n = graph.out_ways
        # c = graph.way_categories
        for s in segments:
            v1 = s.s
            v2 = s.t
            self.mpl.ax.plot(
                (v1[0], v2[0]),
                (v1[1], v2[1]),
                **RoadPolygonsRenderer.styles[s.category],
                zorder=50
            )
            # x = (v1[0]+v2[0])/2.
            # y = (v1[1]+v2[1])/2.
            # plt.text(x,y,'  %4.1f'%(e.length))
            # plt.text(x,y,e.category)
            # plt.text(v1[0],v1[1],'  '+str(e.s))
            # plt.text(v2[0],v2[1],'  '+str(e.t))

    def plotJunctions(self, graph, crossings, color):
        ax = self.mpl.ax
        
        # edges = graph.edges
        # v = graph.vertices
        # n = graph.out_ways
        # c = graph.way_categories
        
        for crossing in crossings:
            x = [ v[0] for v in crossing ]
            y = [ v[1] for v in crossing ]
            dx = (max(x)-min(x))/2.
            dy = (max(y)-min(y))/2.
            r = 1.2*sqrt(dx*dx+dy*dy)
            ax.set_aspect(1) 
            ax.add_artist(plt.Circle(
                ( min(x)+dx, min(y)+dy ),
                r,
                alpha=0.3,
                color=color,
                zorder=100
            )) 
            # plt.plot(min(x)+dx,min(y)+dy, 'o', ms=15, markerfacecolor=color, alpha=0.3)#, markeredgecolor='red', markeredgewidth=5)
            for v in crossing:
                x,y = v[0], v[1]
                ax.scatter(x, y, 30, color=color, zorder=100)
        
        
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

# def findWayJunctionsFor(graph, seed_crossings, categories, distance):
#         processed = set()
#         wayJunctions = []
#         for node in seed_crossings:
#             if node not in processed:
#                 to_process = deque([node])
#                 seen = {node}
#                 while to_process:
#                     i = to_process.popleft()
#                     outWays = [segment for segment in graph.iterOutSegments(i)]
#                     neighbors = {
#                         (w.source if w.target==i else w.target) for w in outWays \
#                             if w.length < distance and w.category in categories
#                     }
#                     to_process.extend(list(neighbors-seen))
#                     seen.add(i)
#                 if len(seen) > 1:
#                     processed |= seen
#                     wayJunctions.append(seen)
#         return wayJunctions
