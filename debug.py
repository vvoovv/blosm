from matplotlib import pyplot as plt

def plotNetwork(network,waySections=None):
    from mpl.renderer.road_polygons import RoadPolygonsRenderer
    # for section in waySections.values():
    #     seg = section.originalSection
    for count,seg in enumerate(network.iterAllSegments()):
        # plt.plot(seg.s[0],seg.s[1],'k.')
        # plt.plot(seg.t[0],seg.t[1],'k.')
        color = 'r' if seg.category=='scene_border' else 'y'

        for v1,v2 in zip(seg.path[:-1],seg.path[1:]):
            plt.plot( (v1[0], v2[0]), (v1[1], v2[1]), **RoadPolygonsRenderer.styles[seg.category], zorder=50 )
            # plt.plot( (v1[0], v2[0]), (v1[1], v2[1]), 'k', 0.5, zorder=50)


def plotPureNetwork(network,arrows=False,showIDs=False):
    from itertools import tee
    def pairs(iterable):
        # s -> (s0,s1), (s1,s2), (s2, s3), ...
        p1, p2 = tee(iterable)
        next(p2, None)
        return zip(p1,p2)

    for count,seg in enumerate(network.iterAllSegments()):
        color = 'g' if seg.category=='scene_border' else 'k'
        if arrows:
            width = 2
            for v0,v1 in pairs(seg.path):
                x = (v0[0]+v1[0])/2
                y = (v0[1]+v1[1])/2
                arrowprops=dict(color='r', width=width, shrink=0.05, headwidth=width*3, headlength=5*width)
                plt.gca().annotate("", xy=(x,y), xytext=(v0[0],v0[1]),arrowprops=arrowprops)
                plt.plot([v0[0],v1[0]],[v0[1],v1[1]],color=color, linewidth=width)
        else:
            plotLine(seg.path,False,color,1)
        plt.plot(seg.s[0],seg.s[1],'k.')
        plt.plot(seg.t[0],seg.t[1],'k.')
        if showIDs:
            c = sum(seg.path, Vector((0,0)))/len(seg.path)
            plt.text(c[0],c[1],'  '+str(seg.sectionId) )
            
def plotLine(line,vertsOrder,lineColor='k',width=1.,order=100):
    x = [n[0] for n in line]
    y = [n[1] for n in line]
    plt.plot(x,y,color=lineColor,linewidth=width,zorder=order)
    if vertsOrder:
        for i,(xx,yy) in enumerate(zip(x,y)):
            plt.text(xx,yy,str(i),fontsize=12)

def plotPolygon(poly,vertsOrder,lineColor='k',fillColor='k',width=1.,fill=False,alpha = 0.2,order=100):
    if not poly:
        return
    x = [n[0] for n in poly] + [poly[0][0]]
    y = [n[1] for n in poly] + [poly[0][1]]
    if fill:
        plt.fill(x[:-1],y[:-1],color=fillColor,alpha=alpha,zorder = order)
    plt.plot(x,y,color=lineColor,linestyle=':',linewidth=width,zorder=order)
    if vertsOrder:
        for i,(xx,yy) in enumerate(zip(x[:-1],y[:-1])):
            plt.text(xx,yy,str(i),fontsize=12)

def randomColor(n, name='hsv'):
    cmap = plt.cm.get_cmap(name, n)
    cmapList = [ cmap(i) for i in range(n)]
    import random
    random.shuffle(cmapList)
    i = 0
    while True:
        yield cmapList[i]
        i = (i+1)%n

def plotEnd():
    plt.gca().axis('equal')
    plt.show()
