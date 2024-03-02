from mathutils import Vector
from itertools import tee, islice, cycle
import matplotlib.pyplot as plt
from . import Renderer
from lib.CompGeom.BoolPolyOps import _isectSegSeg
from lib.CompGeom.PolyLine import PolyLine
from way.item import IntConnector, Section, SideLane, SymLane

def cyclePair(iterable):
    # iterable -> (p0,p1), (p1,p2), (p2, p3), ..., (pn, p0)
    prevs, nexts = tee(iterable)
    prevs = islice(cycle(prevs), len(iterable) - 1, None)
    return zip(prevs,nexts)

class StreetRenderer(Renderer):

    def __init__(self, debug):
        super().__init__()
        self.debug = debug
    
    def prepare(self):
        return
    
    def render(self, manager, data):
        for Id,isect in enumerate(manager.intersections):
            plotPolygon(isect.area,False,'r','r',2,True)
            if self.debug:
                c = sum(isect.area,Vector((0,0)))/len(isect.area)
                plt.text(c[0],c[1],str(isect.id),color='r',fontsize=18,zorder=130,ha='center', va='center', clip_on=True)
                for connector in IntConnector.iterate_from(isect.startConnector):
                    side = 'S' if connector.leaving else 'E'
                    p = isect.area[connector.index]
                    plt.text(p[0],p[1],str(abs(connector.item.id))+' '+side, clip_on=True)

                # for id, connector in isectArea.clusterConns.items():
                #     side = 'S' if id>0 else 'E'
                #     p = isectArea.polygon[connector]
                #     plt.text(p[0],p[1],'C'+str(abs(id))+' '+side)

                # # check for self-intersections
                def iterCircular(iterable):
                    A,B,C,D = tee(iterable, 4)
                    B = islice(cycle(B), 1, None)
                    C = islice(cycle(C), 2, None)
                    D = islice(cycle(D), 3, None)
                    return zip(A, B, C, D)
                polyIndx = range(len(isect.area)) 
                for a,b,c,d in iterCircular(polyIndx):
                    p = _isectSegSeg(isect.area[a],isect.area[b],isect.area[c],isect.area[d])
                    if p:
                        print( 'isect.polygon had self-intersection')
                        plt.plot(p[0],p[1],'ro',markersize=12)
                        plotPolygon(isect.area,True,'k','k',2,True,0.1,999)
                        break

        # processedWaySections = set()     
        # for transition in manager.transitionSideLanes:
        #     processedWaySections.update([abs(transition.ways[0]),abs(transition.ways[1])])
        #     way1 = transition.incoming#manager.waySectionLines[abs(transition.ways[0])]
        #     way2 = transition.outgoing#manager.waySectionLines[abs(transition.ways[1])]
        #     smallWay, wideWay = (way1, way2) if (way1.forwardLanes+way1.backwardLanes) < (way2.forwardLanes+way2.backwardLanes) else (way2, way1)
        #     plotSideLaneWay(smallWay,wideWay,'orange')
        #     if self.debug:
        #         center = sum(smallWay.centerline, Vector((0,0)))/len(smallWay.centerline)
        #         plt.text(center[0],center[1],str(abs(transition.ways[0])),color='blue',fontsize=14,zorder=120)
        #         center = sum(wideWay.centerline, Vector((0,0)))/len(wideWay.centerline)
        #         plt.text(center[0],center[1],str(abs(transition.ways[1])),color='blue',fontsize=14,zorder=120)

        # for key,symLane in enumerate(manager.transitionSymLanes):
        #     plotPolygon(symLane.polygon,False,'green','green',2,True,0.5)
        #     if self.debug:
        #         c = sum(symLane.polygon,Vector((0,0)))/len(symLane.polygon)
        #         plt.text(c[0],c[1],str(key),color='g',fontsize=18,zorder=130)


        for src, dst, multKey, street in manager.waymap.edges(data='object',keys=True):
            for item in street.iterItems():
                if isinstance(item,Section):
                    section = item
                    p1 = section.centerline[0]
                    plt.plot(p1[0],p1[1],'bo')
                    plt.text(p1[0],p1[1],' s: '+str(section.street.id),color='brown',fontsize=8,zorder=999, clip_on=True)
                    if not section.valid:
                        if self.debug:
                            center = sum(section.centerline, Vector((0,0)))/len(section.centerline)
                            plt.text(center[0],center[1],str(street.id),color='orange',fontsize=8,zorder=999, clip_on=True)
                            polyline = PolyLine(section.centerline)
                            width = section.width 
                            buffer = polyline.buffer(width/2,width/2)
                            plotPolygon(buffer,False,'r','r',1,True,0.9)
                        continue
                    if self.debug:
                        plotWay(section.centerline,False,'b',2.)
                        center = sum(section.centerline, Vector((0,0)))/len(section.centerline)
                        plt.text(center[0],center[1],str(street.id),color='blue',fontsize=14,zorder=120)
                    else:
                        center = sum(section.centerline, Vector((0,0)))/len(section.centerline)
                        plt.text(center[0],center[1],str(street.id),color='blue',fontsize=8,zorder=120, clip_on=True)
                        plotWay(section.centerline,False,'b',2.)
                        polyline = PolyLine(section.centerline)
                        width = section.width 
                        buffer = polyline.buffer(width/2,width/2)
                        plotPolygon(buffer,False,'k','b',1,True,0.1)
                else:
                    if isinstance(item,SideLane):
                        p = item.location
                        plt.plot(p[0],p[1],'cD',markersize=15,zorder=900)
                        plt.text(p[0],p[1],' s: '+str(item.street.id),color='brown',fontsize=8,zorder=999, clip_on=True)
                        if self.debug:
                            plt.text(p[0],p[1],str(item.id),color='b',fontsize=18,zorder=999,ha='center', va='center', clip_on=True)
                    elif isinstance(item,SymLane):
                        plotPolygon(item.area,False,'g','g',2,True)
                        # if self.debug:
                        p = item.location
                        plt.text(p[0],p[1],str(item.id),color='g',fontsize=18,zorder=999,ha='center', va='center', clip_on=True)
                        plt.text(p[0],p[1],'         s:'+str(item.street.id),color='brown',fontsize=8,zorder=999, clip_on=True)

                        # if not section.start:
                #     plt.plot(polyline[0][0],polyline[0][1],'cs',markersize=6)
                # if not section.end:
                #     plt.plot(polyline[-1][0],polyline[-1][1],'cs',markersize=6)

        # for Id,cluster in manager.wayClusters.items(): 
        #     from lib.CompGeom.PolyLine import PolyLine 
        #     centerlinClust = PolyLine(cluster.centerline)
        #     centerlinClust.plot('m',1)

        #     for sect in cluster.waySections:     
        #         way = centerlinClust.parallelOffset(-sect.offset)
        #         poly = way.buffer(sect.width/2.,sect.width/2.)
        #         plotPolygon(poly,False,'g','g',1.,True,0.3,120)

        #     poly = centerlinClust.buffer(cluster.distToLeft+cluster.waySections[0].width/2.,
        #                                  cluster.distToLeft+cluster.waySections[-1].width/2.)
        #     plotPolygon(poly,False,'c','c',1,True,0.2,110)

        #     if not cluster.startConnected:
        #         # centerlinClust.plot('r',4)
        #         plt.plot(cluster.centerline[0][0],cluster.centerline[0][1],'cs',markersize=6)
        #     if not cluster.endConnected:
        #         # centerlinClust.plot('r',4)
        #         plt.plot(cluster.centerline[-1][0],cluster.centerline[-1][1],'cs',markersize=6)

        #     if self.debug:
        #         plotWay(cluster.centerline,False,'b',2.)
        #         center = sum(cluster.centerline, Vector((0,0)))/len(cluster.centerline)
        #         plt.text(center[0],center[1],str(Id),fontsize=22,zorder=130)


        # if self.debug:
        #     # Check connector IDs and counter-clockwise order
        #     for isectArea in manager.intersectionAreas:
        #         for id, connector in isectArea.connectors.items():
        #             if abs(id) not in manager.waySectionLines:
        #                 print('missing id in waySectionLines',id)
        #                 v0,v1 = isectArea.polygon[connector:connector+2]
        #                 plt.plot([v0[0],v1[0]],[v0[1],v1[1]],'c',linewidth=10,zorder=999)
        #                 # plotPolygon(isectArea.polygon,False,'c','c',2,True,0.4,999)
        #         area = sum( (p2[0]-p1[0])*(p2[1]+p1[1]) for p1,p2 in cyclePair(isectArea.polygon))
        #         if area >= 0.:
        #             print('intersectionArea not counter-clockwise')
        #             plotPolygon(isectArea.polygon,True,'c','c',10,True,0.4,999)
        #         for id, connector in isectArea.clusterConns.items():
        #             if abs(id) not in manager.wayClusters:
        #                 print('missing id in wayClusters',id)
        #                 plotPolygon(isectArea.polygon,False,'g','g',10,True,0.4,999)
        #     # Check connections at ends
        #     for sectionNr,section_gn in manager.waySectionLines.items():
        #         if section_gn.startConnected is None:
        #             print('no endConnected in section',sectionNr)
        #         if section_gn.startConnected is None:
        #             print('no endConnected in section',sectionNr)
        #     for Id,cluster in manager.wayClusters.items():
        #         if cluster.startConnected is None:
        #             print('no endConnected in cluster',Id)
        #         if cluster.startConnected is None:
        #             print('no endConnected in cluster',Id)




def plotPolygon(poly,vertsOrder,lineColor='k',fillColor='k',width=1.,fill=False,alpha = 0.2,order=100):
    x = [n[0] for n in poly] + [poly[0][0]]
    y = [n[1] for n in poly] + [poly[0][1]]
    if fill:
        plt.fill(x[:-1],y[:-1],color=fillColor,alpha=alpha,zorder = order)
    plt.plot(x,y,lineColor,linewidth=width,zorder=order)
    if vertsOrder:
        for i,(xx,yy) in enumerate(zip(x[:-1],y[:-1])):
            plt.text(xx,yy,str(i),fontsize=12, clip_on=True)

def plotWay(way,vertsOrder,lineColor='k',width=1.,order=100):
    x = [n[0] for n in way]
    y = [n[1] for n in way]
    plt.plot(x,y,lineColor,linewidth=width,zorder=order)
    if vertsOrder:
        for i,(xx,yy) in enumerate(zip(x,y)):
            plt.text(xx,yy,str(i),fontsize=12, clip_on=True)

def plotSideLaneWay(smallWay,wideWay,color):
    from lib.CompGeom.PolyLine import PolyLine
    smallLine = PolyLine(smallWay.centerline)
    wideLine = PolyLine(wideWay.centerline)
    smallLine.plotWithArrows('g',2)
    wideLine.plotWithArrows('r',2)

    wideLine = wideLine.parallelOffset(-wideWay.offset)
    poly = wideLine.buffer(wideWay.width/2.,wideWay.width/2.)
    plotPolygon(poly,False,color,color,1.,True,0.3,120)
    poly = smallLine.buffer(smallWay.width/2.,smallWay.width/2.)
    plotPolygon(poly,False,color,color,1.,True,0.3,120)

def plotEnd():
    plt.gca().axis('equal')
    plt.show()