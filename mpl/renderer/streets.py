from mathutils import Vector
import matplotlib.pyplot as plt
from . import Renderer


class StreetRenderer(Renderer):
    
    def __init__(self, debug):
        super().__init__()
        self.debug = debug
    
    def prepare(self):
        return
    
    def render(self, manager, data):
        for isectArea in manager.intersectionAreas:
            plotPolygon(isectArea.polygon,False,'r','r',2,True)
            if self.debug:
                for id, connector in isectArea.connectors.items():
                    p = isectArea.polygon[connector[0]]
                    plt.text(p[0],p[1],str(id)+' '+connector[1])
                for id, connector in isectArea.clusterConns.items():
                    p = isectArea.polygon[connector[0]]
                    plt.text(p[0],p[1],'C'+str(id)+' '+connector[1]+' '+str(connector[2]))

        for sectionNr,section_gn in manager.waySectionLines.items():
            if self.debug:
                plotWay(section_gn.centerline,True,'b',2.)
                center = sum(section_gn.centerline, Vector((0,0)))/len(section_gn.centerline)
                plt.text(center[0],center[1],str(sectionNr),zorder=120)
            else:
                plotWay(section_gn.centerline,False,'b',2.)
                from lib.CompGeom.PolyLine import PolyLine
                polyline = PolyLine(section_gn.centerline)
                width = section_gn.width 
                buffer = polyline.buffer(width/2,width/2)
                plotPolygon(buffer,False,'k','b',1,True,0.1)

        for Id,cluster in manager.wayClusters.items(): 
            from lib.CompGeom.PolyLine import PolyLine 
            centerlinClust = PolyLine(cluster.centerline)
            centerlinClust.plot('m',1)         

            leftWay = centerlinClust.parallelOffset(cluster.distToLeft)
            leftDesc = cluster.wayDescriptors[0]
            leftPoly = leftWay.buffer(leftDesc[1]/2.,leftDesc[1]/2.)
            plotPolygon(leftPoly,False,'g','g',1.,True,0.3,120)

            rightWay = centerlinClust.parallelOffset(-cluster.distToLeft)
            rightDesc = cluster.wayDescriptors[1]
            rightPoly = rightWay.buffer(rightDesc[1]/2.,rightDesc[1]/2.)
            plotPolygon(rightPoly,False,'g','g',1.,True,0.3,120)

            poly = centerlinClust.buffer(cluster.distToLeft+leftDesc[1]/2,cluster.distToLeft+rightDesc[1]/2.)
            plotPolygon(poly,False,'c','c',1,True,0.2,110)

            if self.debug:
                plotWay(cluster.centerline,False,'b',2.)
                center = sum(cluster.centerline, Vector((0,0)))/len(cluster.centerline)
                plt.text(center[0],center[1],str(Id),fontsize=22,zorder=130)



def plotPolygon(poly,vertsOrder,lineColor='k',fillColor='k',width=1.,fill=False,alpha = 0.2,order=100):
    x = [n[0] for n in poly] + [poly[0][0]]
    y = [n[1] for n in poly] + [poly[0][1]]
    if fill:
        plt.fill(x[:-1],y[:-1],color=fillColor,alpha=alpha,zorder = order)
    plt.plot(x,y,lineColor,linewidth=width,zorder=order)
    if vertsOrder:
        for i,(xx,yy) in enumerate(zip(x[:-1],y[:-1])):
            plt.text(xx,yy,str(i),fontsize=12)

def plotWay(way,vertsOrder,lineColor='k',width=1.,order=100):
    x = [n[0] for n in way]
    y = [n[1] for n in way]
    plt.plot(x,y,lineColor,linewidth=width,zorder=order)
    if vertsOrder:
        for i,(xx,yy) in enumerate(zip(x,y)):
            plt.text(xx,yy,str(i),fontsize=12)

def plotEnd():
    plt.gca().axis('equal')
    plt.show()