from mathutils import Vector
import numpy as np
from itertools import tee, islice, cycle
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from . import Renderer
from lib.CompGeom.BoolPolyOps import _isectSegSeg
from lib.CompGeom.PolyLine import PolyLine
from lib.CompGeom.algorithms import circumCircle
from way.item import Street, Bundle, Intersection, Section, SideLane, SymLane

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
        def isSmallestCategory(section):
             return  section.category in  ['footway', 'cycleway']
        
        def isMinorCategory(section):
            return  section.category in  ['footway', 'cycleway','service']
        
        for location,isect in manager.majorIntersections.items():
            p = location
            plt.plot(p[0],p[1],'ro',markersize=5,zorder=999,markeredgecolor='red', markerfacecolor='orange')
            if self.debug:
                plt.text(p[0],p[1],' '+str(isect.id),color='r',fontsize=10,zorder=130,ha='left', va='top', clip_on=True)
            if isect.connectsBundles:
                ends = []
                nrOfBundles = 0
                for intSec in isect:
                    if isinstance(intSec.item,Street):
                        ends.append( intSec.item.src if intSec.leaving else intSec.item.dst )
                    elif isinstance(intSec.item,Bundle):
                        ends.extend( intSec.item.headLocs if intSec.leaving else intSec.item.tailLocs)
                        nrOfBundles += 1
                center,radius = circumCircle(list(ends))
                color = 'green' if nrOfBundles>1 else 'orange'
                circle = plt.Circle(center, radius*1.1, color=color, alpha=0.6)
                plt.gca().add_patch(circle)
                

        for location,isect in manager.minorIntersections.items():
            p = location
            plt.plot(p[0],p[1],'cv',markersize=5,zorder=999,markeredgecolor='cyan', markerfacecolor='cyan')
            if self.debug:
                plt.text(p[0],p[1],'  '+str(isect.id),color='c',fontsize=6,zorder=130,ha='left', va='top', clip_on=True)

        # DEBUG: Plot streets from waymap
        # for src, dst, multKey, street in manager.waymap.edges(data='object',keys=True):
        #     allVertices = []
        #     for item in street.iterItems():
        #         if isinstance(item,Section):
        #             allVertices.extend(item.centerline)
        #             item.polyline.plot('b',1,'dotted')
        #     if len(allVertices):
        #         c = sum(allVertices,Vector((0,0))) / len(allVertices)
        #         plt.text(c[0],c[1],'S '+str(street.id),color='blue',fontsize=6,zorder=130,ha='left', va='top', clip_on=True)
        # return

        for street in manager.iterStreets():
            if isinstance(street,Street):
                allVertices = []
                interiorOfBundle = street.bundle is not None
                if self.debug:
                    color = 'gray' if isSmallestCategory(street.head) else 'b' if isMinorCategory(street.head) else 'r'
                    fontsize = 6 if isSmallestCategory(street.head) else 8 if isMinorCategory(street.head) else 10
                    upset = 0 if isSmallestCategory(street.head) else 1 if isMinorCategory(street.head) else 2
                    srcVec, _ = street.endVectors()
                    vu = srcVec/srcVec.length * (1+upset)
                    p = street.src
                    plt.text(p[0]+vu[0],p[1]+vu[1],'   S'+str(street.id),fontsize=fontsize,color=color)
                for item in street.iterItems():

                    if isinstance(item,Section):
                        section = item
                        allVertices.extend(section.centerline)
                        if section.valid:
                            color = 'gray' if isSmallestCategory(section) else 'b' if isMinorCategory(section) else 'r'
                            width = 0.8 if isSmallestCategory(section) else 1 if isMinorCategory(section) else 1.2
                            style = 'dotted' if isSmallestCategory(section) else '--' if isMinorCategory(section) else 'solid'
                            if interiorOfBundle:
                                color = 'gold'
                                style='solid'
                                width = width+0.5
                            section.polyline.plotWithArrows(color,width,0.5,style,False,950)
                            # if self.debug:
                            #     p = section.src
                            #     plt.text(p[0],p[1]+1,' s'+str(section.id),fontsize=10,color='red')

                    if isinstance(item,Intersection):
                        if not item.isMinor:
                            print('Must be minor when in Street!!!! street.id', street.id,' item.id ',item.id)
                            continue
                        p = item.location
                        plt.plot(p[0],p[1],'cv',markersize=5,zorder=999,markeredgecolor='green', markerfacecolor='none')
                        if self.debug:
                            plt.text(p[0],p[1],'  M '+str(item.id),color='g',fontsize=10,zorder=130,ha='left', va='top', clip_on=True)

                        if False:#self.debug:
                            for conn in Intersection.iterate_from(item.leftHead):
                                line = conn.item.head.polyline if conn.leaving else conn.item.tail.polyline
                                vec = line[1]-line[0] if conn.leaving else line[-2]-line[-1]
                                vec = vec/vec.length
                                p0 = line[0] if conn.leaving else line[-1]
                                p1 = p0 +3*vec
                                plt.plot([p0[0],p1[0]], [p0[1],p1[1]], 'g')
                                if self.debug:
                                    plt.text(p1[0]+2,p1[1]-2,'C '+str(conn.id),color='k',fontsize=8,zorder=130,ha='left', va='top', clip_on=True)
                            for conn in Intersection.iterate_from(item.rightHead):
                                line = conn.item.head.polyline if conn.leaving else conn.item.tail.polyline
                                vec = line[1]-line[0] if conn.leaving else line[-2]-line[-1]
                                vec = vec/vec.length
                                p0 = line[0] if conn.leaving else line[-1]
                                p1 = p0 +3*vec
                                plt.plot([p0[0],p1[0]], [p0[1],p1[1]], 'r')
                                if self.debug:
                                    plt.text(p1[0]+2,p1[1]-2,'C '+str(conn.id),color='k',fontsize=8,zorder=130,ha='left', va='top', clip_on=True)

                    if isinstance(item,SideLane):
                        p = item.location
                        plt.plot(p[0],p[1],'rs',markersize=6,zorder=999,markeredgecolor='green', markerfacecolor='cyan')
                        if self.debug:
                            plt.text(p[0]+2,p[1]-2,'Side '+str(item.id),color='k',fontsize=8,zorder=130,ha='left', va='top', clip_on=True)

                    if isinstance(item,SymLane):
                        p = item.location
                        plt.plot(p[0],p[1],'rP',markersize=6,zorder=999,markeredgecolor='green', markerfacecolor='cyan')
                        if self.debug:
                            plt.text(p[0]+2,p[1]-2,'Sym '+str(item.id),color='k',fontsize=8,zorder=130,ha='left', va='top', clip_on=True)


                # color = 'crimson'
                # width =  10
                # if len(allVertices):
                #     c = sum(allVertices,Vector((0,0))) / len(allVertices)
                #     if self.debug:
                #         plt.text(c[0]+2,c[1]-2,' S '+str(street.id),color=color,fontsize=width,zorder=130,ha='left', va='top', clip_on=True)

            else:
                print('Unknown object: ', type(street))

        for bundle in manager.iterBundles():
            vertices = []
            for street in bundle.streetsHead:
                p = street.head.polyline[1]
                plt.text(p[0],p[1]+1.5,'   S'+str(street.id),fontsize=10,color='green',ha='right', va='bottom')
                for item in street.iterItems():
                    if isinstance(item,Section):
                        section = item
                        if section.valid:
                            color = 'gray' if isSmallestCategory(section) else 'g' if isMinorCategory(section) else 'g'
                            width = 1 if isSmallestCategory(section) else 1.1 if isMinorCategory(section) else 1.5
                            style = 'dotted' if isSmallestCategory(section) else '--' if isMinorCategory(section) else 'solid'
                            upset = 0 if isSmallestCategory(section) else 1 if isMinorCategory(section) else 2
                            section.polyline.plotWithArrows(color,width,0.5,style,False,950)
                            vertices.extend(section.centerline)

            for street in bundle.streetsTail:
                if street in bundle.streetsHead:
                    continue# already drawn
                p = street.head.polyline[1]
                plt.text(p[0],p[1]+1.5,'   S'+str(street.id),fontsize=10,color='green',ha='right', va='bottom')
                for item in street.iterItems():
                    if isinstance(item,Section):
                        section = item
                        if section.valid:
                            color = 'gray' if isSmallestCategory(section) else 'g' if isMinorCategory(section) else 'g'
                            width = 1 if isSmallestCategory(section) else 1.1 if isMinorCategory(section) else 1.5
                            style = 'dotted' if isSmallestCategory(section) else '--' if isMinorCategory(section) else 'solid'
                            upset = 0 if isSmallestCategory(section) else 1 if isMinorCategory(section) else 2
                            section.polyline.plotWithArrows(color,width,0.5,style,False,950)
                            vertices.extend(section.centerline)

            c = sum(vertices,Vector((0,0)))/len(vertices)
            plt.text(c[0],c[1]+upset,str(bundle.id),fontsize=18,color='red',ha='center', va='center')
            



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