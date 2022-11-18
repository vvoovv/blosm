from mathutils import Vector
from itertools import tee

from defs.way_cluster_params import minTemplateLength, minNeighborLength, searchDist, canPair
from lib.CompGeom.StaticSpatialIndex import StaticSpatialIndex, BBox
from lib.CompGeom.GraphBasedAlgos import DisjointSets
from lib.CompGeom.LinePolygonClipper import LinePolygonClipper

debug = []
if debug:
    import matplotlib.pyplot as plt

# helper functions -----------------------------------------------
def pairs(iterable):
    # iterable -> (p0,p1), (p1,p2), (p2, p3), ...
    p1, p2 = tee(iterable)
    next(p2, None)
    return zip(p1,p2)

def triples(iterable):
    # iterable -> (p0,p1,p2), (p1,p2,p3), (p2,p3,p4), ...
    p1, p2, p3 = tee(iterable,3)
    next(p2, None)
    next(p3, None)
    next(p3, None)
    return zip(p1,p2,p3)

def maxAngleTan(polyline):
    # computes the tanget of the maximum angle between polyline segments
    if len(polyline) < 3:
        return 0.
    maxtan = 0.
    for p1, p2, p3 in triples(polyline):
        d1, d2 = p2-p1, p3-p2
        cross = d1.cross(d2)
        dot = d1.dot(d2)
        tan_angle = abs(cross/dot) if dot else float('inf')
        if tan_angle > maxtan:
            maxtan = tan_angle
    return maxtan
 # ----------------------------------------------------------------


def findWayCluster(cls,waySections):
    print('start')

    # Create spatial index (R-tree) from way sections
    spatialIndex = StaticSpatialIndex()
    indx2Id = dict()
    boxes = dict()
    for Id, section in waySections.items():
        min_x = min(v[0] for v in section.polyline.verts)
        min_y = min(v[1] for v in section.polyline.verts)
        max_x = max(v[0] for v in section.polyline.verts)
        max_y = max(v[1] for v in section.polyline.verts)
        bbox = BBox(None,min_x,min_y,max_x,max_y)
        index = spatialIndex.add(min_x,min_y,max_x,max_y)
        indx2Id[index] = Id
        bbox.index = index
        boxes[Id] = (min_x,min_y,max_x,max_y)
    spatialIndex.finish()

    parallelWaysIndxs = DisjointSets()
    for Id, section in waySections.items():
        # if Id not in [778,814]:
        #     continue

        # section.polyline.plot('k',1)
        if 1 in debug:
            center = sum(section.polyline.verts,Vector((0,0)))/len(section.polyline)
            plt.text(center[0],center[1],str(Id),zorder=999)

        # Use only ways of minimal length as template.
        if section.polyline.length() > minTemplateLength:
            # Expand the template way, <expandBy> is some kind of search range.
            templateCategory = section.originalSection.category
            expandBy = searchDist[templateCategory]
            templatePoly = section.polyline.buffer(expandBy,expandBy)
            # section.polyline.plot('k',0.5)

            # Create polygon clipper from expanded template way
            clipper = LinePolygonClipper(templatePoly.verts)

            # Get neighbors of template way from static spatial index with a search
            # bounding box expanded by the same width as the template way is expanded.
            min_x,min_y,max_x,max_y = boxes[Id]
            results = stack = []
            result = spatialIndex.query(min_x-expandBy,min_y-expandBy,max_x+expandBy,max_y+expandBy,results,stack)

            # Check all neighbor way-sections
            for indx in result:
                if indx2Id[indx] == Id: continue    # don't check template way-section

                # Check if the pairing is allowed
                neighborCategory = waySections[indx2Id[indx]].originalSection.category
                if not canPair[templateCategory][neighborCategory]: continue

                # Clip the neighbor line with the template polygon
                neighborLine = waySections[indx2Id[indx]].polyline
                if neighborLine.length() > minNeighborLength:
                    fragments, fragsLength, nrOfON = clipper.clipLine(neighborLine.verts)

                    if fragsLength == 0.:
                        continue # nothing inside the polygon

                    # Evaluate the "slope" relative to the template's line
                    p1, d1 = section.polyline.distTo(fragments[0][0])     # distance to start of fragment
                    p2, d2 = section.polyline.distTo(fragments[-1][-1])   # distance to end of fragment
                    p_dist = (p2-p1).length
                    slope = abs(d1-d2)/p_dist if p_dist else 1.

                    reasonText = 'slope: %s, ends: %s, ON: %s, frags %s'%(slope < 0.15, min(d1,d2) <= expandBy, nrOfON <= 2, fragsLength > expandBy/2)
                    conditions = slope < 0.15 and min(d1,d2) <= expandBy and  nrOfON <= 2 and fragsLength > expandBy/2
                    if conditions:
                        parallelWaysIndxs.addSegment(Id,indx2Id[indx])

                        if 2 in debug:
                            print( fragsLength , expandBy/2)
                            if slope < 0.1 and min(d1,d2) <= expandBy and nrOfON <= 2:# and fragsLength > expandBy/2. :
                                plotNetwork(cls.sectionNetwork, waySections)
                                plt.plot(p1[0],p1[1],'ro')
                                plt.plot(p2[0],p2[1],'ro')
                                plotPolygon(templatePoly.verts,False,'k','k',0.5)
                                section.polyline.plot('r',2)
                                # neighborLine.plot('b',2)
                                for f in fragments:
                                    plotLine([f[0],f[1]],False,'r',2)
                                plt.title('OK ' + reasonText )
                                plotEnd()
                    else:
                        if 2 in debug:
                            print( fragsLength , expandBy/2)
                            plotNetwork(cls.sectionNetwork, waySections)
                            plotPolygon(templatePoly.verts,False,'k','k',0.5)
                            section.polyline.plot('b',4)
                            neighborLine.plot('k',2)
                            for f in fragments:
                                plotLine([f[0],f[1]],False,'b',4)
                            plt.title('BAD ' + reasonText )
                            plotEnd()

    if 3 in debug:
        for section in waySections.values():
            section.polyline.plot('k')
        import numpy as np
        N = len([w for w in parallelWaysIndxs])
        import matplotlib.cm as cm
        from math import pi
        colors = cm.prism(np.linspace(0, 1, N))#int(pi*2.5*N)))
        # import random
        # random.shuffle(colors)
        for pW, color in zip(parallelWaysIndxs,colors):
            # for section in waySections.values():
            #     section.polyline.plot('k')
            for id in pW:
                section = waySections[id]
                section.polyline.plot(color,4)
            # plotEnd()

        # import os
        # plt.title(os.path.basename(wayManager.app.osmFilepath))

    if 4 in debug:
        for clusterIDs in parallelWaysIndxs:
            partialIntersections(cls,clusterIDs, waySections)
        #     break

def partialIntersections(cls,clusterIDs, waySections):
    pass
    # plotNetwork(cls.sectionNetwork, waySections)
    for i,Id in enumerate(clusterIDs):
        waySections[Id].polyline.plot('b',2)
        oS = waySections[Id].originalSection
        marker = ['ro', 'cx', 'bo', 'bo', 'bo', 'bo', 'bo', 'bo', 'bo', 'bo', 'bo', 'bo', 'bo', 'bo', ]
        for p in [oS.s,oS.t]:
            order = cls.sectionNetwork.borderlessOrder(p)
            v = p
            plt.plot(v.x,v.y,marker[order-1],markersize=6,zorder=999)
            plt.text(v.x,v.y,'  '+str(order),fontsize=12)

    plotEnd()


def constructCluster(clusterIDs, waySections):
    # find projection line along cluster using polyline of longest section
    lengths_IDs = [(Id,waySections[Id].polyline.length()) for Id in clusterIDs]
    longestID, length = max(lengths_IDs,key=lambda x: x[1])

    # find noraml to line between endpoints of longest line
    # plotLine([waySections[longestID].polyline[-1],waySections[longestID].polyline[0]],False,'r',2)
    end0, end1 = waySections[longestID].polyline[-1], waySections[longestID].polyline[0]
    endVec = (end1-end0)
    normVec = Vector((-endVec[1],endVec[0]))

    # For every vertex in the clustered way-sections find intersection points along normal
    # with other polylines.
    centerline = []
    for Id in clusterIDs:
        verts = waySections[Id].polyline.verts
        for v in verts:
            isects = [v]
            p1,p2 = v, v+normVec
            # plt.plot(v[0],v[1],'bo')
            # plotLine([p1,p2],False,'r:')
            for subID in clusterIDs:
                if subID == Id:
                    continue
                waySections[Id].polyline.plot('k')
                waySections[subID].polyline.plot('g')
                for p3,p4 in pairs(waySections[subID].polyline):
                    # compute intersection point
                    d1, d2 = p2-p1, p4-p3
                    cross = d1.cross(d2)
                    if cross == 0.:
                        continue
                    d3 = p1-p3
                    # t1 = (d2[0]*d3[1] - d2[1]*d3[0])/cross
                    t2 = (d1[0]*d3[1] - d1[1]*d3[0])/cross
                    if 0. <= t2 <= 1.:
                        v0 = p3 + d2*t2
                        isects.append(v0)
                        plt.plot(v0[0],v0[1],'rx')
                        # print(t1,t2)
                    test = 1
            center = sum(isects,Vector((0,0)))/len(isects)
            centerline.append(center)
            # plt.plot(center[0],center[1],'r.',markersize=8)

    for Id in clusterIDs:
        waySections[Id].polyline.plot('k',0.5)
    for center in centerline:
        plt.plot(center[0],center[1],'r.')#,markersize=8)
    plotEnd()
    test = 1

def plotNetwork(network,waySections):
    from mpl.renderer.road_polygons import RoadPolygonsRenderer
    for section in waySections.values():
        seg = section.originalSection
    # for count,seg in enumerate(network.iterAllSegments()):
        # plt.plot(seg.s[0],seg.s[1],'k.')
        # plt.plot(seg.t[0],seg.t[1],'k.')
        color = 'r' if seg.category=='scene_border' else 'y'

        for v1,v2 in zip(seg.path[:-1],seg.path[1:]):
            plt.plot( (v1[0], v2[0]), (v1[1], v2[1]), **RoadPolygonsRenderer.styles[seg.category], zorder=50 )


def plotPolygon(poly,vertsOrder,lineColor='k',fillColor='k',width=1.,fill=False,alpha = 0.2,order=100):
    x = [n[0] for n in poly] + [poly[0][0]]
    y = [n[1] for n in poly] + [poly[0][1]]
    if fill:
        plt.fill(x[:-1],y[:-1],color=fillColor,alpha=alpha,zorder = order)
    plt.plot(x,y,lineColor,linewidth=width,zorder=order)
    if vertsOrder:
        for i,(xx,yy) in enumerate(zip(x[:-1],y[:-1])):
            plt.text(xx,yy,str(i),fontsize=12)

def plotLine(line,vertsOrder,lineColor='k',width=1.,order=100):
    x = [n[0] for n in line]
    y = [n[1] for n in line]
    plt.plot(x,y,lineColor,linewidth=width,zorder=order)
    if vertsOrder:
        for i,(xx,yy) in enumerate(zip(x[:-1],y[:-1])):
            plt.text(xx,yy,str(i),fontsize=12)

def plotEnd():
    plt.gca().axis('equal')
    plt.show()
