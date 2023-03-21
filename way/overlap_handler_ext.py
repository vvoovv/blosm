from lib.CompGeom.centerline import pointInPolygon

# helper functions -----------------------------------------------
def cyclePair(iterable):
    # iterable -> (p0,p1), (p1,p2), (p2, p3), ..., (pn, p0)
    prevs, nexts = tee(iterable)
    prevs = islice(cycle(prevs), len(iterable) - 1, None)
    return zip(prevs,nexts)

def pairs(iterable):
    # iterable -> (p0,p1), (p1,p2), (p2, p3), ...
    p1, p2 = tee(iterable)
    next(p2, None)
    return zip(p1,p2)

# Adapted from https://stackoverflow.com/questions/16542042/fastest-way-to-sort-vectors-by-angle-without-actually-computing-that-angle
# Input:  d: vector.
# Output: a number from the range [0 .. 4] which is monotonic
#         in the angle this vector makes against the x axis.
def pseudoangle(d):
    p = d[0]/(abs(d[0])+abs(d[1])) # -1 .. 1 increasing with x
    return 3 + p if d[1] < 0 else 1 - p 
# ----------------------------------------------------------------

from osmPlot import *

def areaFromEndClusters(cls,ovH):
    # ovH is the overlapHandler

    nodeOutWays = []
    for isect,inID in zip(ovH.outIsects,ovH.wIDs):
        nodeOutWays.append( NodeOutWays(cls,ovH,isect,inID) )
    plotEnd()
    pass


class NodeOutWays():
    ID = 0
    def __init__(self,cls,ovH,isect,inID):
        self.id = NodeOutWays.ID
        NodeOutWays.ID += 1
        self.cls = cls
        self.ovH = ovH

        # Get the IDs of all way-sections that leave this node.
        node = isect.freeze()
        outIds = self.getOutSectionIds(node,inID)

        # If it was an end-node, there are no out-ways. Invalidate this instance.
        if not outIds:
            self.valid = False
            return

        # Create vectors to the right and to the left of the outGoing inID
        s = cls.waySections 

        # Filter by direction of first segments.
        outVInID = self.outVector(node,inID)
        perpVL = Vector((outVInID[1],-outVInID[0]))/outVInID.length
        perpVR = Vector((-outVInID[1],outVInID[0]))/outVInID.length
        outVectors = [self.outVector(node, Id) for Id in outIds+[inID]]

        # Sort
        allVectors = outVectors + [outVInID,perpVL,perpVR]
        # outWays = sorted(allVectors,key=lambda x: pseudoangle(x.polyline[1]-x.polyline[0]) )
        outWays = sorted(allVectors,key=pseudoangle)
        perpVLIndx = outWays.index(perpVL)
        perpVRIndx = outWays.index(perpVR)
        inIndx = outWays.index(outVInID)
        test=1


        poly = self.ovH.subCluster.centerline.buffer(self.ovH.subCluster.outWL(),self.ovH.subCluster.outWR())
        # plotPolygon(poly,False,'g','g',1,True)
        # cls.waySections[inID].polyline.plot('r',3,True)
        # for Id in outIds:
        #     cls.waySections[Id].polyline.plot('k')
        # plotEnd()

        # Filter out ways that are inside the cluster polygon
        filteredIDs = []
        for Id in outIds:
            section = cls.waySections[Id]
            out0 = pointInPolygon(poly,section.polyline[0]) == 'OUT'
            out1 = pointInPolygon(poly,section.polyline[-1]) == 'OUT'
            if out0 or out1:
                filteredIDs.append(Id)

        plotPolygon(poly,False,'g','g',1,True)
        cls.waySections[inID].polyline.plot('r',3,True)
        for Id in filteredIDs:
            cls.waySections[Id].polyline.plot('k')
        # plotEnd()

    def outVector(self,node, wayID):
        s = self.cls.waySections
        outV = s[wayID].polyline[1]-s[wayID].polyline[0] if s[wayID].polyline[0] == node else s[wayID].polyline[-2]-s[wayID].polyline[-1]
        return outV



    # Find the IDs of all way-sections that leave the <node>, but are not inside the way-cluster.
    def getOutSectionIds(self,node,inID):
        outSectionIDs = []
        if node in self.cls.sectionNetwork:
            for net_section in self.cls.sectionNetwork.iterOutSegments(node):
                if net_section.category != 'scene_border':
                    if net_section.sectionId != inID:
                        # If one of the nodes is outside of the clusters ...
                        if not (net_section.s in self.ovH.outIsects and net_section.t in self.ovH.outIsects):                               
                            outSectionIDs.append(net_section.sectionId)
                        else: # ... else we have to check if this section is inside or outside of the way-cluster.
                            nonNodeVerts = net_section.path[1:-1]
                            if len(nonNodeVerts): # There are vertices beside the nodes at the ends.
                                # Check, if any of these vertices is outside of the way-cluster
                                clusterPoly = self.ovH.subCluster.centerline.buffer(self.ovH.subCluster.endWidth/2.,self.ovH.subCluster.endWidth/2.)
                                if any( pointInPolygon(clusterPoly,vert) == 'OUT' for vert in nonNodeVerts ):
                                    outSectionIDs.append(net_section.sectionId) # There are, so keep this way ...
                                    continue
                                # ... else invalidate this way
                                self.cls.waySections[net_section.sectionId].isValid = False
        return outSectionIDs    

