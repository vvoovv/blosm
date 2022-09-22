from mathutils import Vector
from functools import cmp_to_key
from math import atan2, pi

from defs.way import allRoadwayCategoriesRank

PI2 = 2.*pi

class NetSection():
    ID = 0  # just used during debugging

    def __init__(self, *args):
        self.ID = NetSection.ID 
        self.sectionId = self.ID
        NetSection.ID += 1      
        if len(args) > 1:
            self.initFromDetails(*args)
        else:
            self.initFromOther(*args)

    def initFromDetails(self, source, target, category, tags, length=None, path=None):
        self.s = Vector(source).freeze()    # source node
        self.t = Vector(target).freeze()    # target node
        self.forward = True
        self.category = category
        self.tags = tags
        self.geomLength = (self.t-self.s).length
        if length:
            self.length = length
        else:
            self.length = self.geomLength
        if path:
            self.path = path        # includes source and target
        else:
            self.path = [self.s,self.t]
        self.firstV = self.path[1]-self.path[0]  # vector of first way-segment in path
        self.oneWay = 0
        if self.tags and 'oneway' in self.tags:
            if self.tags['oneway'] == 'true':
                self.oneWay = 1

    def initFromOther(self, other):
        self.s = other.s    # source node
        self.t = other.t    # target node
        self.forward = other.forward
        self.category = other.category
        self.tags = other.tags
        self.geomLength = other.geomLength
        self.length = other.length
        self.path = other.path        # includes source and target
        self.firstV = other.firstV
        self.oneWay = other.oneWay

    def join(self, segment):
        assert self.category == segment.category, "segment to join must have same category"
        if self.t == segment.s:
            self.length += segment.length
            self.path += segment.path[1:]
        else:
            self.length += segment.length + (segment.s-self.t).length
            self.path += segment.path[1:]
        self.t = segment.t
        self.geomLength = (self.t-self.s).length

    def iterPathPair(self):
        path = self.path
        for i in range(len(path)-1):
            yield path[i], path[i+1]

    def __invert__(self):
        # create segment with the reversed direction
        rev = self.__class__(self.t, self.s, self.category, self.tags, self.length, self.path[::-1])
        rev.oneWay = -self.oneWay
        rev.forward = not self.forward
        rev.sectionId = self.sectionId
        return rev
    def __eq__(self, other):
        # comparison of segments (no duplicates allowed)
        return self.category == other.category and self.path == other.path
    def __hash__(self):
        return hash((self.s, self.t, self.length))

class WayNetwork(dict):
    # undirected multigraph
    def __init__(self):
        # only used in search for cycles
        self.counterClockEmbedding = None

    def addNode(self,node):
        # add a <node> to the network, node must be a frozen Vector!!
        self.setdefault( node, dict() )

    def delNode(self, node):
        # remove a <node> from the graph
        node = Vector(node).freeze()
        for segment in list(self.iterInSegments(node)):
            self.delSegment(segment)
        self.pop(node, None)
        # del self[node]

    def hasNode(self, node):
        # test if an <node> exists
        node = Vector(node).freeze()
        return node in self

    def addSegment(self, segment, allowIdenticalSegments=True):
        # add a <segment> to the network
        s, t = segment.s, segment.t

        if not allowIdenticalSegments:
            # check if two ways share the same segment,
            # if yes, keep the one with the lowest category rank 
            if s in self and t in self[s]:
                segmentList = self[s][t]
                if segment in segmentList:                  
                    thisIndex = segmentList.index(segment)
                    oldSegment = segmentList[thisIndex]
                    if allRoadwayCategoriesRank[segment.category] < allRoadwayCategoriesRank[oldSegment.category]:
                        segmentList[thisIndex] = segment
                    return

        self.addNode(s)
        self.addNode(t)
        self[s].setdefault(t, list() ).append(segment)
        # if s != t:  # a loop is added only once
        self[t].setdefault(s, list() ).append(~segment)

    def delSegment(self,segment):
        # remove a <segment> from the network
        if segment.s in self and segment.t in self[segment.s]:
            self[segment.s][segment.t].remove(segment)
            if len(self[segment.s][segment.t]) == 0:
                del self[segment.s][segment.t]
            # a loop is deleted only once
            if segment.s != segment.s:
                self[segment.t][segment.s].remove(~segment)
                if len(self[segment.t][segment.s]) == 0:
                    del self[segment.t][segment.s]

    def getSegment(self, source, target):
        if source in self and target in self[source]:
            segment = self[source][target]
            return segment if isinstance(segment,list) else segment
        else:
            return None        
       
    def iterNodes(self):
        # generator for all nodes from the network
        return iter(self)

    def order(self,source):
        return sum(1 for target in self[source] for segment in self[target][source])

    def iterInSegments(self, source):
        # generator for all in-segments from the network
        source = Vector(source).freeze()
        for target in self[source]:
            for segment in self[target][source]:
                yield segment

    def iterOutSegments(self, source):
        # generator for all out-segments from the network
        source = Vector(source).freeze()
        for target in self[source]:
            for segment in self[source][target]:
                yield segment

    def iterAdjacentNodes(self, source):
        # generator for the nodes adjacent to <source>  nodefrom the network
        source = Vector(source).freeze()
        return iter(self[source])

    def iterAllSegments(self):
        # generator for all segments from the network
        for source in self.iterNodes():
            for target in self[source]:
                # When source == target, we have a loop segment
                if source <= target: 
                    for segment in self[source][target]:
                        yield segment

    def iterAllForwardSegments(self):
        # generator for all segments from the network
        for source in self.iterNodes():
            for target in self[source]:
                for segment in self[source][target]:
                    if segment.forward:
                        yield segment

    def iterAllIntersectionNodes(self):
        # iterator for all nodes that form an intersection
        # includes also end-nodes!
        for source in self.iterNodes():
            if len(self[source]) != 2: # order of node != 2
                yield source

    def iterAlongWay(self,segment):
        # Generator for nodes that follow the way in the direction given by the
        # <segment>, until a crossing occurs, an end-point is reached or the 
        # way-type changes. The first return is <segment>.
        def changedWayType(category1,category2,tags1,tags2):
            if category1 != category2: return True
            if tags1 and tags2: # category 'scene_border' has no tags
                if tags1.get('lanes') != tags2.get('lanes'): return True
                if tags1.get('oneway') != tags2.get('oneway'): return True
                # if tags1.get('bridge') != tags2.get('bridge'): return True
            return False

        firstCategory = segment.category
        firstTags = segment.tags
        current = segment
        yield current
        while len(self[current.t]) == 2: # order of node == 2 -> no crossing or end-point
            current = [self[current.t][source] for source in self[current.t] if source != current.s][0][0]
            if changedWayType(firstCategory,current.category,firstTags,current.tags):
                break
            yield current

    def getCrossingsThatContain(self, categories):
        found = []
        categories_set = set(categories)
        for source in iter(self):
            segments = [self[source][target][0] for target in iter(self[source])]
            cats_set = {segment.category for segment in segments}
            degree = len(segments)
            if degree or (degree==2 and len(cats_set) > 1):
                if categories_set & cats_set:
                    found.append(source)
        return found

    def compare_angles(self, v1, v2):
        return 1 if atan2(v1[1],v1[0])+(v1[1]<0)*PI2 < atan2(v2[1],v2[0])+(v2[1]<0)*PI2 else -1

    def createCircularEmbedding(self):
        self.counterClockEmbedding = dict(list())
        for node in self:
            neighbors = [seg for seg in self.iterOutSegments(node)]
            if len(neighbors)>1:
                ordering = sorted(neighbors, key = cmp_to_key( lambda a,b: self.compare_angles(a.firstV,b.firstV)) )
            else:
                ordering = neighbors
            self.counterClockEmbedding[node] = ordering

    def getCycles(self):
        if not self.counterClockEmbedding:
            self.createCircularEmbedding()

        # create set of all segments 
        segmentSet = set(segment for segment in self.iterAllSegments())

        cycleSegs = []
        islandSegs = []
        solitarySegs = []
        while (len(segmentSet) > 0):
            # start with a first segment
            s = next(iter(segmentSet))
            segs = [s]
            segmentSet -= set([s])
            while True:
                neighbors = self.counterClockEmbedding[segs[-1].t]
                nextSeg = neighbors[(neighbors.index(~segs[-1])+1)%(len(neighbors))]
                if nextSeg == segs[0]:
                    cycVerts = [v for s in segs for v in s.path[:-1] ] + [segs[0].s]
                    area = sum( (p2[0]-p1[0])*(p2[1]+p1[1]) for p1,p2 in zip(cycVerts,cycVerts[1:]+[cycVerts[0]])) 
                    # Clockwise cycles are outer contours.
                    if area < 0.:
                            cycleSegs.append(segs)
                    # Counter-clockwise cycles are from islands (holes).
                    # Exception: the outer contour of the scene border
                    elif area > 0.:
                        if not all(s.category == 'scene_border' for s in segs):
                            islandSegs.append(segs)
                            # plotSimpleCycle(segs,'r')
                    # These are solitary spurs or dead-end ways
                    else:
                        solitarySegs.append(segs)
                        # plotSimpleCycle(segs,'b')
                    break
                else:
                    if nextSeg in segs:
                        print('ERROR: Endless loop stopped in WayNetwork.getCycles(')
                        break
                    segs.append(nextSeg)
                    segmentSet -= set([nextSeg])
        return cycleSegs, islandSegs, solitarySegs

    
# ------------------------------------------------------------------
# this part is only used to temporary visualize the cycles during development.
from itertools import *
import matplotlib.pyplot as plt
def _iterCircularPrevNext(lst):
    prevs, nexts = tee(lst)
    prevs = islice(cycle(prevs), len(lst) - 1, None)
    return zip(prevs,nexts)

def _iterCircularPrevThisNext(lst):
    prevs, this, nexts = tee(lst, 3)
    prevs = islice(cycle(prevs), len(lst) - 1, None)
    nexts = islice(cycle(nexts), 1, None)
    return zip(prevs, this, nexts)

def plotCycle(cycle):
    nodes = [n for s in cycle for n in s.path[:-1]]

    scaledNodes = []
    for n0,n1,n2 in _iterCircularPrevThisNext(nodes):
        v0 = (n0-n1).normalized()
        v1 = (n2-n1).normalized()
        is_reflex = v0.cross(v1) > 0
        bisector = ( (v0+v1)*(-1 if is_reflex else 1) ).normalized()
        c = abs(v0.cross(bisector))
        # cc = 1.0#(1/c if c>0.1 else 10.) 
        bisector *= 3
        scaledNodes.append(n1+bisector)

    x = [n[0] for n in scaledNodes]
    y = [n[1] for n in scaledNodes]
    plt.fill(x,y,'#ff0000',alpha = 0.03,zorder = 500)
    for v1,v2 in _iterCircularPrevNext(scaledNodes):
        plt.plot((v1[0], v2[0]),(v1[1], v2[1]),'b:',alpha = 1.0,zorder = 500,linewidth=0.5)

cCount = 0
cColors = ['r','b','g']

def plotSingleCycle(cycle):
    global cCount,cColors
    nodes = [n for s in cycle for n in s.path[:-1]]
    x = [n[0] for n in nodes]
    y = [n[1] for n in nodes]
    plt.fill(x,y,'#ff0000',alpha = 0.03,zorder = 500)
    color = cColors[cCount]
    cCount +=1
    for v1,v2 in _iterCircularPrevNext(nodes):
        plt.plot((v1[0], v2[0]),(v1[1], v2[1]),'r:',alpha = 1.0,zorder = 500,linewidth=1)
    for wayseg in cycle:
        x = (v1[0]+v2[0])/2
        y = (v1[1]+v2[1])/2
        plt.text(x,y,str(wayseg.ID))

def plotSimpleCycle(cycle,color='k'):
    nodes = [n for s in cycle for n in s.path[:-1]]
    for v1,v2 in _iterCircularPrevNext(nodes):
        plt.plot((v1[0], v2[0]),(v1[1], v2[1]),color,alpha = 1.0,zorder = 900,linewidth=2)
    for i,wayseg in enumerate(cycle):
        v1,v2 = wayseg.s,wayseg.t
        x = (v1[0]+v2[0])/2
        y = (v1[1]+v2[1])/2
        # plt.text(x,y,str(wayseg.ID))
        # plt.text(x,y,str(i))

def plotEnd():
    plt.gca().axis('equal')
    plt.show()




