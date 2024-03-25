# -*- coding: utf-8 -*-

"""
https://github.com/prochitecture/bpypolyskel

Implementation of the straight skeleton algorithm as described by Felkel and Obdržálek in their 1998 conference paper 
'Straight skeleton implementation'.

The code for skeletonize() has been ported from the implementation by Botffy at https://github.com/Botffy/polyskel,
in order to be able to use it in Blender. The main changes are:

- The order of the vertices of the polygon has been changed to a right-handed coordinate system
  (as used in Blender). The positive x and y axes point right and up, and the z axis points into 
  your face. Positive rotation is counterclockwise around the z-axis.
- The geometry objects used from the library euclid3 in the implementation of Bottfy have been
  replaced by objects based on mathutils.Vector. These objects are defined in the new library bpyeuclid.
- The signature of skeletonize() has been changed to lists of edges for the polygon and eventual hole.
  These are of type Edge2, defined in bpyeuclid.
- Some parts of the skeleton computations have been changed to fix errors produced by the original implementation.
- Algorithms to merge clusters of skeleton nodes and to filter ghost edges have been added.
- A pattern matching algorithm to detect apses, that creates a multi-edge event to create a proper apse skeleton.
"""

import heapq
from collections import namedtuple
from itertools import *
from collections import Counter
from operator import itemgetter
import re


from .bpyeuclid import *
from .poly2FacesGraph import poly2FacesGraph

EPSILON = 0.00001
PARALLEL = 0.01     # set this value to 1-cos(alpha), where alpha is the largest angle 
                    # between lines to accept them as parallelaccepted as 'parallel'.

# Add a key to enable debug output. For example:
# debugOutputs["skeleton"] = 1
# Then the Python list <skeleton> will be added to <debugOutputs> with the key <skeleton>
# in the function skeletonize(..)
debugOutputs = {}


def _iterCircularPrevNext(lst):
    prevs, nexts = tee(lst)
    prevs = islice(cycle(prevs), len(lst) - 1, None)
    return zip(prevs,nexts)

def _iterCircularPrevThisNext(lst):
    prevs, this, nexts = tee(lst, 3)
    prevs = islice(cycle(prevs), len(lst) - 1, None)
    nexts = islice(cycle(nexts), 1, None)
    return zip(prevs, this, nexts)
    
def _approximately_equals(a, b):
    return a == b or ( (a-b).magnitude <= max( a.magnitude, b.magnitude) * 0.001)

def robustFloatEqual(f1,f2):
    if abs(f1-f2) <= EPSILON:
        return True
    else:
        return abs(f1-f2) <= EPSILON * max(abs(f1),abs(f2))

class _SplitEvent(namedtuple("_SplitEvent", "distance, intersection_point, vertex, opposite_edge")):
    __slots__ = ()

    def __lt__(self, other):
        return self.distance < other.distance

class _EdgeEvent(namedtuple("_EdgeEvent", "distance intersection_point vertex_a vertex_b")):
    __slots__ = ()

    def __lt__(self, other):
        return self.distance < other.distance

class DormerEvent():
    def __init__(self, distance, intersection_point, eventList):
        self.distance = distance
        self.intersection_point = intersection_point
        self.eventList = eventList
    def __lt__(self, other):
        # print('difference S: ', self.distance - other.distance, self.distance < other.distance, (self.distance - other.distance)< -EPSILON)
        return self.distance < other.distance
    def __str__(self):
        return "DormerEvent:%4d d=%4.2f, ip=%s"%(self.id,self.distance,self.intersection_point)
    def __repr__(self):
        return "DormerEvent:%4d d=%4.2f, ip=%s"%(self.id,self.distance,self.intersection_point)

_OriginalEdge = namedtuple("_OriginalEdge", "edge bisector_prev, bisector_next")

Subtree = namedtuple("Subtree", "source, height, sinks")

class _LAVertex:
    def __init__(self, point, edge_prev, edge_next, direction_vectors=None,forceConvex=False):
        # point is the vertex V(i)
        # edge_prev is the edge from vertex V(i-1) to V(i)
        # edge_next is the edge from vertex V(i) to V(i+1)

        self.point = point
        self.edge_prev = edge_prev
        self.edge_next = edge_next
        self.prev = None
        self.next = None
        self.lav = None
        self._valid = True  # TODO this might be handled better. Maybe membership in lav implies validity?

        # creator_vectors are unit vectors: ( V(i) to V(i-1), V(i) to V(i+1) )
        creator_vectors = (edge_prev.norm * -1, edge_next.norm)
        if direction_vectors is None:
            direction_vectors = creator_vectors

        dv0 = direction_vectors[0]
        dv1 = direction_vectors[1]
        self._is_reflex = dv0.cross(dv1) > 0
        if forceConvex:
            self._is_reflex = False
        op_add_result = creator_vectors[0] + creator_vectors[1]
        self._bisector = Ray2(self.point, op_add_result * (-1 if self._is_reflex else 1))
        # Vertex created

    def invalidate(self):
        if self.lav is not None:
            self.lav.invalidate(self)
        else:
            self._valid = False

    @property
    def bisector(self):
        return self._bisector

    @property
    def is_reflex(self):
        return self._is_reflex

    @property
    def original_edges(self):
        return self.lav._slav._original_edges

    @property
    def is_valid(self):
        return self._valid

    def next_event(self):
        events = []
        if self.is_reflex:
            # a reflex vertex may generate a split event
            # split events happen when a vertex hits an opposite edge, splitting the polygon in two.
            for edge in self.original_edges:
                if edge.edge == self.edge_prev or edge.edge == self.edge_next:
                    continue

                # a potential b is at the intersection of between our own bisector and the bisector of the
                # angle between the tested edge and any one of our own edges.

                # we choose the "less parallel" edge (in order to exclude a potentially parallel edge)
                prevdot = abs(self.edge_prev.norm.dot(edge.edge.norm))
                nextdot = abs(self.edge_next.norm.dot(edge.edge.norm))
                selfedge = self.edge_prev if prevdot < nextdot else self.edge_next 

                i = Line2(selfedge).intersect(Line2(edge.edge))
                if i is not None and not _approximately_equals(i, self.point):
                    # locate candidate b
                    linvec = (self.point - i).normalized()
                    edvec = edge.edge.norm
                    if abs(self.bisector.v.cross(linvec) - 1.0) < EPSILON:
                        linvec = (self.point - i + edvec*0.01 ).normalized()
                    if self.bisector.v.cross(linvec) < 0: 
                        edvec = -edvec

                    bisecvec = edvec + linvec
                    if not bisecvec.magnitude:
                        continue
                    bisector = Line2(i, bisecvec, 'pv')

                    b = bisector.intersect(self.bisector)

                    if b is None:
                        continue

                    # check eligibility of b
                    # a valid b should lie within the area limited by the edge and the bisectors of its two vertices:
                    xprev	= ( (edge.bisector_prev.v.normalized()).cross( (b - edge.bisector_prev.p).normalized() )) < EPSILON
                    xnext	= ( (edge.bisector_next.v.normalized()).cross( (b - edge.bisector_next.p).normalized() )) > -EPSILON
                    xedge	= ( edge.edge.norm.cross( (b - edge.edge.p1).normalized() )) > -EPSILON

                    parallel = edge.bisector_next.v.length == 0.
                    if not (xprev and xnext and xedge and not parallel):
                        # Candidate discarded
                        continue

                    # Found valid candidate
                    events.append(_SplitEvent(Line2(edge.edge).distance(b), b, self, edge.edge))

        i_prev = self.bisector.intersect(self.prev.bisector)
        i_next = self.bisector.intersect(self.next.bisector)

        if i_prev is not None:
            events.append(_EdgeEvent(Line2(self.edge_prev).distance(i_prev), i_prev, self.prev, self))
        if i_next is not None:
            events.append(_EdgeEvent(Line2(self.edge_next).distance(i_next), i_next, self, self.next))

        if not events:
            return None

        ev = min(events, key=lambda event: (self.point-event.intersection_point).length)

        # Generated new event
        return ev

class _LAV:
    def __init__(self, slav):
        self.head = None
        self._slav = slav
        self._len = 0

    @classmethod
    # edgeContour is a list of edges of class Edge2
    def from_polygon(cls, edgeContour, slav):
        lav = cls(slav)
        for prev, nxt in _iterCircularPrevNext(edgeContour):
            # V(i) is the current vertex
            # prev is the edge from vertex V(i-1) to V(i)
            # this is the edge from vertex V(i-) to V(i+1)
            lav._len += 1
            vertex = _LAVertex(nxt.p1, prev, nxt)
            vertex.lav = lav
            if lav.head is None:
                lav.head = vertex
                vertex.prev = vertex.next = vertex
            else:
                vertex.next = lav.head
                vertex.prev = lav.head.prev
                vertex.prev.next = vertex
                lav.head.prev = vertex
        return lav

    @classmethod
    def from_chain(cls, head, slav):
        lav = cls(slav)
        lav.head = head
        for vertex in lav:
            lav._len += 1
            vertex.lav = lav
        return lav

    def invalidate(self, vertex):
        assert vertex.lav is self, "Tried to invalidate a vertex that's not mine"
        vertex._valid = False
        if self.head == vertex:
            self.head = self.head.next
        vertex.lav = None

    def unify(self, vertex_a, vertex_b, point):
        replacement = _LAVertex(point, vertex_a.edge_prev, vertex_b.edge_next,
                                (vertex_b.bisector.v.normalized(), vertex_a.bisector.v.normalized()))
        replacement.lav = self

        if self.head in [vertex_a, vertex_b]:
            self.head = replacement

        vertex_a.prev.next = replacement
        vertex_b.next.prev = replacement
        replacement.prev = vertex_a.prev
        replacement.next = vertex_b.next

        vertex_a.invalidate()
        vertex_b.invalidate()

        self._len -= 1
        return replacement

    def __len__(self):
        return self._len

    def __iter__(self):
        visited = set()
        cur = self.head
        while True:
            yield cur
            cur = cur.next
            if cur == self.head:
                return
            if cur in visited:
                raise RuntimeError("(Infinite loop) circular reference detected in LAV.")
            visited.add(cur)

class _SLAV:
    def __init__(self, edgeContours):
        self._lavs = [_LAV.from_polygon(edgeContour, self) for edgeContour in edgeContours]

        # store original polygon edges for calculation of split events
        self._original_edges = [
            _OriginalEdge(vertex.edge_prev, vertex.prev.bisector, vertex.bisector)
            for vertex in chain.from_iterable(self._lavs)
        ]

    def __iter__(self):
        for lav in self._lavs:
            yield lav

    def empty(self):
        return not self._lavs

    def handle_dormer_event(self, event):
        # handle split events (indices 0 and 1)
        ev_prev = event.eventList[0]
        ev_next = event.eventList[1]
        ev_edge = event.eventList[2]
        v_prev = ev_prev.vertex
        v_next = ev_next.vertex

        lav = ev_prev.vertex.lav
        if lav is None:
            return ([],[])

        toRemove = [v_prev,v_prev.next,v_next,v_next.prev]
        lav.head = v_prev.prev

        v_prev.prev.next = v_next.next
        v_next.next.prev = v_prev.prev

        new_lav = [_LAV.from_chain(lav.head, self)]
        self._lavs.remove(lav)
        self._lavs.append(new_lav[0])

        p = v_prev.bisector.intersect(v_next.bisector)
        arcs = []
        # from edge event
        arcs.append( Subtree(ev_edge.intersection_point, ev_edge.distance, [ev_edge.vertex_a.point,ev_edge.vertex_b.point,p] ) )

        # from split events
        arcs.append( Subtree(p, (ev_prev.distance+ev_next.distance)/2.0, [v_prev.point,v_next.point])   )
    
        for v in toRemove:
            v.invalidate()

        return (arcs, [])

    def handle_edge_event(self, event):
        sinks = []
        events = []

        lav = event.vertex_a.lav
        if event.vertex_a.prev == event.vertex_b.next:
            # Peak event at intersection
            self._lavs.remove(lav)
            for vertex in list(lav):
                sinks.append(vertex.point)
                vertex.invalidate()
        else:
            # Edge event at intersection            
            new_vertex = lav.unify(event.vertex_a, event.vertex_b, event.intersection_point)
            if lav.head in (event.vertex_a, event.vertex_b):
                lav.head = new_vertex
            sinks.extend((event.vertex_a.point, event.vertex_b.point))
            next_event = new_vertex.next_event()
            if next_event is not None:
                events.append(next_event)
        return (Subtree(event.intersection_point, event.distance, sinks), events)

    def handle_split_event(self, event):
        lav = event.vertex.lav

        sinks = [event.vertex.point]
        vertices = []
        x = None  # next vertex
        y = None  # previous vertex
        norm = event.opposite_edge.norm
        for v in chain.from_iterable(self._lavs):
            if norm == v.edge_prev.norm and event.opposite_edge.p1 == v.edge_prev.p1:
                x = v
                y = x.prev
            elif norm == v.edge_next.norm and event.opposite_edge.p1 == v.edge_next.p1:
                y = v
                x = y.next

            if x:
                xprev	= (y.bisector.v.normalized()).cross((event.intersection_point - y.point).normalized()) <= EPSILON
                xnext	= (x.bisector.v.normalized()).cross((event.intersection_point - x.point).normalized()) >= -EPSILON

                if xprev and xnext:
                    break
                else:
                    x = None
                    y = None

        if x is None:
            # Split event failed (equivalent edge event is expected to follow)
            return (None, [])

        v1 = _LAVertex(event.intersection_point, event.vertex.edge_prev, event.opposite_edge,None,True)
        v2 = _LAVertex(event.intersection_point, event.opposite_edge, event.vertex.edge_next,None,True)

        v1.prev = event.vertex.prev
        v1.next = x
        event.vertex.prev.next = v1
        x.prev = v1

        v2.prev = y
        v2.next = event.vertex.next
        event.vertex.next.prev = v2
        y.next = v2

        new_lavs = None
        self._lavs.remove(lav)
        if lav != x.lav:
            # the split event actually merges two lavs
            self._lavs.remove(x.lav)
            new_lavs = [_LAV.from_chain(v1, self)]
        else:
            new_lavs = [_LAV.from_chain(v1, self), _LAV.from_chain(v2, self)]

        for l in new_lavs:
            if len(l) > 2:
                self._lavs.append(l)
                vertices.append(l.head)
            else:
                # LAV has collapsed into the line
                sinks.append(l.head.next.point)
                for v in list(l):
                    v.invalidate()

        events = []
        for vertex in vertices:
            next_event = vertex.next_event()
            if next_event is not None:
                events.append(next_event)

        event.vertex.invalidate()
        return (Subtree(event.intersection_point, event.distance, sinks), events)


class _EventQueue:
    def __init__(self):
        self.__data = []

    def put(self, item):
        if item is not None:
            heapq.heappush(self.__data, item)

    def put_all(self, iterable):
        for item in iterable:
            heapq.heappush(self.__data, item)

    def get(self):
        return heapq.heappop(self.__data)

    def getAllEqualDistance(self):
        item = heapq.heappop(self.__data)
        equalDistanceList = [item]
        # from top of queue, get all events that have the same distance as the one on top
        while self.__data and abs(self.__data[0].distance-item.distance) < 0.001:
            queueTop = heapq.heappop(self.__data)
            equalDistanceList.append(queueTop)
        return equalDistanceList

    def empty(self):
        return not self.__data

    def peek(self):
            return self.__data[0]

    def show(self):
        for item in self.__data:
            print(item)


def checkEdgeCrossing(skeleton):
    # extract all edges
    sk_edges = []
    for arc in skeleton:
        p1 = arc.source
        for p2 in arc.sinks:
            sk_edges.append( Edge2(p1,p2) )

    combs = combinations(sk_edges,2)
    nrOfIntsects = 0
    for e in combs:
        # check for intersection, exclude endpoints
        denom = ((e[0].p2.x-e[0].p1.x)*(e[1].p2.y-e[1].p1.y))-((e[0].p2.y-e[0].p1.y)*(e[1].p2.x-e[1].p1.x))
        if not denom:
            continue
        n1 = ((e[0].p1.y-e[1].p1.y)*(e[1].p2.x-e[1].p1.x))-((e[0].p1.x-e[1].p1.x)*(e[1].p2.y-e[1].p1.y))
        r = n1 / denom
        n2 = ((e[0].p1.y-e[1].p1.y)*(e[0].p2.x-e[0].p1.x))-((e[0].p1.x-e[1].p1.x)*(e[0].p2.y-e[0].p1.y))
        s = n2 / denom
        if ((r <= EPSILON or r >= 1.0-EPSILON) or (s <= EPSILON or s >= 1.0-EPSILON)):
            continue    # no intersection
        else:
            nrOfIntsects += 1
    return nrOfIntsects


def removeGhosts(skeleton):
    # remove loops
    for arc in skeleton:
        if arc.source in arc.sinks:
            arc.sinks.remove(arc.source)
    # find and resolve parallel or crossed skeleton edges
    for arc in skeleton:
        source = arc.source
        # search for nearly parallel edges in all sinks from this node
        sinksAltered = True
        while sinksAltered:
            sinksAltered = False
            combs = combinations(arc.sinks,2)
            for pair in combs:
                s0 = pair[0]-source
                s1 = pair[1]-source
                s0m = s0.magnitude
                s1m = s1.magnitude
                if s0m!=0.0 and s1m!=0.0:
                    # check if this pair of edges is parallel
                    dotCosineAbs = abs(s0.dot(s1) / (s0m*s1m) - 1.0)
                    if dotCosineAbs < PARALLEL:
                        if s0m < s1m:
                            farSink = pair[1]
                            nearSink = pair[0]
                        else:
                            farSink = pair[0]
                            nearSink = pair[1]

                        nodeIndexList = [i for i, node in enumerate(skeleton) if node.source == nearSink]
                        if not nodeIndexList:   # both sinks point to polygon vertices (maybe small triangle)
                            break

                        nodeIndex = nodeIndexList[0]

                        if dotCosineAbs < EPSILON:  # We have a ghost edge, sinks almost parallel
                            skeleton[nodeIndex].sinks.append(farSink)
                            arc.sinks.remove(farSink)
                            arc.sinks.remove(nearSink)
                            sinksAltered = True
                            break
                        else:   # maybe we have a spike that crosses other skeleton edges
                                # Spikes normally get removed with more success as face-spike in polygonize().
                                # Remove it here only, if it produces any crossing. 
                            for sink in skeleton[nodeIndex].sinks:
                                if intersect(source, farSink, nearSink, sink):
                                    skeleton[nodeIndex].sinks.append(farSink)
                                    arc.sinks.remove(farSink)
                                    arc.sinks.remove(nearSink)
                                    sinksAltered = True
                                    break

def detectApses(outerContour):
    import re
    # compute cross-product between consecutive edges of outer contour
    # set True for angles a, where sin(a) < 0.5 -> 30°
    sequence = "".join([ 'L' if abs(p.norm.cross(n.norm))<0.5 else 'H' for p,n in _iterCircularPrevNext(outerContour) ])
    # special case, see test_306011654_pescara_pattinodromo
    if all([p=='L' for p in sequence]):
        return None
    N = len(sequence)
    # match at least 6 low angles in sequence (assume that the first match is longest)
    pattern = re.compile(r"(L){6,}")
    # sequence may be circular, like 'LLHHHHHLLLLL'
    matches = [r for r in pattern.finditer(sequence+sequence)]
    if not matches:
        return None

    centers = []
    # circular overlapping pattern must start in first sequence
    nextStart = 0
    for apse in matches:
        s = apse.span()[0]
        if s < N and s >= nextStart:
            apseIndices = [ i%len(sequence) for i in range(*apse.span())]
            apseVertices = [outerContour[i].p1 for i in apseIndices]
            center, R = fitCircle3Points(apseVertices)
            centers.append(center)
    
    return centers

def findClusters(skeleton, candidates, contourVertices, edgeContours, thresh):
    apseCenters = detectApses(edgeContours[0])
    clusters = []
    while candidates:
        c0 = candidates[0]
        cluster = [c0]
        ref = skeleton[c0]
        for c in candidates[1:]:
            arc = skeleton[c]
            # use Manhattan distance
            if abs(ref.source.x-arc.source.x) + abs(ref.source.y-arc.source.y) < thresh:
                cluster.append(c)
        for c in cluster:
            if c in candidates:
                candidates.remove(c)
        if len(cluster)>1:
            # if cluster is near to an apse center, don't merge any nodes
            if apseCenters:
                isApseCluster = False
                for apseCenter in apseCenters:
                    for node in cluster:
                        if abs(apseCenter.x-skeleton[node].source.x) + abs(apseCenter.y-skeleton[node].source.y) < 3.0:
                            isApseCluster = True
                            break
                    if isApseCluster:
                        break
                if isApseCluster:
                    continue

            # detect sinks in this cluster, that are contour vertices of the footprint
            nrOfContourSinks = 0
            contourSinks = []
            for node in cluster:
                sinks = skeleton[node].sinks
                contourSinks.extend( [s for s in sinks if s in contourVertices] )
                nrOfContourSinks += sum(el in sinks for el in contourVertices)

            # less than 2, then we can merge the cluster
            if nrOfContourSinks < 2:
                clusters.append(cluster)
                continue

            # Two or more contour sinks, maybe its an architectural detail, that we shouldn't merge.
            # There are only few sinks, so the minimal distance is computed by brute force
            minDist = 3*thresh
            combs = combinations(contourSinks,2)
            for pair in combs:
                minDist = min( (pair[0]-pair[1]).magnitude, minDist )
 
            if minDist > 2*thresh:
                clusters.append(cluster)    # contour sinks too far, so merge

    return clusters

def mergeCluster(skeleton, cluster):
    nodesToMerge = cluster.copy()

    # compute center of gravity as source of merged node.
    # in the same time, collect all sinks of the merged nodes
    x,y,height = (0.0,0.0,0.0)
    mergedSources = []
    for node in cluster:
        x += skeleton[node].source.x
        y += skeleton[node].source.y
        height += skeleton[node].height
        mergedSources.append(skeleton[node].source)
    N = len(cluster)
    new_source = mathutils.Vector((x/N,y/N))
    new_height = height/N

    # collect all sinks of merged nodes, that point outside the cluster
    new_sinks = []
    for node in cluster:
        for sink in skeleton[node].sinks:
            if sink not in mergedSources and sink not in new_sinks:
                new_sinks.append(sink)

    # create the merged node
    newnode = Subtree(new_source, new_height, new_sinks)

    # redirect all sinks of nodes outside the cluster, that pointed to 
    # one of the clustered nodes, to the new node
    for arc in skeleton:
        if arc.source not in mergedSources:
            to_remove = []
            for i,sink in enumerate(arc.sinks):
                if sink in mergedSources:
                    if new_source in arc.sinks:
                        to_remove.append(i)
                    else: 
                        arc.sinks[i] = new_source
            for i in sorted(to_remove, reverse = True):
                del arc.sinks[i]

    # remove clustered nodes from skeleton
    # and add the new node
    for i in sorted(nodesToMerge, reverse = True):
        del skeleton[i]
    skeleton.append(newnode)

def mergeNodeClusters(skeleton,edgeContours):
    # first merge all nodes that have exactly the same source
    sources = {}
    to_remove = []
    for i, p in enumerate(skeleton):
        source = tuple(i for i in p.source)
        if source in sources:
            source_index = sources[source]
            # source exists, merge sinks
            for sink in p.sinks:
                if sink not in skeleton[source_index].sinks:
                    skeleton[source_index].sinks.append(sink)
            to_remove.append(i)
        else:
            sources[source] = i
    for i in reversed(to_remove):
        skeleton.pop(i)

    contourVertices = [edge.p1 for contour in edgeContours for edge in contour]

    # Merge all clusters that have small distances due to floating-point inaccuracies
    smallThresh = 0.1
    hadCluster = True
    while hadCluster:
        hadCluster = False
        # find clusters within short range and short height difference
        candidates = [c for c in range(len(skeleton))]
        clusters = findClusters(skeleton,candidates,contourVertices,edgeContours,smallThresh) 
        # check if there are cluster candidates
        if not clusters:
            break
        hadCluster = True
        # use largest cluster
        cluster = max(clusters, key = lambda clstr: len(clstr))
        mergeCluster(skeleton, cluster)

    return skeleton

def detectDormers(slav, edgeContours):
    import re
    outerContour = edgeContours[0]
    def coder(cp):
        if cp > 0.99:
            code = 'L'
        elif cp < -0.99:
            code = 'R'
        else:
            code = '0'
        return code

    sequence = "".join([ coder(p.norm.cross(n.norm)) for p,n in _iterCircularPrevNext(outerContour) ])
    N = len(sequence)
   # match a pattern of almost rectangular turns to right, then to left, to left and again to right
    # positive lookahead used to find overlapping patterns
    pattern = re.compile(r"(?=(RLLR))")
    # sequence may be circular, like 'LRLL000LL00RL', therefore concatenate two of them
    matches = [r for r in pattern.finditer(sequence+sequence)]

    dormerIndices = []
    # circular overlapping pattern must start in first sequence
    nextStart = 0
    for dormer in matches:
        s = dormer.span()[0]
        if s < N and s >= nextStart:
            oi = [ i%len(sequence) for i in range(*(s,s+4))]    # indices of candidate dormer
            dormerIndices.append(oi)
            nextStart = s+3

    # filter overlapping dormers
    toRemove = []
    for oi1,oi2 in zip(dormerIndices, dormerIndices[1:] + dormerIndices[:1]):
        if oi1[3] == oi2[0]:
            toRemove.extend([oi1,oi2])
    for sp in toRemove:
        if sp in dormerIndices:
            dormerIndices.remove(sp)

    # check if contour consists only of dormers, if yes then skip, can't handle that
    # (special case for test_51340792_yekaterinburg_mashinnaya_35a)
    dormerVerts = set()
    for oi in dormerIndices:
        dormerVerts.update( oi )
    if len(dormerVerts) == len(outerContour):
        return []

    dormers = []
    for oi in dormerIndices:
        w = outerContour[oi[1]].length_squared()    # length^2 of base edge
        d1 = outerContour[oi[0]].length_squared()   # length^2 of side edge
        d2 = outerContour[oi[2]].length_squared()   # length^2 of side edge
        d = abs(d1-d2)/(d1+d2)  # "contrast" of side edges lengths^2
        d3 = outerContour[(oi[0]+N-1)%N].length_squared()   # length^2 of previous edge
        d4 = outerContour[oi[3]].length_squared()           # length^2 of next edge
        facLeft = 0.125 if sequence[(s+N-1)%N] != 'L' else 1.5
        facRight = 0.125 if sequence[(s+4)%N] != 'L' else 1.5
        if w < 100 and d < 0.35 and d3 >= w*facLeft and d4 >= w*facRight:
            dormers.append((oi,(outerContour[oi[1]].p1-outerContour[oi[1]].p2).magnitude))
            
    return dormers

def processDormers(dormers,initialEvents):
    dormerEvents = []
    dormerEventIndices = []
    for dormer in dormers:
        dormerIndices = dormer[0]
        d_events = [ev for i,ev in enumerate(initialEvents) if i in dormerIndices]
        if all([(d is not None) for d in d_events]): # if all events are valid
            if  not isinstance(d_events[0], _SplitEvent) or \
                not isinstance(d_events[1], _EdgeEvent) or \
                not isinstance(d_events[3], _SplitEvent):
                continue
            ev_prev = d_events[0]
            ev_next = d_events[3]
            v_prev = ev_prev.vertex
            v_next = ev_next.vertex
            p = v_prev.bisector.intersect(v_next.bisector)
            d = dormer[1]/2.0
            # process events:                         split1       split2       edge
            dormerEvents.append( DormerEvent(d, p, [d_events[0], d_events[3] ,d_events[1]]) )
            dormerEventIndices.extend(dormerIndices)

    remainingEvents = [ev for i,ev in enumerate(initialEvents) if i not in dormerEventIndices]
    del initialEvents[:]
    initialEvents.extend(remainingEvents)
    initialEvents.extend(dormerEvents)


def skeletonize(edgeContours):
    """
skeletonize() computes the straight skeleton of a polygon. It accepts a simple description of the
contour of a footprint polygon, including those of evetual holes, and returns the nodes and edges 
of its straight skeleton.

The polygon is expected as a list of contours, where every contour is a list of edges of type Edge2
(imported from bpyeuclid). The outer contour of the polygon is the first list of in the list of
contours and is expected in counterclockwise order. In the right-handed coordinate system, seen from
top, the polygon is on the left of its contour.

If the footprint has holes, their contours are expected as lists of their edges, following the outer
contour of the polygon. Their edges are in clockwise order, seen from top, the polygon is on the left
of the hole's contour.

Arguments:
---------
edgeContours:   A list of contours of the polygon and eventually its holes, where every contour is a
                list of edges of type `Edge2` (imported from `bpyeuclid`). It is expected to as:

                edgeContours = [ polygon_edge,<hole1_edges>, <hole2_edges>, ...]

                polygon_egdes is a list of the edges of the outer polygon contour in counterclockwise
                order. <hole_edges> is an optional list of the edges of a hole contour in clockwise order. 

Output:
------
return:         A list of subtrees (of type Subtree) of the straight skeleton. A Subtree contains the
                attributes (source, height, sinks), where source is the node vertex, height is its
                distance to the nearest polygon edge, and sinks is a list of vertices connected to the
                node. All vertices are of type mathutils.Vector with two dimension x and y. 
    """
    slav = _SLAV(edgeContours)

    dormers = detectDormers(slav, edgeContours)

    initialEvents = []
    for lav in slav:
        for vertex in lav:
            initialEvents.append(vertex.next_event())

    if dormers:
        processDormers(dormers,initialEvents)

    output = []
    prioque = _EventQueue()
    for ev in initialEvents:
        if ev:
            prioque.put(ev)

    while not (prioque.empty() or slav.empty()):
        topEventList = prioque.getAllEqualDistance()
        for i in topEventList:
            if isinstance(i, _EdgeEvent):
                if not i.vertex_a.is_valid or not i.vertex_b.is_valid:
                   continue
                (arc, events) = slav.handle_edge_event(i)
            elif isinstance(i, _SplitEvent):
                if not i.vertex.is_valid:
                    continue
                (arc, events) = slav.handle_split_event(i)
            elif isinstance(i, DormerEvent):
                if not i.eventList[0].vertex.is_valid or not i.eventList[1].vertex.is_valid:
                    continue
                (arc, events) = slav.handle_dormer_event(i)
            prioque.put_all(events)

            if arc is not None:
                if isinstance(arc, list):
                    output.extend(arc)
                else:
                    output.append(arc)

    output = mergeNodeClusters(output,edgeContours)
    removeGhosts(output)

    return output

def polygonize(verts, firstVertIndex, numVerts, holesInfo=None, height=0., tan=0., faces=None, unitVectors=None):
    """
    polygonize() computes the faces of a hipped roof from a footprint polygon of a building, skeletonized
    by a straight skeleton. It accepts a simple description of the vertices of the footprint polygon,
    including those of evetual holes, and returns a list of polygon faces.

    The polygon is expected as a list of vertices in counterclockwise order. In a right-handed coordinate
    system, seen from top, the polygon is on the left of its contour. Holes are expected as lists of vertices
    in clockwise order. Seen from top, the polygon is on the left of the hole's contour.

    Arguments:
    ----------
    verts:              A list of vertices. Vertices that define the outer contour of the footprint polygon are
                        located in a continuous block of the verts list, starting at the index firstVertIndex.
                        Each vertex is an instance of `mathutils.Vector` with 3 coordinates x, y and z. The
                        z-coordinate must be the same for all vertices of the polygon.

                        The outer contour of the footprint polygon contains `numVerts` vertices in counterclockwise
                        order, in its block in `verts`.

                        Vertices that define eventual holes are also located in `verts`. Every hole takes its continuous
                        block. The start index and the length of every hole block are described by the argument
                        `holesInfo`. See there.

                        The list of vertices verts gets extended by `polygonize()`. The new nodes of the straight
                        skeleton are appended at the end of the list.

    firstVertIndex: 	The first index of vertices of the polygon index in the verts list that defines the footprint polygon.

    numVerts:           The first index of the vertices in the verts list of the polygon, that defines the outer
                        contour of the footprint.

    holesInfo:          If the footprint polygon contains holes, their position and length in the verts list are
                        described by this argument. `holesInfo` is a list of tuples, one for every hole. The first
                        element in every tuple is the start index of the hole's vertices in `verts` and the second
                        element is the number of its vertices.

                        The default value of holesInfo is None, which means that there are no holes.

    height: 	        The maximum height of the hipped roof to be generated. If both `height` and `tan` are equal
                        to zero, flat faces are generated. `height` takes precedence over `tan` if both have a non-zero
                        value. The default value of `height` is 0.0.

    tan:                In many cases it's desirable to deal with the roof pitch angle instead of the maximum roof
                        height. The tangent `tan` of the roof pitch angle can be supplied for that case. If both `height`
                        and `tan` are equal to zero, flat faces are generated. `height` takes precedence over `tan` if
                        both have a non-zero value. The default value of `tan` is 0.0.

    faces:              An already existing Python list of faces. Every face in this list is itself a list of
                        indices of the face-vertices in the verts list. If this argument is None (its default value),
                        a new list with the new faces created by the straight skeleton is created and returned by
                        polygonize(), else faces is extended by the new list.

    unitVectors:        A Python list of unit vectors along the polygon edges (including holes if they are present).
                        These vectors are of type `mathutils.Vector` with three dimensions. The direction of the vectors
                        corresponds to order of the vertices in the polygon and its holes. The order of the unit
                        vectors in the unitVectors list corresponds to the order of vertices in the input Python list
                        verts.

                        The list `unitVectors` (if given) gets used inside polygonize() function instead of calculating
                        it once more. If this argument is None (its default value), the unit vectors get calculated
                        inside polygonize().

    Output:
    ------
    verts:              The list of the vertices `verts` gets extended at its end by the vertices of the straight skeleton.

    return:             A list of the faces created by the straight skeleton. Every face in this list is a list of
                        indices of the face-vertices in the verts list. The order of vertices of the faces is
                        counterclockwise, as the order of vertices in the input Python list `verts`. The first edge of
                        a face is always an edge of the polygon or its holes.

                        If a list of faces has been given in the argument faces, it gets extended at its end by the
                        new list.
    """
    # assume that all vertices of polygon and holes have the same z-value
    zBase = verts[firstVertIndex][2]

    # compute center of gravity of polygon
    center = mathutils.Vector((0.0, 0.0, 0.0))
    for i in range(firstVertIndex,firstVertIndex+numVerts):
        center += verts[i]
    center /= numVerts
    center[2] = 0.0

    # create 2D edges as list and as contours for skeletonization and graph construction
    lastUIndex = numVerts-1
    lastVertIndex = firstVertIndex + lastUIndex
    if unitVectors:
        edges2D = [
            Edge2(index, index+1, unitVectors[uIndex], verts, center)\
                for index, uIndex in zip( range(firstVertIndex, lastVertIndex), range(lastUIndex) )
        ]
        edges2D.append(Edge2(lastVertIndex, firstVertIndex, unitVectors[lastUIndex], verts, center))
    else:
        edges2D = [
            Edge2(index, index+1, None, verts, center) for index in range(firstVertIndex, lastVertIndex)
        ]
        edges2D.append(Edge2(lastVertIndex, firstVertIndex, None, verts, center))
    edgeContours = [edges2D.copy()]
    
    uIndex = numVerts
    if holesInfo:
        for firstVertIndexHole,numVertsHole in holesInfo:
            lastVertIndexHole = firstVertIndexHole + numVertsHole-1
            if unitVectors:
                lastUIndex = uIndex+numVertsHole-1
                holeEdges = [
                    Edge2(index, index+1, unitVectors[uIndex], verts, center)\
                    for index, uIndex in zip(range(firstVertIndexHole, lastVertIndexHole), range(uIndex, lastUIndex))
                ]
                holeEdges.append(Edge2(lastVertIndexHole, firstVertIndexHole, unitVectors[lastUIndex], verts, center))
            else:
                holeEdges = [
                    Edge2(index, index+1, None, verts, center) for index in range(firstVertIndexHole, lastVertIndexHole)
                ]
                holeEdges.append(Edge2(lastVertIndexHole, firstVertIndexHole, None, verts, center))
            edges2D.extend(holeEdges)
            edgeContours.append(holeEdges)
            uIndex += numVertsHole

	# compute skeleton
    skeleton = skeletonize(edgeContours)

    # evetual debug output of skeleton
    if 'skeleton' in debugOutputs:
        debugOutputs['skeleton'] = skeleton

	# compute skeleton node heights and append nodes to original verts list,
	# see also issue #4 at https://github.com/prochitecture/bpypolyskel
    if height:
        maxSkelHeight = max(arc.height for arc in skeleton)
        tan_alpha = height/maxSkelHeight
    else:
        tan_alpha = tan
    skeleton_nodes3D = []
    for arc in skeleton:
        node = mathutils.Vector((arc.source.x, arc.source.y, arc.height*tan_alpha+zBase))
        skeleton_nodes3D.append(node+center)
    firstSkelIndex = len(verts) # first skeleton index in verts
    verts.extend(skeleton_nodes3D)

    # instantiate the graph for faces
    graph = poly2FacesGraph()

    # add polygon and hole indices to graph using indices in verts
    for edge in _iterCircularPrevNext( range(firstVertIndex, firstVertIndex+numVerts) ):
        graph.add_edge(edge)
    
    if holesInfo:
        for firstVertIndexHole,numVertsHole in holesInfo:
            for edge in _iterCircularPrevNext( range(firstVertIndexHole, firstVertIndexHole+numVertsHole) ):
                graph.add_edge(edge)

    # add skeleton edges to graph using indices in verts
    for index, arc in enumerate(skeleton):
        aIndex = index + firstSkelIndex
        for sink in arc.sinks:
            # first search in input edges
            edge = [edge for edge in edges2D if edge.p1==sink]
            if edge:
                sIndex = edge[0].i1
            else: # then it should be a skeleton node
                skelIndex = [index for index, arc in enumerate(skeleton) if arc.source==sink]
                if skelIndex:
                    sIndex = skelIndex[0] + firstSkelIndex
                else:
                    sIndex = -1 # error
            graph.add_edge( (aIndex,sIndex) )

    # generate clockwise circular embedding
    embedding = graph.circular_embedding(verts,'CCW')

    # compute list of faces, the vertex indices are still related to verts2D
    faces3D = graph.faces(embedding, firstSkelIndex)

    # find and remove spikes in faces
    hadSpikes = True
    while hadSpikes:
        hadSpikes = False
        # find spike
        for face in faces3D:
            if len(face) <= 3:   # a triangle is not considered as spike
                continue
            for prev, this, _next in _iterCircularPrevThisNext(face):
                s0 = verts[this]-verts[prev]  # verts are 3D vectors
                s1 = verts[_next]-verts[this]
                s0 = s0.xy  # need 2D-vectors
                s1 = s1.xy 
                s0m = s0.magnitude
                s1m = s1.magnitude
                if s0m and s1m:
                    dotCosine = s0.dot(s1) / (s0m*s1m)
                else:
                    continue
                crossSine = s0.cross(s1)
                if abs(dotCosine + 1.0) < PARALLEL and crossSine > -EPSILON: # spike edge to left
                    # the spike's peak is at 'this'
                    hadSpikes = True
                    break
                else:
                   continue

            if not hadSpikes:
                continue   # try next face

            # find faces adjacent to spike,
            # on right side it must have adjacent vertices in the order 'this' -> 'prev',
            # on left side it must have adjacent vertices in the order '_next' -> 'this',
            rightIndx, leftIndx = (None,None)
            for i,f in enumerate(faces3D):
                if [ p for p,n in _iterCircularPrevNext(f) if p == this and n== prev ]:
                    rightIndx = i
                if [ p for p,n in _iterCircularPrevNext(f) if p == _next and n== this ]:
                    leftIndx = i

            # part of spike is original polygon and cant get removed.
            if rightIndx is None or leftIndx is None:
                hadSpikes = False
                continue

            if rightIndx == leftIndx:   # single line into a face, but not separating it
                commonFace = faces3D[rightIndx]
                # remove the spike vertice and one of its neighbors
                commonFace.remove(this)
                commonFace.remove(prev)
                if this in face:
                    face.remove(this)
                break   # that's it for this face

            # rotate right face so that 'prev' is in first place
            rightFace = faces3D[rightIndx]
            rotIndex = next(x[0] for x in enumerate(rightFace)  if x[1] == prev )
            rightFace = rightFace[rotIndex:] + rightFace[:rotIndex]

            # rotate left face so that 'this' is in first place
            leftFace = faces3D[leftIndx]
            rotIndex = next(x[0] for x in enumerate(leftFace)  if x[1] == this )
            leftFace = leftFace[rotIndex:] + leftFace[:rotIndex]

            mergedFace = rightFace + leftFace[1:]

            # rotate edge list so that edge of original polygon is first edge
            nextOrigIndex = next(x[0] for x in enumerate(mergedFace) if x[0]<firstSkelIndex and x[1]< firstSkelIndex)
            mergedFace = mergedFace[nextOrigIndex:] + mergedFace[:nextOrigIndex]

            if mergedFace == face:  # no change, will result in endless loop
                raise Exception('Endless loop in spike removal') 

            face.remove(this) # remove the spike
            for i in sorted([rightIndx,leftIndx], reverse = True):
                del faces3D[i]
            faces3D.append(mergedFace)

            break   # break looping through faces and restart main while loop,
                    # because it is possible that new spikes have been generated

    # fix adjacent parallel edges in faces
    counts = Counter(chain.from_iterable(faces3D))
    for face in faces3D:
        if len(face) > 3:   # a triangle cant have parallel edges
            verticesToRemove = []
            for prev, this, _next in _iterCircularPrevThisNext(face):
                # Can eventually remove vertice, if it appears only in 
                # two adjacent faces, otherwise its a node.
                # But do not remove original polygon vertices.
                if counts[this] < 3 and this >= firstSkelIndex:
                    s0 = verts[this]-verts[prev]
                    s1 = verts[_next]-verts[this]
                    s0 = mathutils.Vector((s0[0],s0[1]))    # need 2D-vector
                    s1 = mathutils.Vector((s1[0],s1[1]))
                    s0m = s0.magnitude
                    s1m = s1.magnitude
                    if s0m!=0.0 and s1m!=0.0:
                        dotCosine = s0.dot(s1) / (s0m*s1m)
                        if abs(dotCosine - 1.0) < PARALLEL: # found adjacent parallel edges
                            verticesToRemove.append(this)
                    else:
                        if this not in verticesToRemove:    # duplicate vertice
                            verticesToRemove.append(this)   
            for item in verticesToRemove:
                face.remove(item) 

        # remove one of adjacent identical vertices
        verticesToRemove = []
        for prev,_next in _iterCircularPrevNext(face):
            if prev == _next:
                verticesToRemove.append(prev)
        for item in verticesToRemove:
            face.remove(item) 

    if faces is None:
        return faces3D
    else:
        faces.extend(faces3D)
        return faces
