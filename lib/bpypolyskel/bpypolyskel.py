# -*- coding: utf-8 -*-

"""
https://github.com/prochitecture/bpypolyskel

Implementation of the straight skeleton algorithm as described by Felkel and Obdržálek in their 1998 conference paper 
'Straight skeleton implementation'.

The code for skeletonize() has been ported from the implementation by Botffy at https://github.com/Botffy/polyskel,
in order to be able to use it in Blender. The main changes are:

-   The order of the vertices of a polygon has been changed to a right-handed coordinate system (as used in Blender).
    The positive x and y axes point right and up, and the z axis points into your face. Positive rotation is 
    counterclockwise about the z-axis.

-   The vector objects used from the library euclid3 have been replaced by objects from the library 
    mathutils, for the mathematical computations. These are defined in the new library bpyeuclid.

-   The class 'Debug' and all the calls to it have been removed, as an image can't be drwan in Blender

-   The signature of skeletonize() has been changed to lists of edges for the polygon and eventual hole. These are of
    type Edge2, defined in bpyeuclid. 
    
-   A new function polygonize() has been added. It creates a list of faces from the skeletonized polygon.
    For the definiton of the signature, see the description at the function definition.
"""

import heapq
from collections import namedtuple
from itertools import *
from collections import Counter

from .bpyeuclid import *
from .poly2FacesGraph import poly2FacesGraph

EPSILON = 0.00001
PARALLEL = 0.01     # set this value to 1-cos(alpha), where alpha is the largest angle 
                    # between lines to accept them as parallelaccepted as 'parallel'.

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

                    if not (xprev and xnext and xedge):
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
        for prev, next in _iterCircularPrevNext(edgeContour):
            # V(i) is the current vertex
            # prev is the edge from vertex V(i-1) to V(i)
            # this is the edge from vertex V(i-) to V(i+1)
            lav._len += 1
            vertex = _LAVertex(next.p1, prev, next)
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
        cur = self.head
        while True:
            yield cur
            cur = cur.next
            if cur == self.head:
                return

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
        samePositionList = []
        # from top of queue, get all events that have the same distance as the one on top
        while self.__data and robustFloatEqual( self.__data[0].distance, item.distance):
            queueTop = heapq.heappop(self.__data)
            # don't extract queueTop if identical position with item
            if _approximately_equals(queueTop.intersection_point,item.intersection_point ):
                samePositionList.append(queueTop)
            else:
                equalDistanceList.append(queueTop)
        self.put_all(samePositionList)
        return equalDistanceList

    def empty(self):
        return not self.__data

    def peek(self):
            return self.__data[0]

    def show(self):
        for item in self.__data:
            print(item)

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

def mergeNodeClusters(skeleton,mergeRange = 0.1):
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

    # sort arcs by x-position of nodes
    skeleton = sorted(skeleton, key=lambda arc: arc.source.x )

    # find pairs of nodes where the sources are in a square of size mergeRange.
    # the entry in 'candidates' is the index of the node in 'skeleton'
    candidates = []
    # for i,pair in enumerate(_iterCircularPrevNext(range(len(skeleton)))):
    combs = combinations(range(len(skeleton)),2)
    for pair in combs:
        distx = abs(skeleton[pair[1]].source.x - skeleton[pair[0]].source.x)
        disty = abs(skeleton[pair[1]].source.y - skeleton[pair[0]].source.y)
        if distx<mergeRange and disty<mergeRange:
            candidates.extend(pair)

    # check if there are cluster candidates
    if not candidates:
        return skeleton

    # remove duplicates
    candidates = list(dict.fromkeys(candidates))

    # distances between canddates
    dist = [(skeleton[p1].source-skeleton[p0].source).magnitude for p0,p1 in zip(candidates,candidates[1:]) ]

    # classify them, 1: dist<5*mergeRange  2: dist>=5*mergeRange
    classes = [ (i,1 if d<5*mergeRange else 2) for i,d in enumerate(dist)]

    # clusters are groups of consecutive elements of class 1
    clusters = []
    nodesToMerge = []
    for key, group in groupby(classes, lambda x: x[1]):
        if key==1:
            indx = [tup[0] for tup in list(group)]
            cluster = candidates[ indx[0]:(indx[-1]+2) ]
            clusters.append(cluster)
            nodesToMerge.extend(cluster)

    # find new centers of merged clusters as center of gravity.
    # in the same time, collect all sinks of the merged nodes
    newNodes = []
    for cluster in clusters:
        # compute center of gravity as source of merged node
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

        # collect all sinks of merged nodes that are not in set of merged nodes
        new_sinks = []
        for node in cluster:
            for sink in skeleton[node].sinks:
                if sink not in mergedSources and sink not in new_sinks:
                    new_sinks.append(sink)

        # create the merged node and remember it for later use
        newnode = Subtree(new_source, new_height, new_sinks)
        newNodes.append(newnode)

       # redirect all sinks that pointed to one of the clustered nodes to the new node
        for arc in skeleton:
            if arc.source not in mergedSources:
                for i,sink in enumerate(arc.sinks):
                    if sink in mergedSources:
                        arc.sinks[i] = new_source

        # redirect eventual sinks of new nodes that point to one of the clustered nodes to the new node
        for arc in newNodes:
            if arc.source not in mergedSources: #???
                for i,sink in enumerate(arc.sinks):
                    if sink in mergedSources:
                        arc.sinks[i] = new_source

    # remove clustered nodes from skeleton
    # and add new nodes
    for i in sorted(nodesToMerge, reverse = True):
        del skeleton[i]
    skeleton.extend(newNodes)

    return skeleton

# def isEventInItsLAV(event):
#     point = event.intersection_point    # vertice of event
#     lav = event.vertex.lav              # LAV of event
#     pv = [v.point for v in lav]         # vertices in LAV
#     # signed area of a polygon. If > 0 -> polygon counterclockwise.
#     # See: https://mathworld.wolfram.com/PolygonArea.html 
#     signedArea = 0.0
#     # a ray from event along x-axis crosses odd edges, if inside polygon
#     isInLAV = False
#     for p,n  in _iterCircularPrevNext(pv):
#         signedArea += p.x*n.y - n.x*p.y
#         if intersect(p,n,point,point+mathutils.Vector((1.e9,0.0))):
#             isInLAV = not isInLAV
#     if signedArea < 0.0:
#         isInLAV = not isInLAV
#     return isInLAV

def skeletonize(edgeContours,mergeRange=0.1):
    """
    Compute the straight skeleton of a polygon.

    The polygon is expected as a list of vertices in counterclockwise order. In a right-handed coordinate system, 
    seen from top, the polygon is on the left of its contour. Holes are expected as lists of vertices in clockwise order.
    Seen from top, the polygon is on the right of the hole's contour.

    The argument 'edgeContours' is expected to as:
        edgeContours = [ polygon_edges, <hole1_edges>, <hole2_edges>, ...]

    'polygon_egdes' is a list of the edges of the polygon in counterclockwise order: [ egde0, edge1, ...]
    'hole_edges' is a list of the edges of a hole in clockwise order: [ egde0, edge1, ...]

    Returns the straight skeleton as a list of "subtrees", which are in the form of (source, height, sinks),
    where source is the highest points, height is its height, and sinks are the points connected to the source.
    """
    slav = _SLAV(edgeContours)

    output = []
    prioque = _EventQueue()

    for lav in slav:
        for vertex in lav:
            prioque.put(vertex.next_event())

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
                # if not isEventInItsLAV(i):
                #     continue
                (arc, events) = slav.handle_split_event(i)
            prioque.put_all(events)

            if arc is not None:
                output.append(arc)

    output = mergeNodeClusters(output,mergeRange)
    removeGhosts(output)

    return output

def polygonize(verts, firstVertIndex, numVerts, holesInfo=None, height=0., tan=0., faces=None, unitVectors=None,mergeRange=0.1):
    """
    Compute the faces of a polygon, skeletonized by a straight skeleton.

    The polygon is expected as a list of vertices in counterclockwise order. In a right-handed coordinate system, 
    seen from top, the polygon is on the left of its contour. Holes are expected as lists of vertices in clockwise order.
    Seen from top, the polygon is on the right of the hole's contour.

    Arguments:
    ----------
    verts:              A list of vertices. Vertices that define the polygon and possibly its holes are located at the end
                        of the 'verts' list starting at the index 'firstVertIndex'. Each vertex is an instance of 
                        mathutils.Vector with 3 coordinates. The z-coordinate is the same for all vertices of the polygon.

                        For the polygon without holes, there are 'numVerts' vertices in the counterclockwise order.

                        For the polygon with holes there are:
                            - 'numVerts' vertices in counterclockwise order for the outer contour
                            - 'numVertsHoles[0]' vertices in clockwise order for the first hole
                            - 'numVertsHoles[1]' vertices in clockwise order for the second hole
                            ...
                            - 'numVertsHoles[-1]' vertices in clockwise order for the last hole

                        'verts' gets extended by the vertices of the straight skeleton by 'polygonize()'.
                    
    firstVertIndex:     The first index of vertices in the verts list that define the polygon and possibly its holes.

    numVerts:           The number of vertices in the polygon for the one without holes.
                        The number of vertices in the outer contour of the polygon for the one with holes.

    numVertsHoles:      A Python tuple or list. The elements define the number of the vertices in the related hole.


    height:             The maximum height of the hipped roof to be generated. If both 'height' and 'tan' are equal to zero,
                        flat faces are generated. 'height' takes precedence over 'tan' if both have a non-zero value.

    tan:                It's desirable in many case to deal with roof pitch angle instead of maximum roof height. The tangent
                        of the roof pitch angle can be supplied for that case. If both 'height' and 'tan' are equal to zero, flat
                        faces are generated. height takes precedence over tan if both have a non-zero value.

    faces:              A Python list of the resulting faces formed by the straight skeleton. If it is given, the faces formed
                        by the straight skeleton of the new polygon are appended to it and get returned by this function.
                        Otherwise a new Python list of faces is created and returned by this function..

    unitVectors:        A Python list of unit vectors along the polygon edges (including holes if they are present). These vectors
                        are of type mathutils.Vector with three dimensions. The direction of the vectors corresponds to order of
                        the vertices in the polygon and its holes. The order of the unit vectors in the 'unitVectors' list corresponds
                        to the order of vertices in the input Python list 'verts'. The Python list 'unitVectors' (if given) gets used
                        inside the 'polygonize(..)' function instead of calculating it once more. If the list is not given, the unit
                        vectors get calculated inside the 'polygonize(..)' function.

    Returned Value:
    --------------
                        A list of the faces with the indices of vertices in 'verts'. The faces are formed by the straight skeleton.
                        The order of vertices of the faces is counterclockwise, as the order of vertices in the input Python list 'verts'.
                        The first edge of a face is always an edge of the polygon or its holes.

                        For example, suppose one has

                            verts = [..., v1, v2, v3, ...]

                        Then in the return list gets

                            faces = [
                                ..,
                                (index_of_v1, index_of_s2, ...),
                                (index_of_v2, index_of_v3, ...),
                                (index_of_v3, index_of_v4, ...),
                                ...
                            ]
    """
    # assume that all vertices of polygon and holes have the same z-value
    zBase = verts[firstVertIndex][2]

    # compute center of gravity of polygon
    center = mathutils.Vector((0.0,0.0,0.0))
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

    nrOfEdges = len(edges2D)

	# compute skeleton
    skeleton = skeletonize(edgeContours,mergeRange)

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

    if faces is None:
        return faces3D
    else:
        faces.extend(faces3D)
        return faces