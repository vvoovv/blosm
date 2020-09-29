# -*- coding: utf-8 -*-

"""
Implementation of the straight skeleton algorithm as described by Felkel and Obdržálek in their 1998 conference paper Straight skeleton implementation.

The code for skeletonize() has been ported from the implementation by Botffy at https://github.com/Botffy/polyskel.
In order to be able to use it in Blender, the following changes have been made:

- The vector objects used from the library euclid3 have been replaced by objects from the library 
  mathutils for the mathematical computations. These are defined in the new library bpyeuclid.

- Changes within the code realtive to Botffy's original are commented as '*** adapted for bpypolyskel'.

- The class Debug and all calls to it have been removed, as an image can't be drwan in Blender

- The signature of skeletonize() has been changed to a list of contours of the polygon and eventually holes. The polygon
  is expected as a list of vertices in clockwise order. Seen from top, the polygon is on the right of its contour. Holes
  are expected as lists of vertices in counter-clockwise order. Seen from top, the polygon is on the left of the hole 
  contour.
  The argument 'contours' is expected to as:
		contours = [ polygon_list, <hole1_list>, <hole2_list>, ...]

- A new function polygonize() has been added. See its description in the repository https://github.com/prochitecture/bpypolyskel
"""

import logging
import heapq
# *** adapted for bpypolyskel
# from euclid3 import *
from .bpyeuclid import *
from itertools import *
from collections import namedtuple
from .poly2FacesGraph import poly2FacesGraph

log = logging.getLogger("__name__")

EPSILON = 0.00001

def _window(lst):
	prevs, items, nexts = tee(lst, 3)
	prevs = islice(cycle(prevs), len(lst) - 1, None)
	nexts = islice(cycle(nexts), 1, None)
	return zip(prevs, items, nexts)

# *** adapted for bpypolyskel: replaced by mathutils.Vector's cross function
# def _cross(a, b):
# 	res = a.x * b.y - b.x * a.y
# 	return res


def _approximately_equals(a, b):
	# return a == b or (abs(a - b) <= max(abs(a), abs(b)) * 0.001)
	# a and b are assumed to be of type mathutils.Vector
	return a == b or ( (a-b).magnitude <= max( a.magnitude, b.magnitude) * 0.001)


def _approximately_same(point_a, point_b):
	return _approximately_equals(point_a.x, point_b.x) and _approximately_equals(point_a.y, point_b.y)


# *** adapted for bpypolyskel: normalization expected to be done by caller
# def _normalize_contour(contour):
# 	contour = [Point2(float(x), float(y)) for (x, y) in contour]
# 	return contour
#	return [point for prev, point, next in _window(contour) if not (point == next or (point-prev).normalized() == (next - point).normalized())]


class _SplitEvent(namedtuple("_SplitEvent", "distance, intersection_point, vertex, opposite_edge")):
	__slots__ = ()

	def __lt__(self, other):
		return self.distance < other.distance

	def __str__(self):
		return "{} Split event @ {} from {} to {}".format(self.distance, self.intersection_point, self.vertex, self.opposite_edge)


class _EdgeEvent(namedtuple("_EdgeEvent", "distance intersection_point vertex_a vertex_b")):
	__slots__ = ()

	def __lt__(self, other):
		return self.distance < other.distance

	def __str__(self):
		return "{} Edge event @ {} between {} and {}".format(self.distance, self.intersection_point, self.vertex_a, self.vertex_b)


_OriginalEdge = namedtuple("_OriginalEdge", "edge bisector_left, bisector_right")

Subtree = namedtuple("Subtree", "source, height, sinks")

class _LAVertex:
	def __init__(self, point, edge_left, edge_right, direction_vectors=None):
		self.point = point
		self.edge_left = edge_left
		self.edge_right = edge_right
		self.prev = None
		self.next = None
		self.lav = None
		self._valid = True  # TODO this might be handled better. Maybe membership in lav implies validity?

		creator_vectors = (edge_left.norm * -1, edge_right.norm)
		if direction_vectors is None:
			direction_vectors = creator_vectors

		dv0 = direction_vectors[0]
		dv1 = direction_vectors[1]

		# *** adapted for bpypolyskel
		# self._is_reflex = (_cross(*direction_vectors)) < 0
		self._is_reflex = dv0.cross(dv1) < 0#dv0.cross(dv1) < 0 
		# *** adapted for bpypolyskel
		# self._bisector = Ray2(self.point, operator.add(*creator_vectors) * (-1 if self.is_reflex else 1))
		op_add_result = creator_vectors[0] + creator_vectors[1]
		self._bisector = Ray2(self.point, op_add_result * (-1 if self.is_reflex else 1))
		log.info("Created vertex %s", self.__repr__())

	@property
	def bisector(self):
		return self._bisector

	@property
	def is_reflex(self):
		return self._is_reflex

	@property
	def original_edges(self):
		return self.lav._slav._original_edges

	def next_event(self):
		events = []
		if self.is_reflex:
			# a reflex vertex may generate a split event
			# split events happen when a vertex hits an opposite edge, splitting the polygon in two.
			log.debug("looking for split candidates for vertex %s", self)
			for edge in self.original_edges:
				if edge.edge == self.edge_left or edge.edge == self.edge_right:
					continue

				log.debug("\tconsidering EDGE %s", edge)

				# a potential b is at the intersection of between our own bisector and the bisector of the
				# angle between the tested edge and any one of our own edges.

				# we choose the "less parallel" edge (in order to exclude a potentially parallel edge)
				leftdot = abs(self.edge_left.norm.dot(edge.edge.norm))
				rightdot = abs(self.edge_right.norm.dot(edge.edge.norm))
				selfedge = self.edge_left if leftdot < rightdot else self.edge_right
				# otheredge = self.edge_left if leftdot > rightdot else self.edge_right		# *** adapted for bpypolyskel

				i = Line2(selfedge).intersect(Line2(edge.edge))
				if i is not None and not _approximately_equals(i, self.point):
					# locate candidate b
					linvec = (self.point - i).normalized()
					edvec = edge.edge.norm
					if linvec.dot(edvec) < 0:
						edvec = -edvec

					bisecvec = edvec + linvec
					if bisecvec.magnitude == 0:
						continue
					bisector = Line2(i, bisecvec, 'pv')
					b = bisector.intersect(self.bisector)

					if b is None:
						continue

					# check eligibility of b
					# a valid b should lie within the area limited by the edge and the bisectors of its two vertices:
					# *** adapted for bpypolyskel
					# xleft	= _cross(edge.bisector_left.v.normalized(), (b - edge.bisector_left.p).normalized()) > EPSILON
					# xright= _cross(edge.bisector_right.v.normalized(), (b - edge.bisector_right.p).normalized()) < -EPSILON
					# xedge	= _cross(edge.edge.v.normalized(), (b - edge.edge.p).normalized()) < -EPSILON
					xleft	= ( (edge.bisector_left.v.normalized()).cross( (b - edge.bisector_left.p).normalized() )) > EPSILON
					xright	= ( (edge.bisector_right.v.normalized()).cross( (b - edge.bisector_right.p).normalized() )) < -EPSILON
					xedge	= ( edge.edge.norm.cross( (b - edge.edge.p1).normalized() )) < -EPSILON

					if not (xleft and xright and xedge):
						log.debug("\t\tDiscarded candidate %s (%s-%s-%s)", b, xleft, xright, xedge)
						continue

					log.debug("\t\tFound valid candidate %s", b)
					events.append(_SplitEvent(Line2(edge.edge).distance(b), b, self, edge.edge))

		i_prev = self.bisector.intersect(self.prev.bisector)
		i_next = self.bisector.intersect(self.next.bisector)

		if i_prev is not None:
			events.append(_EdgeEvent(Line2(self.edge_left).distance(i_prev), i_prev, self.prev, self))
		if i_next is not None:
			events.append(_EdgeEvent(Line2(self.edge_right).distance(i_next), i_next, self, self.next))

		if not events:
			return None

		# *** adapted for bpypolyskel
		# ev = min(events, key=lambda event: self.point.distance(event.intersection_point))
		ev = min(events, key=lambda event: (self.point-event.intersection_point).length)

		log.info("Generated new event for %s: %s", self, ev)
		return ev

	def invalidate(self):
		if self.lav is not None:
			self.lav.invalidate(self)
		else:
			self._valid = False

	@property
	def is_valid(self):
		return self._valid

	def __str__(self):
		return "Vertex ({:.2f};{:.2f})".format(self.point.x, self.point.y)

	def __repr__(self):
		return "Vertex ({}) ({:.2f};{:.2f}), bisector {}, edges {} {}".format("reflex" if self.is_reflex else "convex",
																			  self.point.x, self.point.y, self.bisector,
																			  self.edge_left, self.edge_right)


class _SLAV:
	# *** adapted for bpypolyskel
	# use directly edgeContours instead of polygon and holes,
	# assume that normalization has already been node
	def __init__(self, edgeContours):#polygon, holes):
		# *** adapted for bpypolyskel
		# contourtest = [_normalize_contour(polygon)]
		# contourtest.extend([_normalize_contour(hole) for hole in holes])
		# contours = [polygon]
		# for hole in holes:
		# 	contours.extend([hole])

		self._lavs = [_LAV.from_polygon(edgeContour, self) for edgeContour in edgeContours]

		# store original polygon edges for calculating split events
		self._original_edges = [
			# _OriginalEdge(Edge2(vertex.prev.point, vertex.point), vertex.prev.bisector, vertex.bisector)
			_OriginalEdge(vertex.edge_left, vertex.prev.bisector, vertex.bisector)
			for vertex in chain.from_iterable(self._lavs)
		]

	def __iter__(self):
		for lav in self._lavs:
			yield lav

	def __len__(self):
		return len(self._lavs)

	def empty(self):
		return len(self._lavs) == 0

	def handle_edge_event(self, event):
		sinks = []
		events = []

		lav = event.vertex_a.lav
		if event.vertex_a.prev == event.vertex_b.next:
			log.info("%.2f Peak event at intersection %s from <%s,%s,%s> in %s", event.distance,
					 event.intersection_point, event.vertex_a, event.vertex_b, event.vertex_a.prev, lav)
			self._lavs.remove(lav)
			for vertex in list(lav):
				sinks.append(vertex.point)
				vertex.invalidate()
		else:
			log.info("%.2f Edge event at intersection %s from <%s,%s> in %s", event.distance, event.intersection_point,
					 event.vertex_a, event.vertex_b, lav)
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
		log.info("%.2f Split event at intersection %s from vertex %s, for edge %s in %s", event.distance,
				 event.intersection_point, event.vertex, event.opposite_edge, lav)

		sinks = [event.vertex.point]
		vertices = []
		x = None  # right vertex
		y = None  # left vertex
		norm = event.opposite_edge.norm
		for v in chain.from_iterable(self._lavs):
			log.debug("%s in %s", v, v.lav)
			if norm == v.edge_left.norm and event.opposite_edge.p1 == v.edge_left.p1:
				x = v
				y = x.prev
			elif norm == v.edge_right.norm and event.opposite_edge.p1 == v.edge_right.p1:
				y = v
				x = y.next

			if x:
				# *** adapted for bpypolyskel
				# xleft	= _cross(y.bisector.v.normalized(), (event.intersection_point - y.point).normalized()) >= EPSILON
				# xright= _cross(x.bisector.v.normalized(), (event.intersection_point - x.point).normalized()) <= -EPSILON
				xleft	= (y.bisector.v.normalized()).cross((event.intersection_point - y.point).normalized()) >= EPSILON
				xright	= (x.bisector.v.normalized()).cross((event.intersection_point - x.point).normalized()) <= -EPSILON
				log.debug("Vertex %s holds edge as %s edge (%s, %s)", v, ("left" if x == v else "right"), xleft, xright)

				if xleft and xright:
					break
				else:
					x = None
					y = None

		if x is None:
			log.info("Failed split event %s (equivalent edge event is expected to follow)", event)
			return (None, [])

		v1 = _LAVertex(event.intersection_point, event.vertex.edge_left, event.opposite_edge)
		v2 = _LAVertex(event.intersection_point, event.opposite_edge, event.vertex.edge_right)

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
			log.debug(l)
			if len(l) > 2:
				self._lavs.append(l)
				vertices.append(l.head)
			else:
				log.info("LAV %s has collapsed into the line %s--%s", l, l.head.point, l.head.next.point)
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


class _LAV:
	def __init__(self, slav):
		self.head = None
		self._slav = slav
		self._len = 0
		log.debug("Created LAV %s", self)

	@classmethod
	def from_polygon(cls, edgeContour, slav):
		lav = cls(slav)
		for prev, item, next in _window(edgeContour):
			lav._len += 1
			# vertex = _LAVertex(point, LineSegment2(prev, point), LineSegment2(point, next))
			vertex = _LAVertex(item.p1, prev, item)
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
		log.debug("Invalidating %s", vertex)
		vertex._valid = False
		if self.head == vertex:
			self.head = self.head.next
		vertex.lav = None

	def unify(self, vertex_a, vertex_b, point):
		replacement = _LAVertex(point, vertex_a.edge_left, vertex_b.edge_right,
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

	def __str__(self):
		return "LAV {}".format(id(self))

	def __repr__(self):
		return "{} = {}".format(str(self), [vertex for vertex in self])

	def __len__(self):
		return self._len

	def __iter__(self):
		cur = self.head
		while True:
			yield cur
			cur = cur.next
			if cur == self.head:
				return

	def _show(self):
		cur = self.head
		while True:
			print(cur.__repr__())
			cur = cur.next
			if cur == self.head:
				break


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

	def empty(self):
		return len(self.__data) == 0

	def peek(self):
		return self.__data[0]

	def show(self):
		for item in self.__data:
			print(item)

def _merge_sources(skeleton):
	"""
	In highly symmetrical shapes with reflex vertices multiple sources may share the same 
	location. This function merges those sources.
	"""
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

			
# *** adapted for bpypolyskel
# use directly edgeContours instead of polygon and holes,
def skeletonize(edgeContours):#polygon, holes=None):
	"""
	Compute the straight skeleton of a polygon.

	The polygon is expected as a list of vertices in clockwise order. Seen from top, the polygon is on the right of its contour. Holes
	are expected as lists of vertices in counter-clockwise order. Seen from top, the polygon is on the left of the hole 
	contour.
	The argument 'contours' is expected to as:
		contours = [ polygon_list, <hole1_list>, <hole2_list>, ...]

	Returns the straight skeleton as a list of "subtrees", which are in the form of (source, height, sinks),
	where source is the highest points, height is its height, and sinks are the point connected to the source.
	"""
	# *** adapted for bpypolyskel
	# use directly contours instead of polygon and holes,
	slav = _SLAV(edgeContours)#polygon, holes)
	output = []
	prioque = _EventQueue()

	for lav in slav:
		for vertex in lav:
			prioque.put(vertex.next_event())

	while not (prioque.empty() or slav.empty()):
		log.debug("SLAV is %s", [repr(lav) for lav in slav])
		i = prioque.get()
		if isinstance(i, _EdgeEvent):
			if not i.vertex_a.is_valid or not i.vertex_b.is_valid:
				log.info("%.2f Discarded outdated edge event %s", i.distance, i)
				continue

			(arc, events) = slav.handle_edge_event(i)
		elif isinstance(i, _SplitEvent):
			if not i.vertex.is_valid:
				log.info("%.2f Discarded outdated split event %s", i.distance, i)
				continue
			(arc, events) = slav.handle_split_event(i)

		prioque.put_all(events)

		if arc is not None:
			output.append(arc)

	_merge_sources(output)
	return output

def polygonize(verts, firstVertIndex, numVerts, unitVectors=None, numVertsHoles=None, height=0., tan=0., faces=None):
	# assume that all vertices of polygon and holes have the same z-value
	zBase = verts[firstVertIndex].z

	# create 2D edges as list and as contours for skeletonization and graph construction
	vIndex = firstVertIndex
	poly = verts[vIndex:vIndex+numVerts] # required to construct pairs of vertices
	if unitVectors is None:
		edges2D = [Edge2(p1,p2,(p2-p1).normalized()) for p1,p2 in zip(poly, poly[1:] + poly[:1])]
	else:
		edges2D = [Edge2(p1,p2,norm) for p1,p2,norm in zip(poly, poly[1:] + poly[:1], unitVectors[vIndex:])]
	edgeContours = [edges2D.copy()]

	vIndex += numVerts
	if numVertsHoles is not None:
		for numHole in numVertsHoles:
			hole = verts[vIndex:vIndex+numHole] # required to construct pairs of vertices
			if unitVectors is None:
				holeEdges = [Edge2(p1,p2,(p2-p1).normalized()) for p1,p2 in zip(hole, hole[1:] + hole[:1])]
			else:
				holeEdges = [Edge2(p1,p2,norm) for p1,p2,norm in zip(hole, hole[1:] + hole[:1], unitVectors[vIndex:])]
			edges2D.extend(holeEdges)
			edgeContours.extend([holeEdges.copy()])
			vIndex += numHole
	
	nrOfEdges = len(edges2D)

	# compute skeleton
	skeleton = skeletonize(edgeContours)

	# compute skeleton node heights and append nodes to original verts list,
	# see also issue #4 at https://github.com/prochitecture/bpypolyskel
	if height != 0.:
		tan_alpha = height/skeleton[-1].height  # the last skeleton node is highest
	else:
		tan_alpha = tan
	skeleton_nodes3D = []
	for arc in skeleton:
		node = mathutils.Vector((arc.source.x,arc.source.y,arc.height*tan_alpha+zBase))
		skeleton_nodes3D.append(node)
	firstSkelIndex = len(verts) # first skeleton index in verts
	verts.extend(skeleton_nodes3D)

	# instantiate the graph for faces
	graph = poly2FacesGraph()

	# add polygon and hole indices to graph using indices in verts
	vIndex = firstVertIndex
	_L = list(range(vIndex,vIndex+numVerts))
	for edge in zip(_L, _L[1:] + _L[:1]):
		graph.add_edge(edge)
	vIndex += numVerts
	if numVertsHoles is not None:
		for numHole in numVertsHoles:
			_L = list(range(vIndex,vIndex+numHole))
			for edge in zip(_L, _L[1:] + _L[:1]):
				graph.add_edge(edge)
			vIndex += numHole

	# add skeleton edges to graph using indices in verts
	for index, arc in enumerate(skeleton):
		aIndex = index + firstSkelIndex
		for sink in arc.sinks:
			# first search in input edges
			edgIndex = [index for index, edge in enumerate(edges2D) if edge.p1==sink]
			if len(edgIndex):
				sIndex = edgIndex[0] + firstVertIndex
			else: # then it should be a skeleton node
				skelIndex = [index for index, arc in enumerate(skeleton) if arc.source==sink]
				if len(skelIndex):
					sIndex = skelIndex[0] + firstSkelIndex
				else:
					sIndex = -1 # error
			graph.add_edge( (aIndex,sIndex) )
			
	# generate clockwise circular embedding
	embedding = graph.circular_embedding(verts,'CW')

	# compute list of faces, the vertex indices are still related to verts2D
	faces3D = graph.faces(embedding, nrOfEdges+firstVertIndex)

	if faces is None:
		return faces3D
	else:
		faces.extend(faces3D)
		return faces