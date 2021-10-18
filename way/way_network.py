from mathutils import Vector
from defs.way import allRoadwayCategoriesRank

class NetSegment():
    ID = 0  # just used during debugging

    def __init__(self, *args):
        self.ID = NetSegment.ID # just used during debugging
        NetSegment.ID += 1      # just used during debugging
        if len(args) > 1:
            self.initFromDetails(*args)
        else:
            self.initFromOther(*args)

    def initFromDetails(self, source, target, category, length=None, path=None):
        self.s = Vector(source).freeze()    # source node
        self.t = Vector(target).freeze()    # target node
        self.category = category
        self.geomLength = (self.t-self.s).length
        if length:
            self.length = length
        else:
            self.length = self.geomLength
        if path:
            self.path = path        # includes source and target
        else:
            self.path = [self.s,self.t]

    def initFromOther(self, other):
        self.s = other.s    # source node
        self.t = other.t    # target node
        self.category = other.category
        self.geomLength = other.geomLength
        self.length = other.length
        self.path = other.path        # includes source and target

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

    def __invert__(self):
        # create segment with the reversed direction
        return self.__class__(self.t, self.s, self.category, self.length, self.path[::-1])
    def __eq__(self, other):
        # comparison of segments (no duplicates allowed)
        selfNodes = {self.s, self.t}
        selfPaths = [self.path, self.path.reverse()]
        return other.s in selfNodes and other.t in selfNodes and other.path in selfPaths
    # def __hash__(self):
    #     return hash((self.source, self.target, self.length))



class WayNetwork(dict):
    # undirected multigraph

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
        if s != t:  # a loop is added only once
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
            return segment[0] if isinstance(segment,list) else segment
        else:
            return None        
       
    def iterNodes(self):
        # generator for all nodes from the network
        return iter(self)

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
                # if source > target: ??
                    for segment in self[source][target]:
                        yield segment

    def iterAllIntersectionNodes(self):
        # iterator for all nodes that form an intersection
        # includes also end-nodes!
        for source in self.iterNodes():
            if len(self[source]) != 2: # order of node != 2
                yield source

    def iterAlongWay(self,segment):
        # generator for nodes that follow the way in the direction given
        # by the first <segment>, until a crossing occurs
        current = segment
        while True:
            if len(self[current.t]) != 2: # order of node != 2
                break
            current = [self[current.t][source] for source in self[current.t] if source != current.s][0][0]
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

