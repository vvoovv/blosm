
class Segment():
    ID = 0  # just used during debugging
    def __init__(self, source, target, length, category):
        self.source = source
        self.target = target
        self.length = length
        self.category = category
        self.ID = Segment.ID
        Segment.ID += 1   # just used during debugging
    def __invert__(self): # just used during debugging
        # segment with the opposite direction
        return self.__class__(self.target, self.source, self.length, self.category)
    # def __eq__(self, other):
    #     # comparison of segments (no duplicates allowed)
    #     selfNodes = {self.source, self.target}
    #     return other.source in selfNodes and other.target in selfNodes



class WayNetwork(dict):
    def __init__(self):
        pass

    def addNode(self,node):
        # add a node to the graph
        if node not in self:
            self[node] = dict()

    def delNode(self, node):
        # remove a node from the graph
        for segment in list(self.iterInSegments(node)):
            self.delSegment(segment)
        del self[node]

    def hasNode(self, node):
        # test if an node exists
        return node in self

    def addSegment(self, source, target, length, category):
        # add a segment to the graph (missing nodes are created)
        segment = Segment(source,target,length,category)
        if source in self and target in self[source]:
            if segment in self[segment.source][segment.target]:
                return # segment already in network
        self.addNode(source)
        self.addNode(target)
        if target not in self[source]:
            self[source][target] = list()
        if source not in self[target]:
            self[target][source] = list()
        # Increase the number of parallel edges.
        self[source][target].append(segment)
        # A loop is added only once.
        if source != target:
            self[target][source].append(~segment)

    def delSegment(self,segment):
        # remove an edge from the graph
        self[segment.source][segment.target].remove(segment)
        if len(self[segment.source][segment.target]) == 0:
            del self[segment.source][segment.target]
        # A loop is deleted only once.
        if segment.source != segment.target:
            self[segment.target][segment.source].remove(~segment)
            if len(self[segment.target][segment.source]) == 0:
                del self[segment.target][segment.source]
       
    def getSegment(self, source, target):
        if source in self and target in self[source]:
            segment = self[source][target]
            return segment[0] if isinstance(segment,list) else segment
        else:
            return None

    def hasSegment(self, segment):
        # test if a segment exists
        return segment.source in self and segment.target in self[segment.source]

    def lengthOfSegment(self, segment):
        # return the segment length or zero
        source, target = segment.source, segment.target
        if source in self and target in self[source]:
            return self[source][target].length
        else:
            return 0

    def iterNodes(self):
        # generate all nodes from the graph on demand
        return iter(self)

    def iterInSegments(self, source):
        # generate the in-segments from the graph on demand
        for target in self[source]:
            yield self[target][source]

    def iterOutSegments(self, source):
        # generate the out-segments from the graph on demand
        for target in self[source]:
            yield self[source][target]

    def iterAdjacentNodes(self, source):
        # generate the adjacent nodes from the graph on demand
        return iter(self[source])

    def iterAdjacentSegments(self, source):
        # generate the adjacent segments from the graph on demand
        for segment in iter(self[source]):
                yield segment

    def iterAllSegments(self):
        # generate all segments from the graph on demand
        for source in self.iterNodes():
            for target in self[source]:
                if source < target:
                    yield self[source][target]

    def iterAllIntersectionNodes(self):
        for source in self.iterNodes():
            if len(self[source]) != 2:
                yield source

    def isIntersection(self,node):
        # includes end nodes
        return len(self[node]) != 2

    def iterAlongWay(self,segment):
        # follow way in the direction of the first segment until crossing occurs
        current = segment
        while True:
            if len(self[current.target]) != 2:
                break
            current = [self[current.target][source] for source in self[current.target] if source != current.source][0]
            current = current[0] if isinstance(current,list) else current
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
        # for indx, v in enumerate(self.vertices):
        #     v_cats = self.way_categories[indx]
        #     cats_set = set(v_cats)
        #     degree = len(v_cats)
        #     if degree != 2 or (degree==2 and len(cats_set) > 1):
        #         if categories_set & cats_set:
        #             found.append(indx)
        return found

# e = Edge('A','B',1.,'C1')
# ee = ~e
# graph = WayNetwork()
# graph.addEdge( 'A','B',1.,'C1')
# graph.addEdge( 'B','C',2.,'C2')
# graph.addEdge( 'C','D',10.,'C1')

# graph.addEdge( 'B','E',3.,'C2')
# graph.addEdge( 'E','F',4.,'C2')
# graph.addEdge( 'F','G',5.,'C2')
# graph.addEdge( 'G','H',6.,'C1')
# graph.addEdge( 'H','I',7.,'C2')

# graph.addEdge( 'I','J',8.,'C2')
# graph.addEdge( 'I','K',9.,'C2')

# from way_algorithms import createSectionNetwork

# sectionGraph = createSectionNetwork(graph)

# segment = graph.getEdge('B','E')
# for e in graph.iterAlongPath(segment):
#     if e.category != segment.category:
#         break
#     print(e.source,e.target,e.length,e.category)


# test = 1
# ways = graph.getNodesForCategories(['primary'])
# neighbors = graph.getNeighbors((1.54,5.12))
# degree = graph.getDegree((7.82,1.11))
# crossings = graph.getAllCrossings()

# from way_algorithms import createSectionNetwork

# sectionNetwork = createSectionNetwork(graph)