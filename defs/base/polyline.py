from mathutils import Vector


class Polyline:
    
    def __init__(self, element):
        self.element = element
    
    def init(self, manager):
        # edges
        self.edges = [
            self.createEdge(nodeId1, nodeId2, manager.data) \
                for nodeId1,nodeId2 in self.element.pairNodeIds(manager.data) \
                    if not manager.data.haveSamePosition(nodeId1, nodeId2)
        ]
        self.numEdges = len(self.edges)
    
    def createEdge(self, nodeId1, nodeId2, data): 
        return Edge(
            nodeId1,
            data.nodes[nodeId1].getData(data),
            nodeId2,
            data.nodes[nodeId2].getData(data)
        )
    
    def isArea(self):
        tags = self.element.tags
        return self.element.isClosed() and ( "building" in tags or "landuse" in tags or "natural" in tags ) 


class Edge:
    
    def __init__(self, id1, v1, id2, v2):
        self.id1 = id1
        self.v1 = Vector(v1)
        self.id2 = id2
        self.v2 = Vector(v2)
        self._length = None
    
    @property
    def length(self):
        # calculation on demand
        if not self._length:
            self._length = (self.v2 - self.v1).length
        return self._length
        
    
    