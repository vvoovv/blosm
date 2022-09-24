from mathutils import Vector
from defs.way import allRoadwayCategoriesSet, allRailwayCategoriesSet


class WaySegment:
    
    __slots__ = (
        "way", "id1", "id2", "v1", "v2", "prev", "next",
        "length", "unitVector",
        "sumVisibility", "sumDistance", "avgDist",
        "id"
    )
    
    ID = 0
    
    def __init__(self, id1, v1, id2, v2, way):
        self.id1 = id1
        self.v1 = Vector((v1[0], v1[1]))
        self.id2 = id2
        self.v2 = Vector((v2[0], v2[1]))
        vector = self.v2 - self.v1
        self.length = vector.length
        self.unitVector = vector/self.length
        
        self.way = way
        self.avgDist = 0.
        self.sumDistance = 0.
        self.sumVisibility = 0.
        
        self.id = WaySegment.ID
        WaySegment.ID += 1


class Way:
    
    __slots__ = ("element", "category", "segments", "numSegments", "tunnel", "bridge")
    
    def __init__(self, element, manager):
        self.element = element
        
        highwayTag = element.tags.get("highway")
        self.category = highwayTag if highwayTag in allRoadwayCategoriesSet else "other"
        self.tunnel = "tunnel" in element.tags
        self.bridge = "bridge" in element.tags
    
    def init(self, manager):
        data = manager.data
        # segments
        self.segments = segments = tuple(
            WaySegment(
                nodeId1,
                data.nodes[nodeId1].getData(data),
                nodeId2,
                data.nodes[nodeId2].getData(data),
                self
            ) for nodeId1,nodeId2 in self.element.pairNodeIds(manager.data) \
                if not manager.data.haveSamePosition(nodeId1, nodeId2)
        )
        self.numSegments = len(self.segments)
        # set the previous and the next segment for each segment from <self.segments>
        if self.numSegments > 1:
            segments[0].next = segments[1]
            for i in range(1, self.numSegments-1):
                segments[i].prev = segments[i-1]
                segments[i].next = segments[i+1]
            segments[-1].prev = segments[-2]
        if self.element.closed:
            segments[0].prev = segments[-1]
            segments[-1].next = segments[0]
        else:
            segments[0].prev = None
            segments[-1].next = None

class Railway(Way):

    def __init__(self, element, manager):
        self.element = element
        
        railwayTag = element.tags.get("railway")
        self.category = railwayTag if railwayTag in allRailwayCategoriesSet else "other_railway"
        self.tunnel = "tunnel" in element.tags
        self.bridge = "bridge" in element.tags