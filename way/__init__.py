import numpy


allWayCategories = set((
    "other",
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "living_street",
    "service",
    "pedestrian",
    "track",
    "escape",
    "raceway",
    # "road", # other
    "footway",
    "bridleway",
    "steps",
    "path",
    "cycleway"
))


class Category:
    __slots__ = tuple()
    @staticmethod
    def addCategories():
        for category in allWayCategories:
            setattr(Category, category, category)
Category.addCategories()


class WaySegment:
    
    __slots__ = ("way", "id1", "id2", "v1", "v2", "prev", "next")
    
    def __init__(self, id1, v1, id2, v2, way):
        self.id1 = id1
        self.v1 = numpy.array((v1[0], v1[1]))
        self.id2 = id2
        self.v2 = numpy.array((v2[0], v2[1]))
        self.way = way

    def getSegmentInfo(self):
        # segmentCenter, segmentUnitVector, segmentLength
        segmentVector = self.v2 - self.v1
        segmentLength = numpy.linalg.norm(segmentVector)
        return (self.v1 + self.v2)/2., segmentVector/segmentLength, segmentLength


class Way:
    
    __slots__ = ("element", "category", "segments", "numSegments", "tunnel", "bridge")
    
    def __init__(self, element, manager):
        self.element = element
        data = manager.data
        
        highwayTag = element.tags.get("highway")
        self.category = highwayTag if highwayTag in allWayCategories else "other"
        self.tunnel = "tunnel" in element.tags
        self.bridge = "bridge" in element.tags
        
        # segments
        self.segments = segments = tuple(
            WaySegment(
                nodeId1,
                data.nodes[nodeId1].getData(data),
                nodeId2,
                data.nodes[nodeId2].getData(data),
                self
            ) for nodeId1,nodeId2 in element.pairNodeIds(manager.data) \
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