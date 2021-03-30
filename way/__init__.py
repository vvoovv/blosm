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


class RealWay:
    
    __slots__ = ("element", "polyline", "category", "tunnel", "bridge")
    
    def __init__(self, element):
        self.element = element
        RealWay.initOsm(self, element)
        self.polyline = None
    
    def segments(self, data):
        # segmentCenter, segmentUnitVector, segmentLength
        coord0 = coord1 = None
        for coord2 in ( numpy.array((coord[0], coord[1])) for coord in self.element.getData(data) ):
            if coord1 is None:
                coord0 = coord2
            else:
                segmentVector = coord2 - coord1
                segmentLength = numpy.linalg.norm(segmentVector)
                yield (coord1 + coord2)/2., segmentVector/segmentLength, segmentLength
            coord1 = coord2
        if self.element.isClosed():
            segmentVector = coord0 - coord1
            segmentLength = numpy.linalg.norm(segmentVector)
            yield (coord1 + coord0)/2., segmentVector/segmentLength, segmentLength
    
    @staticmethod
    def initOsm(way, element):
        highwayTag = element.tags.get("highway")
        way.category = highwayTag if highwayTag in allWayCategories else "other"
        way.tunnel = "tunnel" in element.tags
        way.bridge = "bridge" in element.tags


#_osmHighwayToCategory