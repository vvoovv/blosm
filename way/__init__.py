import numpy


class RealWay:
    
    motorway = 1
    primary = 2
    secondary = 3
    tertiary = 4
    residential = 5
    # service
    service = 6
    parking_aisle = 7
    driveway = 8
    #
    pedestrian = 9
    track = 10
    footway = 11
    steps = 12
    cycleway = 13
    other = 14
    
    def __init__(self, element):
        self.element = element
        self.polyline = None
        self.category = RealWay.osmClassify(self)
    
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
    def osmClassify(way):
        tags = way.element.tags


_osmHighwayToCategory