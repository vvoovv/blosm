import numpy


class RealWay:
    
    def __init__(self, element):
        self.element = element
        self.polyline = None
    
    def segments(self, data):
        # segmentCenter, segmentUnitVector, segmentLength
        coord0 = coord1 = None
        for coord2 in ( numpy.array((coord[0], coord[1])) for coord in self.element.getData(data) ):
            if coord1 is None:
                coord0 = coord1
            else:
                segmentVector = coord2 - coord1
                segmentLength = numpy.linalg.norm(segmentVector)
                yield (coord1 + coord2)/2., segmentVector/segmentLength, segmentLength
            coord1 = coord2
        if self.element.isClosed():
            segmentVector = coord0 - coord1
            segmentLength = numpy.linalg.norm(segmentVector)
            yield (coord1 + coord0)/2., segmentVector/segmentLength, segmentLength