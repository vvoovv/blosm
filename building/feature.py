"""
A module to define features for a instance of the class <BldgPolygon>
"""

from mathutils import Vector
from building import BldgEdge
from defs.building import BldgPolygonFeature, StraightAngleType


class Feature:
    
    __slots__ = (
        "featureId", "active", "startVector", "endVector", "startEdge",
        "startNextVector", "parent", "child", "numVectors",
        "startSin", "nextSin"
    )
    
    def __init__(self, featureId, startVector, endVector):
        self.featureId = featureId
        self.active = True
        
        # <startVector> will be used as a proxy vector for the feature
        self.startVector = startVector
        self.endVector = endVector
        
        # the parent feature
        if startVector.feature:
            self.parent = startVector.feature
            startVector.feature.child = self
        else:
            self.parent = None
        self.child = None
        
        self.startSin = self.nextSin = None
        
        self.markVectors()
        
    def markVectors(self):
        currentVector = self.startVector
        while True:
            currentVector.feature = self
            currentVector = currentVector.next
            if currentVector is self.endVector.next:
                break
    
    def skipVectors(self, manager):        
        currentVector = self.startVector.next
        
        while not currentVector is self.endVector.next:
            currentVector.skip = True
            currentVector = currentVector.next
        
        self._skipVectors(manager)
    
    def _skipVectors(self, manager):
        startVector = self.startVector
        nextVector = self.endVector.next
        
        # instance of <BldgEdge> replaced for <startVector>
        self.startEdge = startVector.edge
        self.startNextVector = startVector.next
        # get the new edge for <startVector> that is also used as a proxy vector for the feature
        nodeId1 = startVector.id1
        nodeId2 = nextVector.id1
        if nodeId1 and nodeId2:
            edge = startVector.edge = manager.getEdge(nodeId1, nodeId2)
            startVector.direct = nodeId1 == edge.id1
        else:
            edge = startVector.edge = BldgEdge(nodeId1, startVector.v1, nodeId2, nextVector.v1)
            startVector.direct = True
    
        nextVector.prev = startVector
        startVector.next = nextVector
        
        # The condition below actually checks if we have the footprint
        # for the whole building or a building part
        if startVector.polygon.building:
            # we have just created a new edge, so we have to add the related vector to the edge
            startVector.edge.addVector(startVector)
    
    def restoreVectors(self):
        """
        Restore the vectors that form the feature
        """
        proxyVector = self.proxyVector
        proxyVector.next.prev = self.endVector
        proxyVector.next = self.startNextVector
        proxyVector.edge = self.startEdge
        # deactivate the feature
        self.active = False


class StraightAngle(Feature):
    
    def __init__(self, startVector, endVector, _type):
        self.type = _type
        self.twoVectors = startVector is endVector
        super().__init__(BldgPolygonFeature.straightAngle, startVector, endVector)

    def markVectors(self):
        if self.twoVectors:
            self.startVector.feature = self.endVector.feature = self
        else:
            super().markVectors()
    
    def skipVectors(self, manager):
        if self.twoVectors:
            self.endVector.skip = True
            self._skipVectors(manager)
        else:
            super().skipVectors(manager)
    
    def extendToLeft(self):
        if self.twoVectors:
            self.twoVectors = False
        
        endVector = self.endVector = self.endVector.next
        endVector.feature = self
        endVector.skip = True
        
        # soft skip
        nextVector = endVector.next
        nextVector.prev = self.startVector
        self.startVector.next = nextVector

class NoSharedBldg:
    """
    A special feature that represents a straight angle
    Both edges attached to a node in question do not have a shared building
    """
    def __init__(self):
        self.category = BldgPolygonFeature.NoSharedBldg


class SharedBldgBothEdges:
    """
    A special feature that represents a straight angle
    Both edges attached to a node in question do have a shared building
    """
    def __init__(self):
        self.category = BldgPolygonFeature.SharedBldgBothEdges
        

class Curved(Feature):
    
    def __init__(self, startVector, endVector):
        super().__init__(BldgPolygonFeature.curved, startVector, endVector)
        

class ComplexConvex(Feature):
    
    def __init__(self, startVector, endVector):
        super().__init__(BldgPolygonFeature.complex_convex, startVector, endVector)

    def skipVectors(self, manager):
        # don't skip it for now
        pass


class ComplexConcave(Feature):
    
    def __init__(self, startVector, endVector):
        super().__init__(BldgPolygonFeature.complex_concave, startVector, endVector)

    def skipVectors(self, manager):
        # don't skip it for now
        pass


class QuadConvex(Feature):
    
    __slots__ = ("middleVector", "endEdge")
        
    def __init__(self, startVector, endVector):
        self.middleVector = startVector.next
        super().__init__(BldgPolygonFeature.quadrangle_convex, startVector, endVector)
    
    def markVectors(self):
        self.startVector.feature = self.middleVector.feature = self.endVector.feature = self
    
    def skipVectors(self, manager):
        # calculate the distance from <self.startVector.v1> and <self.endVector.v2> to <self.middleVector>
        startVector = self.startVector
        endVector = self.endVector
        unitMiddleVector = self.middleVector.unitVector
        _startVector = startVector.vector
        _endVector = endVector.vector
        normalToMiddle = Vector((unitMiddleVector[1], -unitMiddleVector[0]))
        startDistance = abs(_startVector.dot(normalToMiddle))
        endDistance = abs(_endVector.dot(normalToMiddle))
        
        # the middle vector is skipped in any case
        self.middleVector.skip = True
        
        if abs(endDistance - startDistance)/startDistance < 0.09:
            nextVector = endVector.next
            self.startSin = startVector.sin
            self.nextSin = nextVector.sin
            
            endVector.skip = True
            self._skipVectors(manager)
            
            startVector.calculateSin()
            nextVector.calculateSin()
        else:
            # Is the quandrangle located on the left corner of the polygon edge?
            leftCorner = startDistance < endDistance
            if leftCorner:
                self.startSin = startVector.sin
                
                newVert = endVector.v1 - _endVector *\
                    _startVector.cross(unitMiddleVector)/_endVector.cross(unitMiddleVector)
                
                endVector.feature = None
            else: # endDistance < startDistance # the right corner of the polygon edge
                nextVector = endVector.next
                self.nextSin = nextVector.sin
                
                newVert = startVector.v2 + _startVector *\
                    _endVector.cross(unitMiddleVector)/_startVector.cross(unitMiddleVector)
                
                startVector.feature = None
                endVector.sin = self.middleVector.sin
            # instance of <BldgEdge> replaced for <startVector>
            self.startEdge = startVector.edge
            # replace the edge for <startVector>
            startVector.edge = BldgEdge(startVector.id1, startVector.v1, '', newVert)
            startVector.direct = True
            
            # instance of <BldgEdge> replaced for <endVector>
            self.endEdge = endVector.edge
            # replace the edge for <endVector>
            endVector.edge = BldgEdge('', newVert, endVector.id2, endVector.v2)
            endVector.direct = True
            
            startVector.next = endVector
            endVector.prev = startVector
            
            if leftCorner:
                startVector.calculateSin()
            else:
                nextVector.calculateSin()


class QuadConcave(Feature):
    
    def __init__(self, startVector, endVector):
        super().__init__(BldgPolygonFeature.quadrangle_concave, startVector, endVector)
    
    def skipVectors(self, manager):
        # don't skip it for now
        pass
        
        
class TriConvex(Feature):
    
    def __init__(self, startVector, endVector):
        super().__init__(BldgPolygonFeature.triangle_convex, startVector, endVector)


class TriConcave(Feature):
    
    def __init__(self, startVector, endVector):
        super().__init__(BldgPolygonFeature.triangle_concave, startVector, endVector)

    def skipVectors(self, manager):
        # don't skip it for now
        pass