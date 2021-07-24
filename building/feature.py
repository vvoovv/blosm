"""
A module to define features for a instance of the class <BldgPolygon>
"""

from defs.building import BldgPolygonFeature


class Feature:
    
    __slots__ = (
        "featureId", "active", "startVector", "endVector", "startEdge",
        "startNextVector", "parent", "child", "numVectors"
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
        edge = startVector.edge = manager.getEdge(nodeId1, nextVector.id1)
        startVector.direct = nodeId1 == edge.id1
    
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
    
    def __init__(self, startVector, endVector):
        self.numVectors = 2 if startVector is endVector else 0
        super().__init__(BldgPolygonFeature.straightAngle, startVector, endVector)

    def markVectors(self):
        if self.numVectors == 2:
            self.startVector.feature = self.endVector.feature = self
        else:
            super().markVectors()
    
    def skipVectors(self, manager):
        if self.numVectors == 2:
            self.endVector.skip = True
            self._skipVectors(manager)
        else:
            super().skipVectors(manager)


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
        super().__init__(BldgPolygonFeature.complex, startVector, endVector)


class ComplexConcave(Feature):
    
    def __init__(self, startVector, endVector):
        super().__init__(BldgPolygonFeature.complex, startVector, endVector)


class QuadConvex(Feature):
    
    def __init__(self, startVector, endVector):
        super().__init__(BldgPolygonFeature.quadrangle, startVector, endVector)


class QuadConcave(Feature):
    
    def __init__(self, startVector, endVector):
        super().__init__(BldgPolygonFeature.quadrangle, startVector, endVector)
        
        
class TriConvex(Feature):
    
    def __init__(self, startVector, endVector):
        super().__init__(BldgPolygonFeature.triangle, startVector, endVector)


class TriConcave(Feature):
    
    def __init__(self, startVector, endVector):
        super().__init__(BldgPolygonFeature.triangle, startVector, endVector)