"""
A module to define features for a instance of the class <BldgPolygon>
"""

from mathutils import Vector
from building import BldgEdge
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

    def skipVectors(self, manager):
        # don't skip it for now
        pass


class ComplexConcave(Feature):
    
    def __init__(self, startVector, endVector):
        super().__init__(BldgPolygonFeature.complex, startVector, endVector)

    def skipVectors(self, manager):
        # don't skip it for now
        pass


class QuadConvex(Feature):
    
    __slots__ = ("middleVector", "endEdge")
        
    def __init__(self, startVector, endVector):
        self.middleVector = startVector.next
        super().__init__(BldgPolygonFeature.quadrangle, startVector, endVector)
    
    def markVectors(self):
        self.startVector.feature = self.middleVector.feature = self.endVector.feature = self
    
    def skipVectors(self, manager):
        # calculate the distance from <self.startVector.v1> and <self.endVector.v2> to <self.middleVector>
        unitMiddleVector = self.middleVector.unitVector
        startVector = self.startVector.vector
        endVector = self.endVector.vector
        normalToMiddle = Vector((unitMiddleVector[1], -unitMiddleVector[0]))
        startDistance = abs(startVector.dot(normalToMiddle))
        endDistance = abs(endVector.dot(normalToMiddle))
        
        self.middleVector.skip = True
        if (endDistance - startDistance)/startDistance < 0.09:
            self.endVector.skip = True
            self._skipVectors(manager)
        elif startDistance < endDistance:
            newVert = self.endVector.v1 - endVector *\
                startVector.cross(unitMiddleVector)/endVector.cross(unitMiddleVector)
            
            startVector = self.startVector
            # instance of <BldgEdge> replaced for <startVector>
            self.startEdge = startVector.edge
            # replace the edge for <startVector>
            startVector.edge = BldgEdge(startVector.id1, startVector.v1, '', newVert)
            startVector.direct = True
            
            endVector = self.endVector
            # instance of <BldgEdge> replaced for <endVector>
            self.endEdge = endVector.edge
            # replace the edge for <endVector>
            endVector.edge = BldgEdge('', newVert, endVector.id2, endVector.v2)
            startVector.direct = True
            
            startVector.next = endVector
            endVector.prev = startVector
            endVector.feature = None
        else: # endDistance < startDistance
            newVert = startVector.v2 + startVector *\
                endVector.cross(unitMiddleVector)/startVector.cross(unitMiddleVector)
            
        


class QuadConcave(Feature):
    
    def __init__(self, startVector, endVector):
        super().__init__(BldgPolygonFeature.quadrangle, startVector, endVector)
    
    def skipVectors(self, manager):
        # don't skip it for now
        pass
        
        
class TriConvex(Feature):
    
    def __init__(self, startVector, endVector):
        super().__init__(BldgPolygonFeature.triangle, startVector, endVector)


class TriConcave(Feature):
    
    def __init__(self, startVector, endVector):
        super().__init__(BldgPolygonFeature.triangle, startVector, endVector)

    def skipVectors(self, manager):
        # don't skip it for now
        pass