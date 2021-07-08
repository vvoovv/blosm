"""
A module to define features for a instance of the class <BldgPolygon>
"""

from defs.building import BldgPolygonFeature


class Feature:
    
    __slots__ = (
        "featureId", "active", "startVector", "endVector", "startEdge",
        "startNextVector", "parent", "child"
    )
    
    def __init__(self, featureId, startVector, endVector, skip, manager):
        self.featureId = featureId
        self.active = True
        
        # <startVector> will be used as a proxy vector for the feature
        self.startVector = startVector
        self.endVector = endVector
        nextVector = endVector.next
        
        # the parent feature
        if startVector.feature:
            self.parent = startVector.feature
            startVector.feature.child = self
        else:
            self.parent = None
        self.child = None
        
        currentVector = startVector
        while True:
            currentVector.feature = self
            if skip:
                currentVector.skip = True
            currentVector = currentVector.next
            if currentVector is nextVector:
                break
        if skip:
            # instance of <BldgEdge> replaced for <startVector>
            self.startEdge = startVector.edge
            self.startNextVector = startVector.next
            # <startVector> is also used as a proxy vector for the feature
            startVector.skip = False
            # get the new edge for <startVector> that is also used as a proxy vector for the feature
            nodeId1 = startVector.id1
            edge = startVector.edge = manager.getEdge(nodeId1, nextVector.id1)
            startVector.direct = nodeId1 == edge.id1
        
            nextVector.prev = startVector
            startVector.next = nextVector
        
        # The condition below actually checks if we have the footprint
        # for the whole building or a building part
        if startVector.polygon.building:
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