"""
A module to define features for a instance of the class <BldgPolygon>
"""

from defs.building import BldgPolygonFeature


class Feature:
    
    __slots__ = ("active", "proxyVector", "endVector", "startEdge", "startNextVector", "parent", "child")
    
    def __init__(self, featureId, startVector, endVector, manager):
        self.featureId = featureId
        self.active = True
        # <startVector> will be used as a proxy vector for the feature
        self.proxyVector = startVector
        self.endVector = endVector
        # instance of <BldgEdge> replaced for <startVector>
        self.startEdge = startVector.edge
        self.startNextVector = startVector.next
        
        # get the new edge for <self.proxyVector> (aka <startVector>)
        nodeId1 = startVector.id1
        edge = startVector.edge = manager.getEdge(nodeId1, endVector.id2)
        # The condition below actually checks if we have the footprint
        # for the whole building or a building part
        if startVector.polygon.building:
            edge.addVector(startVector)
        startVector.direct = nodeId1 == edge.id1
        
        endVector.next.prev = startVector
        startVector.next = endVector.next
        
        # the parent feature
        if startVector.feature:
            self.parent = startVector.feature
            startVector.feature.child = self
        else:
            self.parent = None
        startVector.feature = self
        self.child = None

    def restoreVectors(self):
        """
        Restore the vectors that form the feature
        """
        proxyVector = self.proxyVector
        proxyVector.next.prev = self.endVector
        proxyVector.next = self.startNextVector
        proxyVector.edge = self.startEdge
        # deactivate the feature
        self.


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