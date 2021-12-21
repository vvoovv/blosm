"""
A module to define features for a instance of the class <BldgPolygon>
"""

from mathutils import Vector
from building import BldgEdge, BldgPolygon
from defs.building import BldgPolygonFeature


class Feature:
    
    __slots__ = (
        "type", "skipped", "startVector", "endVector", "startEdge",
        "startNextVector", "parent", "numVectors",
        "startSin", "endSin", "nextSin",
        "prev"
    )
    
    def __init__(self, _type, startVector, endVector):
        self.type = _type
        self.skipped = False
        
        # <startVector> will be used as a proxy vector for the feature
        self.startVector = startVector
        self.endVector = endVector
        
        self.startSin = self.endSin = self.nextSin = None
        
        self.setParentFeature()
        self.markVectors()
    
    def setParentFeature(self):
        self.parent = self.startVector.feature
    
    def markVectors(self):
        self.startVector.feature = self
    
    def markVectorsAll(self):
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
            currentVector.polygon.numEdges -= 1
            currentVector = currentVector.next
        
        self._skipVectors(manager)
    
    def _skipVectors(self, manager):
        startVector = self.startVector
        nextVector = self.endVector.next
        
        # instance of <BldgEdge> replaced for <startVector>
        self.startEdge = (startVector.edge, startVector.direct)
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
        
        self.skipped = True
        
        # The condition below actually checks if we have the footprint
        # for the whole building or a building part
        if startVector.polygon.building:
            # we have just created a new edge, so we have to add the related vector to the edge
            startVector.edge.addVector(startVector)
    
    def _skipVectorsCached(self):
        startVector = self.startVector
        nextVector = self.endVector.next
        
        #self.startNextVector = startVector.next
        
        startVector.edge, startVector.direct = self.startEdge
        
        nextVector.prev = startVector
        startVector.next = nextVector
        
        self.skipped = True
    
    def unskipVectors(self):
        self._unskipVectors()
        
        currentVector = self.startVector
        
        while True:
            currentVector.skip = False
            if currentVector is self.endVector:
                break
            currentVector = currentVector.next
    
    def _unskipVectors(self):
        """
        Restore the vectors that form the feature
        """
        startVector = self.startVector
        startVector.next.prev = self.endVector
        startVector.next = self.startNextVector
        # remember <startVector.edge> and <startVector.direct> for the future use
        (startVector.edge, startVector.direct), self.startEdge = self.startEdge, (startVector.edge, startVector.direct)
        self.skipped = False
    
    def getProxyVector(self):
        """
        Get a proxy vector for the skipped feature
        """
        return self.startVector
    
    def _calculateSinsSkipped(self):
        self.startSin = self.startVector.sin
        self.startVector.calculateSin()
        self._calculateSinNextVector()
    
    def _calculateSinNextVector(self):
        nextVector = self.endVector.next
        feature = nextVector.feature
        if not (feature and feature.type != BldgPolygonFeature.straightAngle and feature.skipped):
            self.nextSin = nextVector.sin
            nextVector.calculateSin()
    
    def _setSinNextVectorCached(self):
        nextVector = self.endVector.next
        feature = nextVector.feature
        if not (feature and feature.type != BldgPolygonFeature.straightAngle and feature.skipped):
            nextVector.sin, self.nextSin = self.nextSin, nextVector.sin
    
    def _skipVectorsKeepStartVectorCached(self):
        self._skipVectorsCached()
        
        self.startVector.sin = self.startSin
        self._setSinNextVectorCached()
    
    def restoreSins(self):
        # remember <self.startVector.sin> and <self.endVector.next.sin> for the future use
        self.startVector.sin, self.startSin = self.startSin, self.startVector.sin
        self._restoreSinNextVector()
    
    def _restoreSinNextVector(self):
        nextVector = self.endVector.next
        feature = nextVector.feature
        if not (feature and feature.type != BldgPolygonFeature.straightAngle and feature.skipped):
            self.endVector.next.sin, self.nextSin = self.nextSin, self.endVector.next.sin
    
    def _checkTriangularFeature(self):
        """
        Check if the triangular feature is located in a corner before this feature and
        <endVector> of the triangular feature forms a straight angle with the neighbor edge of this feature
        """
        startVector = self.startVector
        prevVector = startVector.prev
        if prevVector.featureType == BldgPolygonFeature.triangle_convex:
            triangularFeature = prevVector.feature
            triangularEndVector = triangularFeature.endVector
            startVector.prev = triangularEndVector
            startVector.calculateSin()
            if startVector.hasStraightAngle:
                triangularFeature.unskipVectors()
                triangularFeature.invalidate()
            else:
                startVector.prev = prevVector


class StraightAnglePart(Feature):
    """
    A part of <StraightAngle> feature formed by at least 2 edges not shared with another building OR
    at least 2 edges shared with a single another building. It can't share a curved feature in the former case.
    """
    
    def __init__(self, startVector, endVector, _type):
        self.twoVectors = startVector.next is endVector
        super().__init__(_type, startVector, endVector)

    def markVectorsAll(self):
        if self.twoVectors:
            self.startVector.feature = self.endVector.feature = self
        else:
            super().markVectorsAll()
    
    def setStartVector(self, startVector):
        self.startVector = startVector
        if self.twoVectors:
            self.twoVectors = False
        self.setParentFeature()
        self.markVectors()
    
    def skipVectors(self, manager):
        if self.twoVectors:
            self.endVector.skip = True
            self.endVector.polygon.numEdges -= 1
            self._skipVectors(manager)
        else:
            super().skipVectors(manager)
    
    def unskipVectors(self):
        if self.twoVectors:
            self.endVector.skip = False
            self.endVector.polygon.numEdges += 1
            self._unskipVectors()
        else:
            super().unskipVectors()
        
        self.startVector.feature = self.parent


class StraightAngle(StraightAnglePart):
    
    __slots__ = ("hasSharedEdge", "hasFreeEdge", "sharesOnePolygon", "sharesCurve")
    
    def __init__(self, startVector, endVector, _type):
        super().__init__(startVector, endVector, _type)
        
        polygon = startVector.polygon
        self.prev = polygon.saFeature
        polygon.saFeature = self
        
        self.hasSharedEdge = self.hasFreeEdge = self.sharesCurve = False
        self.sharesOnePolygon = True
    
    def isCurved(self):
        """
        Is it actually a curved feature?
        """
        # Calcualte the sine of the angle between <self.startVector> and the vector
        # from <self.startVector.v1> and <self.endVector.v2>
        # sin = vector.cross(self.startVector.unitVector)
        return not self.twoVectors and \
            (self.endVector.v2 - self.startVector.v1).normalized().cross(self.startVector.unitVector) \
                > BldgPolygon.straightAngleSin
    
    def getNextVector(self, vector):
        return self.startNextVector if vector is self.startVector else vector.next


class StraightAngleSfs(StraightAnglePart):
    # <sfs> stands for "small feature skipped"
    
    def __init__(self, startVector, endVector):
        super().__init__(startVector, endVector, BldgPolygonFeature.straightAngleSfs)
        
        polygon = startVector.polygon
        self.prev = polygon.saSfsFeature
        polygon.saSfsFeature = self
    
    def inheritFacadeClass(self):
        """
        Inherit the facade class from <self.startVector.edge>
        """
        startVector = self.startVector
        
        self.startEdge[0].cl = startVector.edge.cl
        
        if self.twoVectors:
            self.endVector.edge.cl = startVector.edge.cl
        else:
            currentVector = self.startNextVector
            while True:
                currentVector.edge.cl = startVector.edge.cl
                if currentVector is self.endVector:
                    break
                currentVector = currentVector.next


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
        polygon = startVector.polygon
        if not polygon.curvedFeature:
            polygon.curvedFeature = self
    
    def markVectors(self):
        # We need to mark all vectors belonging to the curved feature
        # to avoid side effect when detecting small features
        super().markVectorsAll()
        

class ComplexConvex5(Feature):
    """
    A class for complex convex features with exactly 5 edged
    """
    
    __slots__ = ("endPrevVector", "middleEdge", "endEdge")
    
    def __init__(self, startVector, endVector):
        super().__init__(BldgPolygonFeature.complex5_convex, startVector, endVector)
        # <self.endEdge> also serves as an indicator how the feature vectors were skipped.
        # If <self.endEdge> is equal to None, then only <self.startVector> was kept.
        # Otherwise <self.startVector>, the middle vector and <self.endVector> were kept.
        self.endEdge = None
        
        polygon = startVector.polygon
        self.prev = polygon.smallFeature
        polygon.smallFeature = self
    
    def skipVectors(self, manager):
        startVector = self.startVector
        endVector = self.endVector
        
        _startVector = startVector.vector
        _endVector = endVector.vector
        
        keepOnlyStartVector = False
        
        if _startVector.dot(_endVector) <= 0:
            keepOnlyStartVector = True
        else:
            middleVector = endVector.prev.prev
            _middleVector = middleVector.vector
            # Check if have a quandrangular feature formed by the vectors
            # <startVector>, <startVector.next>, <middleVector>.
            unitRefVector1 = startVector.next.unitVector
            normalToRefVector = Vector((unitRefVector1[1], -unitRefVector1[0]))
            distance1 = abs(_startVector.dot(normalToRefVector))
            distance2 = abs(_middleVector.dot(normalToRefVector))
            
            # The distance <distance2> from the end vertex of <middleVector> to <startVector.next>
            # must be larger than the distance <distance1> from start vertex of <startVector> to <startVector.next>
            if distance2 > distance1:
                # Check if have a quandrangular feature formed by the vectors
                # <middleVector>, <endVector.prev>, <endVector>.
                unitRefVector2 = endVector.prev.unitVector
                normalToRefVector = Vector((unitRefVector2[1], -unitRefVector2[0]))
                distance1 = abs(_middleVector.dot(normalToRefVector))
                distance2 = abs(_endVector.dot(normalToRefVector))
                # The distance <distance2> from the end vertex of <middleVector> to <startVector.next>
                # must be larger than the distance <distance1> from start vertex of <startVector> to <startVector.next>
                if distance1 > distance2:
                    newVert1 = middleVector.v1 - _middleVector * _startVector.cross(unitRefVector1)/_middleVector.cross(unitRefVector1)
                    newVert2 = middleVector.v2 + _middleVector * _endVector.cross(unitRefVector2)/_middleVector.cross(unitRefVector2)
                    
                    # modify <startVector>
                    # instance of <BldgEdge> replaced for <startVector>
                    self.startEdge = (startVector.edge, startVector.direct)
                    # replace the edge for <startVector>
                    startVector.edge = BldgEdge(startVector.id1, startVector.v1, '', newVert1)
                    startVector.direct = True
                    
                    self.startNextVector = startVector.next
                    startVector.next.skip = True
                    
                    # modify <middleVector>
                    # instance of <BldgEdge> replaced for <middleVector>
                    self.middleEdge = (middleVector.edge, middleVector.direct)
                    # replace the edge for <middleVector>
                    middleVector.edge = BldgEdge('', newVert1, '', newVert2)
                    middleVector.direct = True
                    
                    self.endPrevVector = endVector.prev
                    endVector.prev.skip = True
                    
                    # modify <endVector>
                    # instance of <BldgEdge> replaced for <endVector>
                    self.endEdge = (endVector.edge, endVector.direct)
                    # replace the edge for <endVector>
                    endVector.edge = BldgEdge('', newVert2, endVector.id2, endVector.v2)
                    endVector.direct = True
                    
                    startVector.next = endVector.prev = middleVector
                    middleVector.prev = startVector
                    middleVector.next = endVector
                    
                    startVector.polygon.numEdges -= 2
                    # the following line is needed to remove straight angles
                    endVector.feature = self
                    
                    self.skipped = True
                    
                else:
                    keepOnlyStartVector = True
            else:
                keepOnlyStartVector = True
            
        if keepOnlyStartVector:
            startVector.next.skip = endVector.prev.prev.skip = \
                endVector.prev.skip = endVector.skip = True
            startVector.polygon.numEdges -= 4
            
            self._skipVectors(manager)
    
    def calculateSinsSkipped(self):
        self._checkTriangularFeature()
        
        if self.endEdge:
            self.startSin = self.startVector.sin
            self.endSin = self.endVector.sin
            self.startVector.calculateSin()
            self.endVector.calculateSin()
            self._calculateSinNextVector()
        else:
            self._calculateSinsSkipped()
    
    def skipVectorsCached(self):
        startVector = self.startVector
        endVector = self.endVector
        
        if self.endEdge:
            middleVector = endVector.prev.prev
            
            # modify <startVector>
            startVector.edge, startVector.direct = self.startEdge
            
            #self.startNextVector = startVector.next
            startVector.next.skip = True
            
            # modify <middleVector>
            middleVector.edge, middleVector.direct = self.middleEdge
            
            #self.endPrevVector = endVector.prev
            endVector.prev.skip = True
            
            # modify <endVector>
            endVector.edge, endVector.direct = self.endEdge
            
            startVector.next = endVector.prev = middleVector
            middleVector.prev = startVector
            middleVector.next = endVector
            
            startVector.sin = self.startSin
            endVector.sin = self.endSin
            self._setSinNextVectorCached()
            
            startVector.polygon.numEdges -= 2
            # the following line is needed to remove straight angles
            endVector.feature = self
            
            self.skipped = True
        else:
            startVector.next.skip = endVector.prev.prev.skip = \
                endVector.prev.skip = endVector.skip = True
            startVector.polygon.numEdges -= 4
            
            self._skipVectorsKeepStartVectorCached()
    
    def unskipVectors(self):
        startVector = self.startVector
        endVector = self.endVector
        
        if self.endEdge:
            middleVector = startVector.next
            
            # remember <startVector.edge> and <startVector.direct> for the future use
            (startVector.edge, startVector.direct), self.startEdge = self.startEdge, (startVector.edge, startVector.direct)
            startVector.next = self.startNextVector
            
            self.startNextVector.skip = False
            
            middleVector.prev = self.startNextVector
            # remember <middleVector.edge> and <middleVector.direct> for the future use
            (middleVector.edge, middleVector.direct), self.middleEdge = self.middleEdge, (middleVector.edge, middleVector.direct)
            middleVector.next = self.endPrevVector
            
            self.endPrevVector.skip = False
            
            endVector.prev = self.endPrevVector
            # remember <endVector.edge> and <endVector.direct> for the future use
            (endVector.edge, endVector.direct), self.endEdge = self.endEdge, (endVector.edge, endVector.direct)
            
            startVector.polygon.numEdges += 2
        else:
            self._unskipVectors()
            
            startVector.next.skip = endVector.prev.prev.skip = \
                endVector.prev.skip = endVector.skip = False
            startVector.polygon.numEdges += 4
    
    def restoreSins(self):
        if self.endEdge:
            startVector = self.startVector
            endVector = self.endVector
            startVector.sin, self.startSin = self.startSin, startVector.sin
            endVector.sin, self.endSin = self.endSin, endVector.sin
            self._restoreSinNextVector()
        else:
            super().restoreSins()
    
    def isSkippable(self):
        """
        See the details in QuadConvex.isSkippable()
        """
        return not self.startVector.edge.hasSharedBldgVectors() and \
            not self.startVector.next.edge.hasSharedBldgVectors() and \
            not self.endVector.prev.prev.edge.hasSharedBldgVectors() and \
            not self.endVector.prev.edge.hasSharedBldgVectors() and \
            not self.endVector.edge.hasSharedBldgVectors()
    
    def inheritFacadeClass(self):
        """
        Inherit the facade class from <self.startVector.edge>
        """
        self.startEdge[0].cl = \
        self.startNextVector.edge.cl = \
        self.startNextVector.next.edge.cl = \
        self.endVector.prev.edge.cl = \
        self.endVector.edge.cl = \
        self.startVector.edge.cl


class ComplexConvex4(Feature):
    """
    A class for complex convex features with exactly 4 edged
    """
    
    __slots__ = ("endPrevVector", "endEdge")
    
    def __init__(self, startVector, endVector):
        super().__init__(BldgPolygonFeature.complex4_convex, startVector, endVector)
        
        # <self.endEdge> also serves as an indicator how the feature vectors were skipped.
        # If <self.endEdge> is equal to None, then only <self.startVector> was kept.
        # Otherwise <self.startVector>, the middle vector and <self.endVector> were kept.
        self.endEdge = None
        
        polygon = startVector.polygon
        self.prev = polygon.complex4Feature
        polygon.complex4Feature = self
    
    def skipVectors(self, manager):
        startVector = self.startVector
        endVector = self.endVector
        # the neighbor to the left
        left = startVector.prev
        # the neighbor to the right
        right = endVector.next
        
        keepOnlyStartVector = False
        
        # Check if <left> isn't a part of a curved feature and
        # <right> isn't a part of a curved feature or another complex feature with 4 edges
        if left.featureType == BldgPolygonFeature.curved or\
            right.featureType in (BldgPolygonFeature.curved, BldgPolygonFeature.complex4_convex):
            keepOnlyStartVector = True
        else:
            unitL = left.unitVector
            unitR = -right.unitVector
            cross = unitL.cross(unitR)
            if abs(cross) < 0.2:
                # <unitL> and <unitR> are nearly parallel
                keepOnlyStartVector = True
            else:
                vec = endVector.v2 - startVector.v1
                k1 = vec.cross(unitR)/cross
                k2 = vec.cross(unitL)/cross
                if k1>0. and k2>0.:
                    newVert = startVector.v1 + k1*unitL
                    
                    # modify <startVector>
                    # instance of <BldgEdge> replaced for <startVector>
                    self.startEdge = (startVector.edge, startVector.direct)
                    # replace the edge for <startVector>
                    startVector.edge = BldgEdge(startVector.id1, startVector.v1, '', newVert)
                    startVector.direct = True
                    
                    self.startNextVector = startVector.next
                    startVector.next.skip = True
                    startVector.next = endVector
                    
                    self.endPrevVector = endVector.prev
                    endVector.prev.skip = True
                    endVector.prev = startVector
                    
                    # modify <endVector>
                    # instance of <BldgEdge> replaced for <endVector>
                    self.endEdge = (endVector.edge, endVector.direct)
                    # replace the edge for <endVector>
                    endVector.edge = BldgEdge('', newVert, endVector.id2, endVector.v2)
                    endVector.direct = True
                    
                    startVector.polygon.numEdges -= 2
                    # the following line is needed to remove straight angles
                    endVector.feature = self
                    
                    self.skipped = True
                else:
                    keepOnlyStartVector = True
        
        if keepOnlyStartVector:
            startVector.next.skip = endVector.prev.skip = endVector.skip = True
            startVector.polygon.numEdges -= 3
            
            self._skipVectors(manager)
    
    def calculateSinsSkipped(self):
        self._checkTriangularFeature()
        
        if self.endEdge:
            self.startSin = self.startVector.sin
            self.endSin = self.endVector.sin
            self.startVector.calculateSin()
            self.endVector.calculateSin()
            self._calculateSinNextVector()
        else:
            self._calculateSinsSkipped()
    
    def skipVectorsCached(self):
        startVector = self.startVector
        endVector = self.endVector
        
        if self.endEdge:
            # modify <startVector>
            startVector.edge, startVector.direct = self.startEdge
            
            #self.startNextVector = startVector.next
            startVector.next.skip = True
            startVector.next = endVector
            
            #self.endPrevVector = endVector.prev
            endVector.prev.skip = True
            endVector.prev = startVector
            
            # modify <endVector>
            endVector.edge, endVector.direct = self.endEdge
            
            startVector.sin = self.startSin
            endVector.sin = self.endSin
            self._setSinNextVectorCached()
            
            startVector.polygon.numEdges -= 2
            # the following line is needed to remove straight angles
            endVector.feature = self
        else:
            startVector.next.skip = endVector.prev.skip = endVector.skip = True
            startVector.polygon.numEdges -= 3
            
            self._skipVectorsKeepStartVectorCached()

    def unskipVectors(self):
        startVector = self.startVector
        endVector = self.endVector
        
        if self.endEdge:
            # remember <startVector.edge> and <startVector.direct> for the future use
            (startVector.edge, startVector.direct), self.startEdge = self.startEdge, (startVector.edge, startVector.direct)
            startVector.next = self.startNextVector
            
            startVector.next.skip = False
            
            self.endPrevVector.skip = False
            
            endVector.prev = self.endPrevVector
            # remember <endVector.edge> and <endVector.direct> for the future use
            (endVector.edge, endVector.direct), self.endEdge = self.endEdge, (endVector.edge, endVector.direct)
            
            startVector.polygon.numEdges += 2
        else:
            self._unskipVectors()
            startVector.next.skip = endVector.prev.skip = endVector.skip = False
            startVector.polygon.numEdges += 3
    
    def restoreSins(self):
        if self.endEdge:
            startVector = self.startVector
            endVector = self.endVector
            
            startVector.sin, self.startSin = self.startSin, startVector.sin
            endVector.sin, self.endSin = self.endSin, endVector.sin
            self._restoreSinNextVector()
        else:
            super().restoreSins()
    
    def isSkippable(self):
        """
        See the details in QuadConvex.isSkippable()
        """
        return not self.startVector.edge.hasSharedBldgVectors() and \
            not self.startVector.next.edge.hasSharedBldgVectors() and \
            not self.endVector.prev.edge.hasSharedBldgVectors() and \
            not self.endVector.edge.hasSharedBldgVectors()
    
    def inheritFacadeClass(self):
        """
        Inherit the facade class from <self.startVector.edge>
        """
        self.startEdge[0].cl = \
        self.startNextVector.edge.cl = \
        self.endVector.prev.edge.cl = \
        self.endVector.edge.cl = \
        self.startVector.edge.cl


class QuadConvex(Feature):
    
    __slots__ = ("middleVector", "endEdge", "equalSideEdges", "leftEdgeShorter", "newVert")
        
    def __init__(self, startVector, endVector):
        self.init(BldgPolygonFeature.quadrangle_convex, startVector, endVector)
    
    def init(self, _type, startVector, endVector):
        self.middleVector = startVector.next
        
        # calculate distances from start and end vertices to the middle edge
        unitMiddleVector = self.middleVector.unitVector
        _startVector = startVector.vector
        _endVector = endVector.vector
        normalToMiddle = Vector((unitMiddleVector[1], -unitMiddleVector[0]))
        startDistance = abs(_startVector.dot(normalToMiddle))
        endDistance = abs(_endVector.dot(normalToMiddle))
        
        self.equalSideEdges = abs(endDistance - startDistance)/startDistance < 0.09
        if self.equalSideEdges:
            self.leftEdgeShorter = False
        else:
            self.leftEdgeShorter = startDistance > endDistance
            self.newVert = startVector.v2 + _startVector * _endVector.cross(unitMiddleVector)/_startVector.cross(unitMiddleVector) \
                if self.leftEdgeShorter else \
                endVector.v1 - _endVector * _startVector.cross(unitMiddleVector)/_endVector.cross(unitMiddleVector)
        
        super().__init__(_type, startVector, endVector)
        
        polygon = startVector.polygon
        self.prev = polygon.smallFeature
        polygon.smallFeature = self
    
    def setParentFeature(self):
        self.parent = (self.startVector.feature, self.endVector.feature)
    
    def markVectors(self):
        # <self.middleVector> will be skipped anyway. There is now need to mark it
        self.startVector.feature = self.endVector.feature = self
    
    def skipVectors(self, manager):
        startVector = self.startVector
        endVector = self.endVector
        
        # the middle vector is skipped in any case
        self.middleVector.skip = True
        
        if self.equalSideEdges:
            endVector.skip = True
            startVector.polygon.numEdges -= 2
            
            self._skipVectors(manager)
        else:
            if self.leftEdgeShorter: # endDistance < startDistance
                startVector.feature = None
            else:
                endVector.feature = None
            # instance of <BldgEdge> replaced for <startVector>
            self.startEdge = (startVector.edge, startVector.direct)
            # replace the edge for <startVector>
            startVector.edge = BldgEdge(startVector.id1, startVector.v1, '', self.newVert)
            startVector.direct = True
            
            # instance of <BldgEdge> replaced for <endVector>
            self.endEdge = (endVector.edge, endVector.direct)
            # replace the edge for <endVector>
            endVector.edge = BldgEdge('', self.newVert, endVector.id2, endVector.v2)
            endVector.direct = True
            
            startVector.next = endVector
            endVector.prev = startVector
            
            startVector.polygon.numEdges -= 1
            
            self.skipped = True
    
    def calculateSinsSkipped(self):
        self._checkTriangularFeature()
        
        if self.equalSideEdges:
            self._calculateSinsSkipped()
        elif self.leftEdgeShorter:
            self.endSin = self.endVector.sin
            self.endVector.sin = self.middleVector.sin
            self._calculateSinNextVector()
        else:
            self.startSin = self.startVector.sin
            self.startVector.calculateSin()

    def skipVectorsCached(self):
        startVector = self.startVector
        endVector = self.endVector
        
        # the middle vector is skipped in any case
        self.middleVector.skip = True
        
        if self.equalSideEdges:
            endVector.skip = True
            startVector.polygon.numEdges -= 2
            
            self._skipVectorsKeepStartVectorCached()
        else:
            if self.leftEdgeShorter: # endDistance < startDistance
                startVector.feature = None
                endVector.sin = self.middleVector.sin
                self._setSinNextVectorCached()
            else:
                endVector.feature = None
                
                startVector.sin = self.startSin
            
            startVector.edge, startVector.direct = self.startEdge
            endVector.edge, endVector.direct = self.endEdge
            
            startVector.next = endVector
            endVector.prev = startVector
            
            startVector.polygon.numEdges -= 1
            
            self.skipped = True
    
    def unskipVectors(self):
        startVector = self.startVector
        endVector = self.endVector
        
        self.middleVector.skip = False
        if self.equalSideEdges:
            self._unskipVectors()
            
            endVector.skip = False
            startVector.polygon.numEdges += 2
        else:
            if self.leftEdgeShorter: # endDistance < startDistance
                startVector.feature = self
            else:
                endVector.feature = self
            # remember <startVector.edge> and <startVector.direct> for the future use
            (startVector.edge, startVector.direct), self.startEdge = self.startEdge, (startVector.edge, startVector.direct)
            
            # remember <endVector.edge> and <endVector.direct> for the future use
            (endVector.edge, endVector.direct), self.endEdge = self.endEdge, (endVector.edge, endVector.direct)
            
            startVector.next = self.middleVector
            endVector.prev = self.middleVector
            
            self.skipped = False
    
    def restoreSins(self):
        if self.equalSideEdges:
            super().restoreSins()
        elif self.leftEdgeShorter:
            endVector = self.endVector
            # remember <endVector.sin> for the future use
            endVector.sin, self.endSin = self.endSin, endVector.sin
            self._restoreSinNextVector()
        else:
            startVector = self.startVector
            # remember <startVector.sin> for the future use
            startVector.sin, self.startSin = self.startSin, startVector.sin
    
    def getProxyVector(self):
        return self.endVector if self.leftEdgeShorter else self.startVector
    
    def isSkippable(self):
        """
        Check if the feature can be skipped:
            * It doesn't have edges shared with the other polygons
        """
        return not self.startVector.edge.hasSharedBldgVectors() and \
            not self.startVector.next.edge.hasSharedBldgVectors() and \
            not self.endVector.edge.hasSharedBldgVectors()
    
    def inheritFacadeClass(self):
        """
        Inherit the facade class from <self.startVector.edge>
        """
        if self.equalSideEdges:
            self.startEdge[0].cl = \
            self.middleVector.edge.cl = \
            self.endVector.edge.cl = \
            self.startVector.edge.cl
        elif self.leftEdgeShorter:
            self.startEdge[0].cl = self.startVector.edge.cl
            self.middleVector.edge.cl = \
            self.endEdge[0].cl = \
            self.endVector.edge.cl
        else:
            self.startEdge[0].cl = \
            self.middleVector.edge.cl = \
            self.startVector.edge.cl
            self.endEdge[0].cl = self.endVector.edge.cl


class QuadConcave(QuadConvex):
    
    def __init__(self, startVector, endVector):
        self.init(BldgPolygonFeature.quadrangle_concave, startVector, endVector)
        
        
class TriConvex(Feature):
    
    def __init__(self, startVector, endVector):
        super().__init__(BldgPolygonFeature.triangle_convex, startVector, endVector)
        
        polygon = startVector.polygon
        self.prev = polygon.triangleFeature
        polygon.triangleFeature = self
    
    def skipVectors(self, manager):
        self.endVector.skip = True
        self.startVector.polygon.numEdges -= 1
        self._skipVectors(manager)
    
    def skipVectorsCached(self):
        if self.startVector.feature:
            self.endVector.skip = True
            self.startVector.polygon.numEdges -= 1
            self._skipVectorsKeepStartVectorCached()
    
    def unskipVectors(self):
        self._unskipVectors()
        self.endVector.skip = False
        self.startVector.polygon.numEdges += 1
    
    def invalidate(self):
        self.startVector.feature = None
    
    def isSkippable(self):
        """
        Check if the feature can be skipped:
            * It doesn't have edges shared with the other polygons
        """
        return not self.startVector.edge.hasSharedBldgVectors() and \
            not self.endVector.edge.hasSharedBldgVectors()
    
    def inheritFacadeClass(self):
        """
        Inherit the facade class from <self.startVector.edge>
        """
        self.startEdge[0].cl = self.endVector.edge.cl = self.startVector.edge.cl
    
    def calculateSinsSkipped(self):
        # Check if the triangular feature is located in a corner and
        # <self.startVector> forms a straight angle with the neighbor edge
        startVector = self.startVector
        prevVector = startVector.prev
        if prevVector.feature and not prevVector.featureType in (BldgPolygonFeature.triangle_convex, BldgPolygonFeature.curved):
            # temporarily restore the original attributes <edge> and <direct> for <startVector>
            (startVector.edge, startVector.direct), self.startEdge = self.startEdge, (startVector.edge, startVector.direct)
            startVector.calculateSin()
            # set back the values of the attributes <edge> and <direct> for <startVector>
            (startVector.edge, startVector.direct), self.startEdge = self.startEdge, (startVector.edge, startVector.direct)
            if startVector.hasStraightAngle:
                self.unskipVectors()
                self.invalidate()
                return
        
        self._calculateSinsSkipped()


class TriConcave(Feature):
    
    def __init__(self, startVector, endVector):
        super().__init__(BldgPolygonFeature.triangle_concave, startVector, endVector)

    def skipVectors(self, manager):
        # don't skip it for now
        pass
    
    def unskipVectors(self):
        # do nothing for now
        pass