from defs.building import BldgPolygonFeature
from building.feature import StraightAngleSfs


class SkipFeatures:
    
    def skipFeatures(self, polygon, manager):
        # * Curved features aren't skipped.
        # * Complex features with 4 edges are processed separately since "free" neighbor vectors
        # are needed to skip the vectors of a complex feature. For example, the neighbor vectors
        # may be a part of a quadrangle feature and we need to skip the quadrangle feature first
        # to get the "free" neighbor edges.
        # * The triangle convex features are processed separately since they may be located at a corner
        # and have quadrangle features as neighbors. After skipping the quadrangle features, a sequence of
        # straight angle can be formed. It means that those triangle convex features at a corner
        # should be invalidated.
        
        feature = polygon.smallFeature
        if feature:
            self.skipFeature(feature, manager)
            currentVector = startVector = feature.startVector
        
        # complex features with 4 edges are treated separately
        feature = polygon.complex4Feature
        if feature:
            self.skipFeature(feature, manager)
            currentVector = startVector = feature.startVector
        
        # triangular features are treated separetely
        feature = polygon.triangleFeature
        if feature:
            self.skipFeature(feature, manager)
            currentVector = startVector = feature.startVector
        
        # find <prevNonStraightVector>
        isPrevVectorStraight = False
        while True:
            feature = currentVector.feature
            if not _vectorHasStraightAngle(currentVector):
                prevNonStraightVector = currentVector
                break
            isPrevVectorStraight = True
            currentVector = currentVector.prev
        currentVector = startVector.next
        startVector = prevNonStraightVector
        while True:
            # conditions for a straight angle
            if _vectorHasStraightAngle(currentVector):
                    isPrevVectorStraight = True
            else:
                if isPrevVectorStraight:
                    StraightAngleSfs(prevNonStraightVector, currentVector.prev).skipVectors(manager)
                    isPrevVectorStraight = False
                # remember the last vector with a non straight angle
                prevNonStraightVector = currentVector
                
            currentVector = currentVector.next
            if currentVector is startVector:
                if isPrevVectorStraight:
                    StraightAngleSfs(prevNonStraightVector, currentVector.prev).skipVectors(manager)
                break
    
    def skipFeature(self, feature, manager):
        while True:
            if feature.isSkippable():
                feature.skipVectors(manager)
            if feature.prev:
                feature = feature.prev
            else:
                break
        

def _vectorHasStraightAngle(vector):
    return \
    (
        #vector.featureType==BldgPolygonFeature.quadrangle_convex and \
        vector.feature and \
        vector.feature.startSin and \
        vector.hasStraightAngle
    ) \
    or \
    (
        #vector.featureType!=BldgPolygonFeature.quadrangle_convex and \
        #vector.prev.featureType==BldgPolygonFeature.quadrangle_convex and \
        vector.prev.feature and \
        vector.prev.feature.nextSin and \
        vector.hasStraightAngle
    )