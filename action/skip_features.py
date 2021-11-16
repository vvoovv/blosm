from defs.building import BldgPolygonFeature
from building.feature import StraightAngleSfs


class SkipFeatures:
    
    def skipFeatures(self, polygon, manager):
        if polygon.smallFeature:
            currentVector = startVector = polygon.smallFeature.startVector
            while True:
                feature = currentVector.feature
                if feature:
                    # * Curved features aren't skipped.
                    # * Complex features with 4 edges are processed separately since "free" neighbor vectors
                    # are needed to skip the vectors of a complex feature. For example, the neighbor vectors
                    # may be a part of a quadrangle feature and we need to skip the quadrangle feature first
                    # to get the "free" neighbor edges.
                    # * The triangle convex features are processed separately since they may be located at a corner
                    # and have quadrangle features as neighbors. After skipping the quadrangle features, a sequence of
                    # straight angle can be formed. It means that those triangle convex features at a corner
                    # should be invalidated.
                    if feature.type in (
                                BldgPolygonFeature.quadrangle_convex,
                                BldgPolygonFeature.quadrangle_concave,
                                BldgPolygonFeature.complex5_convex
                            ) and \
                            feature.isSkippable():
                        feature.skipVectors(manager) 
                    currentVector = feature.endVector.next
                else:
                    currentVector = currentVector.next
                if currentVector is startVector:
                    break
        
        # complex features with 4 edges are treated separately
        if polygon.complex4Feature:
            currentVector = startVector = polygon.complex4Feature.startVector
            while True:
                feature = currentVector.feature
                if feature:
                    if feature.type == BldgPolygonFeature.complex4_convex and \
                            feature.isSkippable():
                        feature.skipVectors(manager)
                    currentVector = feature.endVector.next
                else:
                    currentVector = currentVector.next
                if currentVector is startVector:
                    break
        
        # triangular features are treated separetely
        if polygon.triangleFeature:
            currentVector = startVector = polygon.triangleFeature.startVector
            while True:
                feature = currentVector.feature
                if feature:
                    if feature.type == BldgPolygonFeature.triangle_convex and \
                            feature.isSkippable():
                        feature.skipVectors(manager)
                    currentVector = feature.endVector.next
                else:
                    currentVector = currentVector.next
                if currentVector is startVector:
                    break
        
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