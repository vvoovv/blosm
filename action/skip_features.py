from building.feature import StraightAngleSfs


class SkipFeatures:
    
    def skipFeatures(self, polygon, checkIsSkippable, manager):
        # * Curved features aren't skipped.
        # * Complex features with 4 edges are processed separately since "free" neighbor vectors
        # are needed to skip the vectors of a complex feature. For example, the neighbor vectors
        # may be a part of a quadrangle feature and we need to skip the quadrangle feature first
        # to get the "free" neighbor edges.
        # * The triangle convex features are processed separately since they may be located at a corner
        # and have quadrangle features as neighbors. After skipping the quadrangle features, a sequence of
        # straight angle can be formed. It means that those triangle convex features at a corner
        # should be invalidated.
        
        startVector = None
        
        feature = polygon.smallFeature
        if feature:
            startVector = feature.startVector
            self._skipFeatures(feature, checkIsSkippable, manager)
        
        # complex features with 4 edges are treated separately
        feature = polygon.complex4Feature
        if feature:
            if not startVector:
                startVector = feature.startVector
            self._skipFeatures(feature, checkIsSkippable, manager)
        
        # triangular features are treated separetely
        feature = polygon.triangleFeature
        if feature:
            if not startVector:
                startVector = feature.startVector
            self._skipFeatures(feature, checkIsSkippable, manager)
            # calculate the sines for the skipped features
            self._calculateSinsSkipped(polygon.triangleFeature)
        
        # calculate the sines for the skipped features
        
        if polygon.smallFeature:
            self._calculateSinsSkipped(polygon.smallFeature)
        
        if polygon.complex4Feature:
            self._calculateSinsSkipped(polygon.complex4Feature)
        
        if startVector:
            self.skipStraightAnglesAfterSfs(startVector, manager)
    
    def skipStraightAnglesAfterSfs(self, startVector, manager):
        
        currentVector = startVector
        
        # find <prevNonStraightVector>
        isPrevVectorStraight = False
        while True:
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
    
    def _skipFeatures(self, feature, checkIsSkippable, manager):
        while True:
            if not checkIsSkippable or feature.isSkippable():
                feature.skipVectors(manager)
            else:
                # mark as <None> to distinguish the feature from "normal" unskipped features
                feature.skipped = None
            
            if feature.prev:
                feature = feature.prev
            else:
                break
    
    def _calculateSinsSkipped(self, feature):
        """
        Calculate sines for the skipped features.
        We have to skip ALL features first and only then calculate the sines for the skipped feature.
        Otherwise the sines would be incorrect.
        """
        while True:
            if feature.skipped:
                feature.calculateSinsSkipped()
            
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