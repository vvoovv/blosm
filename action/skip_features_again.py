from defs.building import BldgPolygonFeature


class SkipFeaturesAgain:
    """
    This action is used to test how the feature can be skipped again after unskipping
    """
    
    def __init__(self, skipFeaturesAction, unskipFeaturesAction, featureDetectionAction):
        self.skipFeaturesAction = skipFeaturesAction
        self.unskipFeaturesAction = unskipFeaturesAction
        self.featureDetectionAction = featureDetectionAction
    
    def do(self, manager):
        
        for building in manager.buildings:
            polygon = building.polygon
            
            if self.unskipFeaturesAction:
                self.unskipFeaturesAction.unskipFeatures(polygon)
            
            if polygon.saFeature and self.skipSaFeatures(polygon, manager):
                self.featureDetectionAction.detectFeatures(polygon)
                self.skipFeaturesAction.skipFeatures(polygon, False, manager)
            else:
                self.skipFeatures(polygon, manager)
    
    def skipFeatures(self, polygon, manager):
        # there are features (with <feature.skip> equal to zero) that were not skipped in the first pass
        featuresNotSkipped = False
        startVector = None
        
        # quadrangular features and features with 5 edges
        feature = polygon.smallFeature
        if feature:
            featuresNotSkipped = self._skipFeatures(feature, manager) or featuresNotSkipped
            if featuresNotSkipped:
                startVector = feature.startVector
        
        # complex features with 4 edges
        feature = polygon.complex4Feature
        if feature:
            featuresNotSkipped = self._skipFeatures(feature, manager) or featuresNotSkipped
            if featuresNotSkipped and not startVector:
                startVector = feature.startVector
        
        # triangular features
        feature = polygon.triangleFeature
        if feature:
            featuresNotSkipped = self._skipFeatures(feature, manager) or featuresNotSkipped
            if featuresNotSkipped and not startVector:
                startVector = feature.startVector
        
        # straight angle features (<sfs> stands for "small feature skipped")
        if featuresNotSkipped:
            # calculate sines for the the features that were not skipped in the first pass
            
            if polygon.smallFeature:
                self._calculateSins(polygon.smallFeature)
            
            if polygon.complex4Feature:
                self._calculateSins(polygon.complex4Feature)
            
            if polygon.triangleFeature:
                self._calculateSins(polygon.triangleFeature)
            
            polygon.saSfsFeature = None
            self.skipFeaturesAction.skipStraightAnglesAfterSfs(startVector, manager)
        else:
            feature = polygon.saSfsFeature
            if feature:
                while True:
                    feature.skipVectors(manager)
                    feature.markVectors()
                    
                    if feature.prev:
                        feature = feature.prev
                    else:
                        break
    
    def _skipFeatures(self, feature, manager):
        featuresNotSkipped = False
        while True:
            # <feature.skipped> was set to <None> if <feature.isSkippable()> returned <False> 
            if feature.skipped is None:
                feature.skipVectors(manager)
                if not featuresNotSkipped:
                    featuresNotSkipped = True
            else:
                feature.skipVectorsCached()
            if feature.prev:
                feature = feature.prev
            else:
                break
        return featuresNotSkipped
    
    def _calculateSins(self, feature):
        """
        Calculate sines for the features that were not skipped in the first pass
        """
        startFeature = feature
        while True:
            # a condition to detect a feature that was not skipped in the first pass
            if feature.startSin or feature.endSin:
                prevVectorFeature = feature.startVector.prev.feature
                if prevVectorFeature and not (prevVectorFeature.startSin or prevVectorFeature.endSin):
                    feature.startVector.calculateSin()
            if feature.prev:
                feature = feature.prev
            else:
                break
        
        feature = startFeature
        while True:
            # a condition to detect a feature that was not skipped in the first pass
            if not (feature.startSin or feature.endSin):
                feature.calculateSinsSkipped()
            if feature.prev:
                feature = feature.prev
            else:
                break
    
    def skipSaFeatures(self, polygon, manager):
        """
        If there are straight angle features that were not skipped, perfom cleanup,
        then skip those straight angle features.
        
        Returns:
            <True> if there are straight angle features that were not skipped,
            <False> otherwise.
        """
        saFeature = polygon.saFeature
        result = False
        while True:
            if not saFeature.skipped:
                # check if <saFeature.startVector.feature> was overriden by other features
                if not saFeature.startVector.feature is saFeature:
                    saFeature.startVector.feature = saFeature
                if not result:
                    result = True
            
            if saFeature.prev:
                saFeature = saFeature.prev
            else:
                break
            
        if result:
            # Cleanup vectors from possible features. Stgraight angle features will be preserved
            for vector in polygon.getVectors():
                if vector.feature and vector.featureType != BldgPolygonFeature.straightAngle:
                    vector.feature = None
            polygon.smallFeature = None
            polygon.complex4Feature = None
            polygon.triangleFeature = None
            polygon.saSfsFeature = None
            
            # skip vectors for the straight angle features
            saFeature = polygon.saFeature
            while True:
                if not saFeature.skipped:
                    saFeature.skipVectors(manager)
                
                if saFeature.prev:
                    saFeature = saFeature.prev
                else:
                    break
        
        return result