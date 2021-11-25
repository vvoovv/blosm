

class SkipFeaturesAgain:
    """
    This action is used to test how the feature can be skipped again after unskipping
    """
    
    def __init__(self, unskipFeaturesAction=None):
        self.unskipFeaturesAction = unskipFeaturesAction
    
    def do(self, manager):
        
        for building in manager.buildings:
            polygon = building.polygon
            
            if self.unskipFeaturesAction:
                self.unskipFeaturesAction.unskipFeatures(polygon)
            
            if polygon.saFeature and _neededFeatureDetectionAgain(polygon):
                pass
            else:
                self.skipFeatures(polygon, manager)
    
    def skipFeatures(self, polygon, manager):
        detectSaSfsAgain = False
        
        # quadrangular features and features with 5 edges
        feature = polygon.smallFeature
        if feature:
            detectSaSfsAgain = self._skipFeatures(feature, manager) or detectSaSfsAgain
        
        # complex features with 4 edges
        feature = polygon.complex4Feature
        if feature:
            detectSaSfsAgain = self._skipFeatures(feature, manager) or detectSaSfsAgain
        
        # triangular features
        feature = polygon.triangleFeature
        if feature:
            detectSaSfsAgain = self._skipFeatures(feature, manager) or detectSaSfsAgain
        
        # straight angle features (<sfs> stands for "small feature skipped")
        if detectSaSfsAgain:
            pass
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
        detectSaSfsAgain = False
        while True:
            # <feature.skipped> was set to <None> if <feature.isSkippable()> returned <False> 
            if feature.skipped is None:
                feature.skipVectors(manager)
                if not detectSaSfsAgain:
                    detectSaSfsAgain = True
            else:
                feature.skipVectorsCached()
            if feature.prev:
                feature = feature.prev
            else:
                break
        return detectSaSfsAgain


def _neededFeatureDetectionAgain(polygon):
        saFeature = polygon.saFeature
        while True:
            
            if not saFeature.skipped:
                return True
            
            if saFeature.prev:
                saFeature = saFeature.prev
            else:
                break
        
        return False