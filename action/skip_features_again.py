

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
                print(polygon.building.outline.tags["id"])
            else:
                print(False)


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