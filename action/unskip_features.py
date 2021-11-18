from defs.building import BldgPolygonFeature


class UnskipFeatures:
    
    def unskipFeatures(self, polygon):
        # straight angle
        if polygon.saSfsFeature:
            self._unskipFeatures(polygon.saSfsFeature)
        
        if polygon.smallFeature:
            self._unskipFeatures(polygon.smallFeature)
        
        if polygon.complex4Feature:
            self._unskipFeatures(polygon.complex4Feature)
        
        if polygon.triangleFeature:
            self._unskipFeatures(polygon.triangleFeature)
    
    def _unskipFeatures(self, feature):
        while True:
            # invalidated features are also ignored due to the condition below
            if feature.skipped:
                # Inherit facade or edge class from <currentVector.edge>.
                # Important! Do it before <feature.unskipVectors()>
                feature.inheritFacadeClass()
                
                feature.unskipVectors()
            
            if feature.prev:
                feature = feature.prev
            else:
                break