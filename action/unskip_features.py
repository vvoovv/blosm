

class UnskipFeatures:
    
    def unskipFeatures(self, polygon):
        # straight angle
        if polygon.saSfsFeature:
            self._unskipFeatures(polygon.saSfsFeature)
        
        # restore the sines BEFORE unskipping the features to avoid incorrect values for the kept sines
        
        if polygon.complex4Feature:
            self._restoreSins(polygon.complex4Feature)
        
        if polygon.triangleFeature:
            self._restoreSins(polygon.triangleFeature)
        
        if polygon.smallFeature:
            self._restoreSins(polygon.smallFeature)
            self._unskipFeatures(polygon.smallFeature)
        
        if polygon.complex4Feature:
            self._unskipFeatures(polygon.complex4Feature)
        
        if polygon.triangleFeature:
            self._unskipFeatures(polygon.triangleFeature)
    
    def _restoreSins(self, feature):
        """
        Restore sines for the skipped features.
        We have to restore the sines first and only then unskipp ALL features.
        Otherwise the sines would be incorrect.
        """
        while True:
            # invalidated features are also ignored due to the condition below
            if feature.skipped:
                feature.restoreSins()
            
            if feature.prev:
                feature = feature.prev
            else:
                break
    
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