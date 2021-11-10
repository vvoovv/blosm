from defs.building import BldgPolygonFeature


class UnskipFeatures:
    
    def unskipFeatures(self, polygon):
        # straight angle
        if polygon.saSfsFeature:
            self._unskipFeatures(polygon.saSfsFeature, BldgPolygonFeature.straightAngleSfs)
        # small features
        initialFeature = polygon.smallFeature or polygon.complex4Feature or polygon.triangleFeature
        if initialFeature:
            self._unskipFeatures(initialFeature, None)
    
    def _unskipFeatures(self, initialFeature, featureType):
        startVector = initialFeature.startVector
        currentVector = initialFeature.getProxyVector()
        while True:
            feature = currentVector.feature
            if feature:
                if feature.skipped and \
                        (
                            (featureType and feature.type == featureType) \
                            or\
                            (
                                not featureType and\
                                not feature.type in (BldgPolygonFeature.straightAngle, BldgPolygonFeature.curved)
                            )
                        ):
                    # Inherit facade or edge class from <currentVector.edge>.
                    # Important! Do it before <feature.unskipVectors()>
                    feature.inheritFacadeClass()
                    
                    feature.unskipVectors()
                    
                    currentVector = feature.endVector
                elif not feature.skipped:
                    currentVector = feature.endVector
            currentVector = currentVector.next
            if currentVector is startVector:
                break