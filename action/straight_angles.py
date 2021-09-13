from building.feature import StraightAngle, Curved
from defs.building import BldgPolygonFeature


class StraightAngles:
    
    def do(self, manager):
        for building in manager.buildings:
            polygon = building.polygon
            
            # the start vector for a sequence of straight angles
            startVector = None
            
            curvedFeature = polygon.curvedFeature
            if curvedFeature:
                # If <polygon> has at least one curved feature, we process it here separately,
                # since every curved feature is skipped completely
                endVector = curvedFeature.startVector
                # the current vector
                vector = curvedFeature.endVector.next
                if vector is endVector:
                    # the whole polygon is curved, no straight angles here
                    continue
                # Check if <curvedFeature> is immediately followed by another curved feature
                if vector.featureType != BldgPolygonFeature.curved:
                    # a straight angle formed by <endVector> of a curved feature and its next vector is ignored
                    vector = vector.next
                    if vector is endVector:
                        # we have only one non-curved vector
                        continue
                while True:
                    if vector.featureType == BldgPolygonFeature.curved:
                        if startVector:
                            self.createStraightAngle(startVector, vector.prev, manager, True)
                            startVector = None
                        # Skip the curved features
                        # A straight angle formed by <endVector> of a curved feature and its next vector is ignored
                        vector = vector.feature.endVector.next
                        # Check if the curved feature is immediately followed by another curved feature
                        if vector.featureType == BldgPolygonFeature.curved:
                            vector = vector.prev
                    else:
                        if vector.hasStraightAngle:
                            if not startVector:
                                startVector = vector
                        elif startVector:
                            self.createStraightAngle(startVector, vector.prev, manager, True)
                            startVector = None
                    vector = vector.next
                    if vector is endVector:
                        break
            else:
                firstVector = True
                # Do we have a straight angle feature ("sa" stands for "straight angle") at
                # the first vector
                saAtFirstVector = False
                firstVectorSaFeature = None
                for vector in polygon.getVectorsExceptCurved():
                    if vector.hasStraightAngle:
                        if not startVector:
                            startVector = vector
                            if firstVector:
                                saAtFirstVector = True
                    elif startVector:
                        if saAtFirstVector:
                            firstVectorSaFeature = self.createStraightAngle(startVector, vector.prev, manager, False)
                            saAtFirstVector = False
                        else:
                            self.createStraightAngle(startVector, vector.prev, manager, True)
                        startVector = None
                    if firstVector:
                        firstVector = False
                if startVector:
                    if firstVectorSaFeature:
                        firstVectorSaFeature.setStartVector(startVector.prev)
                        self.skipVectors(firstVectorSaFeature, manager)
                    else:
                        # a sequence of straight angles at the end of the for-cycle
                        self.createStraightAngle(startVector, vector, manager, True)
    
    def createStraightAngle(self, startVectorNext, endVector, manager, skipVectors):
        """
        Create a feature for a sequence of straight angles
        """
        feature = StraightAngle(startVectorNext.prev, endVector, BldgPolygonFeature.straightAngle)
        if skipVectors:
            self.skipVectors(feature, manager)
        return feature
    
    def skipVectors(self, feature, manager):
        if feature.isCurved():
            Curved(feature.startVector, feature.endVector)
        else:
            feature.skipVectors(manager)