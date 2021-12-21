from building import BldgPolygon
from building.feature import StraightAnglePart, StraightAngle, Curved
from defs.building import BldgPolygonFeature


class StraightAngles:
    
    def do(self, manager):
        for building in manager.buildings:
            self.processStraightAngles(building.polygon, manager)
        
        for building in manager.buildings:
            if building.polygon.saFeature:
                self.processStraightAnglesFreeEdges(building.polygon, manager)
        
        for building in manager.buildings:
            if building.polygon.saFeature:
                self.processStraightAnglesSharedEdges(building.polygon, manager)
    
    def processStraightAngles(self, polygon, manager):
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
                return
            # Check if <curvedFeature> is immediately followed by another curved feature
            if vector.featureType != BldgPolygonFeature.curved:
                # a straight angle formed by <endVector> of a curved feature and its next vector is ignored
                vector = vector.next
                if vector is endVector:
                    # we have only one non-curved vector
                    return
            while True:
                if vector.featureType == BldgPolygonFeature.curved:
                    if startVector:
                        self.createStraightAngle(startVector, vector.prev)
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
                        self.createStraightAngle(startVector, vector.prev)
                        startVector = None
                vector = vector.next
                if vector is endVector:
                    if startVector:
                        self.createStraightAngle(startVector, vector.prev)
                    break
        else:
            firstVector = True
            # Do we have a straight angle feature ("sa" stands for "straight angle") at
            # the first vector
            saAtFirstVector = False
            firstVectorSaFeature = None
            for vector in polygon.getVectors():
                if vector.hasStraightAngle:
                    if not startVector:
                        startVector = vector
                        if firstVector:
                            saAtFirstVector = True
                elif startVector:
                    if saAtFirstVector:
                        firstVectorSaFeature = (startVector, vector.prev)
                        saAtFirstVector = False
                    else:
                        self.createStraightAngle(startVector, vector.prev)
                    startVector = None
                if firstVector:
                    firstVector = False
            if startVector:
                if firstVectorSaFeature:
                    self.createStraightAngle(startVector, firstVectorSaFeature[1])
                else:
                    # a sequence of straight angles at the end of the for-cycle
                    self.createStraightAngle(startVector, vector)
            elif firstVectorSaFeature:
                self.createStraightAngle(firstVectorSaFeature[0], firstVectorSaFeature[1])
    
    def createStraightAngle(self, startVectorNext, endVector):
        """
        Create a feature for a sequence of straight angles
        """
        startVector = startVectorNext.prev
        # check if it is actually a curved feature.
        # Calculate the sine of the angle between <startVector> and the vector
        # from <startVector.v1> and <endVector.v2>
        # sin = vector.cross(self.startVector.unitVector)
        if not startVectorNext is endVector and \
            (endVector.v2 - startVector.v1).normalized().cross(startVector.unitVector) \
                > BldgPolygon.straightAngleSin:
            # create a curved feature instead
            Curved(startVector, endVector)
        else:
            StraightAngle(startVector, endVector, BldgPolygonFeature.straightAngle)
            # temporily set attribute <startVector.feature> to None
            startVector.feature = None
    
    def processStraightAnglesFreeEdges(self, polygon, manager):
        saFeature = polygon.saFeature
        startVector = None
        while True:
            
            numFreeEdges = 0
            vector = saFeature.startVector
            while True:
                # <vector>'s edge is shared with another building's polygon
                if vector.edge.hasSharedBldgVectors():
                    if numFreeEdges:
                        if numFreeEdges > 1:
                            StraightAnglePart(startVector, vector.prev, BldgPolygonFeature.straightAngle).skipVectors(manager)
                        numFreeEdges = 0
                else:
                    if len(manager.data.nodes[vector.id1].bldgVectors) > 1:
                        if numFreeEdges > 1:
                            StraightAnglePart(startVector, vector.prev, BldgPolygonFeature.straightAngle).skipVectors(manager)
                        numFreeEdges = 1
                        startVector = vector
                    else:
                        if not numFreeEdges:
                            startVector = vector
                        numFreeEdges += 1
                    
                if vector is saFeature.endVector:
                    if numFreeEdges > 1 and not startVector is saFeature.startVector:
                        StraightAnglePart(startVector, vector, BldgPolygonFeature.straightAngle).skipVectors(manager)
                        # A straight angle feature was formed in the line above with the constructor <StraightAnglePart>.
                        # <saFeature.endVector> is now the start vector of that feature, i.e. <startVector>.
                        # So we change <saFeature.endVector> to <startVector>.
                        saFeature.endVector = startVector
                    break
                vector = vector.next
            
            if saFeature.prev:
                saFeature = saFeature.prev
            else:
                break
    
    def processStraightAnglesSharedEdges(self, polygon, manager):
        saFeature = polygon.saFeature
        startVector = neighborPolygon = None
        while True:
            
            numSharedEdges = 0
            vector = saFeature.startVector
            while True:
                # <vector>'s edge is shared with another building's polygon
                if vector.edge.hasSharedBldgVectors():
                    if not saFeature.hasSharedEdge:
                        saFeature.hasSharedEdge = True
                    
                    if vector.neighbor.featureType == BldgPolygonFeature.curved:
                        if not saFeature.sharesCurve:
                            saFeature.sharesCurve = True
                        if numSharedEdges > 1:
                            StraightAnglePart(startVector, vector.prev, BldgPolygonFeature.straightAngle).skipVectors(manager)
                        numSharedEdges = 0
                    elif numSharedEdges and neighborPolygon is vector.neighbor.polygon:
                        numSharedEdges += 1
                    else:
                        if numSharedEdges and saFeature.sharesOnePolygon:
                            saFeature.sharesOnePolygon = False
                        if numSharedEdges > 1:
                            StraightAnglePart(startVector, vector.prev, BldgPolygonFeature.straightAngle).skipVectors(manager)
                        startVector = vector
                        neighborPolygon = vector.neighbor.polygon
                        numSharedEdges = 1
                else:
                    if not saFeature.hasFreeEdge:
                        saFeature.hasFreeEdge = True
                    if numSharedEdges:
                        if numSharedEdges > 1:
                            StraightAnglePart(startVector, vector.prev, BldgPolygonFeature.straightAngle).skipVectors(manager)
                        numSharedEdges = 0
                    
                if vector is saFeature.endVector:
                    if numSharedEdges > 1 and not startVector is saFeature.startVector:
                        StraightAnglePart(startVector, vector, BldgPolygonFeature.straightAngle).skipVectors(manager)
                        # A straight angle feature was formed in the line above with the constructor <StraightAnglePart>.
                        # <saFeature.endVector> is now the start vector of that feature, i.e. <startVector>.
                        # So we change <saFeature.endVector> to <startVector>.
                        saFeature.endVector = startVector
                    break
                vector = vector.next
            
            saFeature.setParentFeature()
            saFeature.markVectors()
            if not saFeature.hasSharedEdge or (not saFeature.hasFreeEdge and not saFeature.sharesCurve and saFeature.sharesOnePolygon):
                saFeature.skipVectors(manager)
            
            if saFeature.prev:
                saFeature = saFeature.prev
            else:
                break