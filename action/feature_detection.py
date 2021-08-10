import re
from defs.building import BldgPolygonFeature, curvedLengthFactor, \
    longEdgeFactor, midEdgeFactor, sin_lo, sin_me
from building.feature import StraightAngle, Curved, ComplexConvex, QuadConvex, \
    ComplexConcave, QuadConcave, TriConvex, TriConcave
from defs.building import StraightAngleType


def hasAnglesForCurvedFeature(vector):
    """
    Checks if <vector.edge> has an angle between 5° and 30° on both ends of <vector.edge>
    """
    return sin_lo<abs(vector.sin)<sin_me and sin_lo<abs(vector.next.sin)<sin_me


def processCurvedFeature(feature, polygon):
    if not polygon.curvedFeature:
        polygon.curvedFeature = feature


def processSmallFeature(feature, polygon):
    if not polygon.smallFeature:
        polygon.smallFeature = feature


def vectorHasStraightAngle(vector):
    return \
    (
        vector.featureId==BldgPolygonFeature.quadrangle_convex and \
        vector.feature.startSin and \
        vector.hasStraightAngle
    ) \
    or \
    (
        vector.featureId!=BldgPolygonFeature.quadrangle_convex and \
        vector.prev.featureId==BldgPolygonFeature.quadrangle_convex and \
        vector.prev.feature.nextSin and \
        vector.hasStraightAngle
    )


class FeatureDetection:
    
    # a sequence of two or more 'C' matches as curvy sequence.
    # possibly with ends 'S' (start) or 'E' (end)
    curvedPattern = re.compile(r"S?(C){3,}")
    
    # convex complex features
    complexConvexPattern = re.compile(r"(>[L|l]{2,3}<)")

    # concave complex features
    complexConcavePattern = re.compile(r"(<[R|r]{2,3}>)")

    # convex quadrangle features
    quadConvexPattern = re.compile(r"(>[L|l][<|L|R|O|+|=|o])|([>|L|R|O|+|=|o][L|l]<)")
    #quadConvexPattern = re.compile(r"([>|+][L|l][<|L|R|+|=|o])|([>|L|R|+|=|o][L|l][<|=])")  # exclude long ends

    # concave quadrangle features
    quadConcavePattern = re.compile(r"((<[R|r][>|L|R|O|+|=|o])|([>|L|R|O|+|=|o][R|r]>))")
    #quadConcavePattern = re.compile(r"(([<|=][R|r][>|L|R|+|=|o])|([>|L|R|+|=|o][R|r][>|+]))")  # exclude long ends

    # convex triangular features
    # triangle = r">(>|<|l){1,}"
    # left_triangle = r"(l<)" # special case for triangular part of rectangle
    # triConvexPattern = re.compile(triangle + r"|" + left_triangle)
    triConvexPattern = re.compile(r">(>|<|l){1,}" + r"|" + r"(l<)")
      
    # concave triangular features
    triConcavePattern = re.compile(r"<(>|<|r){1,}")

    
    def __init__(self, simplifyPolygons=True):
        self.simplifyPolygons = simplifyPolygons

    def do(self, manager):

        for building in manager.buildings:
            polygon = building.polygon
            
            polygon.prepareVectorsByIndex()

            # detect curved features
            self.detectCurvedFeatures(polygon)
            
            midEdgeThreshold = max(midEdgeFactor * polygon.dimension, 2.)
            longEdgeThreshold = longEdgeFactor * polygon.dimension
            sequence = self.getSequence(polygon, midEdgeThreshold, longEdgeThreshold)
            
            self.detectQuadrangularFeatures(polygon, sequence)
            if self.simplifyPolygons and polygon.smallFeature:
                self.skipQuadrangularFeatures(polygon, manager)
    
    def detectCurvedFeatures(self, polygon):        
        numLowAngles = sum(
            1 for vector in polygon.getVectors() if hasAnglesForCurvedFeature(vector)
        )
        if not numLowAngles:
            return

        # Calculate a length threshold as a mean of the vector lengths among the vectors
        # that satisfy the condition <hasAnglesForCurvedFeature(vector)>
        curvedLengthThreshold = curvedLengthFactor / numLowAngles *\
            sum(
                vector.length for vector in polygon.getVectors()\
                if hasAnglesForCurvedFeature(vector)
            )

        # Feature character sequence: edges with angle between 5° and 30° on either end 
        # and a length below <curvyLengthThresh> get a 'C', else a '0'
        sequence = ''.join(
            '0' if vector.length>curvedLengthThreshold else ( \
                'C' if sin_lo<abs(vector.sin)<sin_me else ( \
                    'S' if sin_lo<abs(vector.next.sin)<sin_me else '0'
                )
            )
            for vector in polygon.getVectors()
        )

        sequenceLength = len(sequence)
        sequence = sequence+sequence # allow cyclic pattern

        self.matchPattern(
            sequence, sequenceLength,
            FeatureDetection.curvedPattern,
            Curved, processCurvedFeature,
            polygon, ''
        )
    
    def detectQuadrangularFeatures(self, polygon, sequence):
        """
        Detects quadrangular patterns (simple and complex ones).
        """
        
        # debug
        self.debugSetFeatureSymbols(polygon, sequence)

        sequenceLength = len(sequence)
        sequence = sequence+sequence # allow cyclic pattern

        sequence = self.matchPattern(
            sequence, sequenceLength,
            FeatureDetection.complexConvexPattern,
            ComplexConvex, processSmallFeature,
            polygon, '1'
        )

        sequence = self.matchPattern(
            sequence, sequenceLength,
            FeatureDetection.quadConvexPattern,
            QuadConvex, processSmallFeature,
            polygon, '2'
        )

        sequence = self.matchPattern(
            sequence, sequenceLength,
            FeatureDetection.complexConcavePattern,
            ComplexConcave, processSmallFeature,
            polygon, '3'
        )

        sequence = self.matchPattern(
            sequence, sequenceLength,
            FeatureDetection.quadConcavePattern,
            QuadConcave, processSmallFeature,
            polygon, '4'
        )   

        sequence = self.matchPattern(
            sequence, sequenceLength,
            FeatureDetection.triConvexPattern,
            TriConvex, processSmallFeature,
            polygon, '5'
        )
        
        self.matchPattern(
            sequence, sequenceLength,
            FeatureDetection.triConcavePattern,
            TriConcave, processSmallFeature,
            polygon, ''
        )

    def matchPattern(self, sequence, sequenceLength, pattern, featureConstructor, featureFunc, polygon, subChar):
        matches = [r for r in pattern.finditer(sequence)]
        if matches:
            for featureSeg in matches:
                s = featureSeg.span()
                if 0 <= s[0] < sequenceLength:
                    feature = featureConstructor(
                        polygon.getVectorByIndex(s[0]), # start vector
                        polygon.getVectorByIndex( (s[1]-1) % sequenceLength ) # end vector
                    )
                    featureFunc(feature, polygon)
                    if subChar and s[1] >= sequenceLength:  # case of cyclic pattern
                        sequence = subChar*(s[1]-sequenceLength) + sequence[(s[1]-sequenceLength):]
            if subChar:
                sequence = re.sub(pattern, lambda m: subChar * len(m.group()), sequence)
        return sequence
    
    def getSequence(self, polygon, midEdgeThreshold, longEdgeThreshold):
        # Long edges (>=lengthThresh and <maxLengthThreshold):
        #       'L': sharp left at both ends
        #       'R': sharp right at both ends
        #       '+': alternate sharp angle, starting to right
        #       '=': alternate sharp angle, starting to left
        #       'O': other long edge
        # Short edges:
        #       'l': medium left at both ends
        #       'r': medium right at both ends
        #       '>': alternate medium angle, starting to right
        #       '<': alternate medium angle, starting to left
        #       'o': other short edge
        return ''.join(
            (
                # prevent interference of the detected curved segments with the small features
                'X' if vector.featureId==BldgPolygonFeature.curved else
                (
                    (
                        'L' if ( vector.sin > sin_me and vector.next.sin > sin_me ) else (
                            'R' if ( vector.sin < -sin_me and vector.next.sin < -sin_me ) else (
                                '+' if ( vector.sin < -sin_me and vector.next.sin > sin_me ) else (
                                    '=' if (vector.sin > sin_me and vector.next.sin < -sin_me) else 'o'
                                )
                            )
                        )
                    ) if vector.length < longEdgeThreshold else 'O'
                ) \
                if vector.length >= midEdgeThreshold else (
                    'l' if (vector.sin > sin_me and vector.next.sin > sin_me) else (
                        'r' if (vector.sin < -sin_me and vector.next.sin < -sin_me) else (
                            '>' if ( vector.sin < -sin_me and vector.next.sin > sin_me ) else (
                                '<' if (vector.sin > sin_me and vector.next.sin < -sin_me) else 'o'
                            )
                        )
                    )
                )
            ) \
            for vector in polygon.getVectors()
        )
    
    def skipQuadrangularFeatures(self, polygon, manager):
        currentVector = startVector = polygon.smallFeature.startVector
        while True:
            feature = currentVector.feature
            if feature and feature.featureId == BldgPolygonFeature.quadrangle_convex:
                feature.skipVectors(manager) 
                currentVector = feature.endVector.next
            else:
                currentVector = currentVector.next
            if currentVector is startVector:
                break
        
        # find <prevNonStraightVector>
        if currentVector.featureId == BldgPolygonFeature.quadrangle_convex:
            startVector = currentVector
            while True:
                feature = currentVector.feature
                if not vectorHasStraightAngle(currentVector):
                    prevNonStraightVector = currentVector
                    break
                polygon.numEdges -= 1
                currentVector = currentVector.prev
            currentVector = startVector
        else:
            prevNonStraightVector = currentVector = currentVector.next
        startVector = prevNonStraightVector
        isPrevVectorStraight = False
        while True:
            # conditions for a straight angle
            if vectorHasStraightAngle(currentVector):
                    polygon.numEdges -= 1
                    isPrevVectorStraight = True
            else:
                if isPrevVectorStraight:
                    self.createStraightAngleFeature(prevNonStraightVector, currentVector.prev, manager)
                    isPrevVectorStraight = False
                # remember the last vector with a non straight angle
                prevNonStraightVector = currentVector
                
            currentVector = currentVector.next
            if currentVector is startVector:
                if isPrevVectorStraight:
                    self.createStraightAngleFeature(prevNonStraightVector, currentVector.prev, manager)
                break
    
    def createStraightAngleFeature(self, startVector, endVector, manager):
        if startVector.featureId == BldgPolygonFeature.triangle_convex:
            startVector.feature.invalidate()
        if endVector.featureId == BldgPolygonFeature.triangle_convex:
            endVector.feature.invalidate()
        StraightAngle(
            startVector,
            endVector,
            StraightAngleType.smallFeatureSkipped
        ).skipVectors(manager)
        
    
    def debugSetFeatureSymbols(self, polygon, sequence):
        for vector,symbol in zip(polygon.getVectors(), sequence):
            vector.featureSymbol = symbol