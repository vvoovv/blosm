import re
from defs.building import BldgPolygonFeature, \
    longEdgeFactor, midEdgeFactor, sin_lo, sin_me
from building.feature import StraightAngleSfs, ComplexConvex5, ComplexConvex4, QuadConvex, \
    QuadConcave, TriConvex, TriConcave


def hasAnglesForCurvedFeature(vector):
    """
    Checks if <vector.edge> has an angle between 5° and 30° on both ends of <vector.edge>
    """
    return sin_lo<abs(vector.sin)<sin_me and sin_lo<abs(vector.next.sin)<sin_me


def vectorHasStraightAngle(vector):
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


class FeatureDetection:
    
    # a sequence of two or more 'C' matches as curvy sequence.
    # possibly with ends 'S' (start) or 'E' (end)
    curvedPattern = re.compile(r"S?(C){3,}")
    
    # convex complex features (exactly 5 eddges)
    complexConvexPattern5 = re.compile(r"(>[L|l][L|l|O][L|l]<)")
    
    # convex complex features (exactly 4 edges)
    complexConvexPattern4 = re.compile(r"(>[L|l][L|l]<)")

    # concave complex features
    #complexConcavePattern = re.compile(r"(<[R|r][R|r]>)")

    # convex quadrangle features
    quadConvexPattern = re.compile(r"(>[L|l][<|L|R|O|+|=|o])|([>|L|R|O|+|=|o][L|l]<)")
    #quadConvexPattern = re.compile(r"([>|+][L|l][<|L|R|+|=|o])|([>|L|R|+|=|o][L|l][<|=])")  # exclude long ends

    # concave quadrangle features
    quadConcavePattern = re.compile(r"((<[R|r][>|L|R|O|+|=|o])|([>|L|R|O|+|=|o][R|r]>))")
    #quadConcavePattern = re.compile(r"(([<|=][R|r][>|L|R|+|=|o])|([>|L|R|+|=|o][R|r][>|+]))")  # exclude long ends

    # convex triangular features
    triConvexPattern = re.compile(r"[+|>][=|<]")
      
    # concave triangular features
    triConcavePattern = re.compile(r"[=|<][+|>]")

    
    def __init__(self, simplifyPolygons=True):
        self.simplifyPolygons = simplifyPolygons

    def do(self, manager):

        for building in manager.buildings:
            polygon = building.polygon
            
            polygon.prepareVectorsByIndex()
            
            self.detectFeatures(polygon)
            
            if self.simplifyPolygons and (polygon.smallFeature or polygon.complex4Feature or polygon.triangleFeature):
                self.skipFeatures(polygon, manager)
    
    def detectFeatures(self, polygon):
        """
        Detects patterns for small features
        """

        midEdgeThreshold = max(midEdgeFactor * polygon.dimension, 2.)
        longEdgeThreshold = longEdgeFactor * polygon.dimension
        
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
        sequence = ''.join(
            (
                # prevent interference of the detected curved segments with the small features
                'X' if vector.featureType==BldgPolygonFeature.curved else
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
        
        # debug
        self.debugSetFeatureSymbols(polygon, sequence)

        sequenceLength = len(sequence)
        sequence = sequence+sequence # allow cyclic pattern

        sequence = self.matchPattern(
            sequence, sequenceLength,
            FeatureDetection.complexConvexPattern5,
            ComplexConvex5,
            polygon, '1'
        )
        
        sequence = self.matchPattern(
            sequence, sequenceLength,
            FeatureDetection.complexConvexPattern4,
            ComplexConvex4,
            polygon, '2'
        )

        sequence = self.matchPattern(
            sequence, sequenceLength,
            FeatureDetection.quadConvexPattern,
            QuadConvex,
            polygon, '3'
        )

        sequence = self.matchPattern(
            sequence, sequenceLength,
            FeatureDetection.quadConcavePattern,
            QuadConcave,
            polygon, '4'
        )   

        self.matchPattern(
            sequence, sequenceLength,
            FeatureDetection.triConvexPattern,
            TriConvex,
            polygon, ''
        )
        
        #self.matchPattern(
        #    sequence, sequenceLength,
        #    FeatureDetection.triConcavePattern,
        #    TriConcave,
        #    polygon, ''
        #)

    def matchPattern(self, sequence, sequenceLength, pattern, featureConstructor, polygon, subChar):
        matches = [r for r in pattern.finditer(sequence)]
        if matches:
            for featureSeg in matches:
                s = featureSeg.span()
                if 0 <= s[0] < sequenceLength:
                    featureConstructor(
                        polygon.getVectorByIndex(s[0]), # start vector
                        polygon.getVectorByIndex( (s[1]-1) % sequenceLength ) # end vector
                    )
                    if subChar and s[1] >= sequenceLength:  # case of cyclic pattern
                        sequence = subChar*(s[1]-sequenceLength) + sequence[(s[1]-sequenceLength):]
            if subChar:
                sequence = re.sub(pattern, lambda m: subChar * len(m.group()), sequence)
        return sequence
    
    def skipFeatures(self, polygon, manager):
        if polygon.smallFeature:
            currentVector = startVector = polygon.smallFeature.startVector
            while True:
                feature = currentVector.feature
                if feature:
                    # * Curved features aren't skipped.
                    # * Complex features with 4 edges are processed separately since "free" neighbor vectors
                    # are needed to skip the vectors of a complex feature. For example, the neighbor vectors
                    # may be a part of a quadrangle feature and we need to skip the quadrangle feature first
                    # to get the "free" neighbor edges.
                    # * The triangle convex features are processed separately since they may be located at a corner
                    # and have quadrangle features as neighbors. After skipping the quadrangle features, a sequence of
                    # straight angle can be formed. It means that those triangle convex features at a corner
                    # should be invalidated.
                    if feature.type in (
                                BldgPolygonFeature.quadrangle_convex,
                                BldgPolygonFeature.quadrangle_concave,
                                BldgPolygonFeature.complex5_convex
                            ) and \
                            feature.isSkippable():
                        feature.skipVectors(manager) 
                    currentVector = feature.endVector.next
                else:
                    currentVector = currentVector.next
                if currentVector is startVector:
                    break
        
        # complex features with 4 edges are treated separately
        if polygon.complex4Feature:
            currentVector = startVector = polygon.complex4Feature.startVector
            while True:
                feature = currentVector.feature
                if feature:
                    if feature.type == BldgPolygonFeature.complex4_convex and \
                            feature.isSkippable():
                        feature.skipVectors(manager)
                    currentVector = feature.endVector.next
                else:
                    currentVector = currentVector.next
                if currentVector is startVector:
                    break
        
        # triangular features are treated separetely
        if polygon.triangleFeature:
            currentVector = startVector = polygon.triangleFeature.startVector
            while True:
                feature = currentVector.feature
                if feature:
                    if feature.type == BldgPolygonFeature.triangle_convex and \
                            feature.isSkippable():
                        feature.skipVectors(manager)
                    currentVector = feature.endVector.next
                else:
                    currentVector = currentVector.next
                if currentVector is startVector:
                    break
        
        # find <prevNonStraightVector>
        isPrevVectorStraight = False
        while True:
            feature = currentVector.feature
            if not vectorHasStraightAngle(currentVector):
                prevNonStraightVector = currentVector
                break
            polygon.numEdges -= 1
            isPrevVectorStraight = True
            currentVector = currentVector.prev
        currentVector = startVector.next
        startVector = prevNonStraightVector
        while True:
            # conditions for a straight angle
            if vectorHasStraightAngle(currentVector):
                    polygon.numEdges -= 1
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
    
    def debugSetFeatureSymbols(self, polygon, sequence):
        for vector,symbol in zip(polygon.getVectors(), sequence):
            vector.featureSymbol = symbol