import re
from defs.building import BldgPolygonFeature, \
    longEdgeFactor, midEdgeFactor, sin_me
from building.feature import ComplexConvex5, ComplexConvex4, QuadConvex, \
    QuadConcave, TriConvex


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

    
    def __init__(self, skipFeaturesAction=None):
        self.skipFeaturesAction = skipFeaturesAction

    def do(self, manager):

        for building in manager.buildings:
            polygon = building.polygon
            
            self.detectFeatures(polygon)
            
            if self.skipFeaturesAction and (polygon.smallFeature or polygon.complex4Feature or polygon.triangleFeature):
                self.skipFeaturesAction.skipFeatures(polygon, True, manager)
    
    def detectFeatures(self, polygon):
        """
        Detects patterns for small features
        """
        
        polygon.prepareVectorsByIndex()

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
    
    def debugSetFeatureSymbols(self, polygon, sequence):
        for vector,symbol in zip(polygon.getVectors(), sequence):
            vector.featureSymbol = symbol