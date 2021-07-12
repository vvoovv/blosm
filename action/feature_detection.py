import re
from defs.building import BldgPolygonFeature, curvyLengthFactor, lengthThreshold, \
    longFeatureFactor, sin_lo, sin_me, sin_hi
from building.feature import Feature


def hasAnglesForCurvedFeature(vector):
    """
    Checks if <vector.edge> has an angle between 5° and 30° on both ends of <vector.edge>
    """
    return sin_lo<abs(vector.sin)<sin_me and sin_lo<abs(vector.next.sin)<sin_me


class FeatureDetection:
    
    # a sequence of two or more 'C' matches as curvy sequence.
    # possibly with ends 'S' (start) or 'E' (end)
    curvedPattern = re.compile(r"S?(C){2,}E?")
    
    # convex rectangular features
    convexRectPattern = re.compile(r"(>[L|l]<)")
    
    # convex triangular features
    # triangle = r">(>|<|l){1,}"
    # left_triangle = r"(l<)" # special case for triangular part of rectangle
    # convexTriPattern = re.compile(triangle + r"|" + left_triangle)
    convexTriPattern = re.compile(r">(>|<|l){1,}" + r"|" + r"(l<)")
    
    # concave rectangular features
    concaveRectPattern = re.compile(r"([\-|<][R,r][\+|>])")
    
    # concave triangular features
    concaveTriPattern = re.compile(r"<(>|<|r){1,}")
    
    def __init__(self):
        pass 

    def do(self, manager):

        for building in manager.buildings:
            polygon = building.polygon
            
            polygon.prepareVectorsByIndex()

            # detect curved features
            self.detectCurvedFeatures(polygon, manager)
            self.detectSmallFeatures(polygon, manager)
    
    def detectCurvedFeatures(self, polygon, manager):        
        numLowAngles = sum(
            1 for vector in polygon.getVectors() if hasAnglesForCurvedFeature(vector)
        )
        if not numLowAngles:
            return

        # Calculate a length threshold as a mean of the vector lengths among the vectors
        # that satisfy the condition <hasAnglesForCurvedFeature(vector)>
        curvyLengthThresh = curvyLengthFactor / numLowAngles *\
            sum(
                vector.length for vector in polygon.getVectors()\
                if hasAnglesForCurvedFeature(vector)
            )

        # Feature character sequence: edges with angle between 5° and 30° on either end 
        # and a length below <curvyLengthThresh> get a 'C', else a '0'
        sequence = ''.join(
            '0' if vector.length>curvyLengthThresh else ( \
                'C' if hasAnglesForCurvedFeature(vector) else ( \
                    'S' if abs(vector.sin)>sin_me and sin_lo<abs(vector.next.sin)<sin_me else ( \
                        'E' if sin_lo<abs(vector.sin)<sin_me and abs(vector.next.sin)>sin_me else '0'
                    )
                )
            )
            for vector in polygon.getVectors()
        )
        
        sequenceLength = len(sequence)
        sequence = sequence+sequence # allow cyclic pattern
        
        self.matchPattern(
            sequence, sequenceLength,
            FeatureDetection.curvedPattern,
            BldgPolygonFeature.curved,
            polygon, manager, ''
        )
    
    def detectSmallFeatures(self, polygon, manager):
        """
        Detects small patterns (rectangular and triangular).
        """
        
        # Long edges (>=lengthThresh and <maxLengthThreshold):
        #       'L': sharp left at both ends
        #       'R': sharp right at both ends
        #       'O': other long edge
        # Short edges:
        #       'l': medium left at both ends
        #       'r': medium right at both ends
        #       '>': alternate medium angle, starting to right
        #       '<': alternate medium angle, starting to left
        #       'o': other short edge
        
        # a primitive filter to avoid spiky edge detection for all buildings
        numLongEdges = sum(
            1 for vector in polygon.getVectors() if vector.length >= lengthThreshold
        )
        numShortEdges = sum(
            1 for vector in polygon.getVectors() if vector.length < lengthThreshold
        )
        if not ( (numLongEdges and numShortEdges > 2) or numShortEdges > 5 ):
            return
        
        # compute length limit <maxLengthThreshold> for long rectangular features
        maxLengthThreshold = longFeatureFactor * polygon.dimension

        sequence = ''.join(
            (
                (
                    'L' if ( vector.sin > sin_hi and vector.next.sin > sin_hi ) else (
                        'R' if ( vector.sin < -sin_hi and vector.next.sin < -sin_hi ) else 'O'
                    )
                ) if vector.length < maxLengthThreshold else 'O'
            ) \
            if vector.length >= lengthThreshold else (
                'l' if (vector.sin > sin_me and vector.next.sin > sin_me) else (
                    'r' if (vector.sin < -sin_me and vector.next.sin < -sin_me) else (
                        '>' if ( vector.sin < -sin_me and vector.next.sin > sin_me ) else (
                            '<' if (vector.sin > sin_me and vector.next.sin < -sin_me) else 'o'
                        )
                    )
                )
            ) \
            for vector in polygon.getVectors()
        )

        sequenceLength = len(sequence)
        sequence = sequence+sequence # allow cyclic pattern
        
        sequence = self.matchPattern(
            sequence, sequenceLength,
            FeatureDetection.convexRectPattern,
            BldgPolygonFeature.rectangle,
            polygon, manager, '1'
        )

        sequence = self.matchPattern(
            sequence, sequenceLength,
            FeatureDetection.convexTriPattern,
            BldgPolygonFeature.triangle,
            polygon, manager, '2'
        )
        
        sequence = self.matchPattern(
            sequence, sequenceLength,
            FeatureDetection.concaveRectPattern,
            BldgPolygonFeature.rectangle,
            polygon, manager, '3'
        )
        
        sequence = self.matchPattern(
            sequence, sequenceLength,
            FeatureDetection.concaveTriPattern,
            BldgPolygonFeature.triangle,
            polygon, manager, ''
        )
    
    def matchPattern(self, sequence, sequenceLength, pattern, featureId, polygon, manager,  subChar):
        matches = [r for r in pattern.finditer(sequence)]
        if matches:
            for featureSeg in matches:
                s = featureSeg.span()
                if 0 <= s[0] < sequenceLength:
                    Feature(
                        featureId,
                        polygon.getVectorByIndex(s[0]), # start vector
                        polygon.getVectorByIndex( (s[1]-1) % sequenceLength ), # end vector
                        False, # skip
                        manager
                    )
                    if subChar and s[1] >= sequenceLength:  # case of cyclic pattern
                        sequence = subChar*(s[1]-sequenceLength) + sequence[(s[1]-sequenceLength):]
            if subChar:
                sequence = re.sub(pattern, lambda m: subChar * len(m.group()), sequence)
        return sequence