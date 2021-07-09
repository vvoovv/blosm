import re
from defs.building import BldgPolygonFeature, curvyLengthFactor, lengthThreshold, \
    sin_me, sin_hi
from building.feature import Feature


class FeatureDetection:
    
    # a sequence of four or more 'C' matches as curvy sequence
    curvedPattern = re.compile(r"(C){4,}")
    
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
            1 for vector in polygon.getVectors() if vector.hasAnglesForCurvedFeature()
        )
        if not numLowAngles:
            return

        # Calculate a length threshold as a mean of the vector lengths among the vectors
        # that satisfy the condition <vector.hasAnglesForCurvedFeature()>
        curvyLengthThresh = curvyLengthFactor / numLowAngles *\
            sum(
                vector.length for vector in polygon.getVectors()\
                if vector.hasAnglesForCurvedFeature()
            )

        # Feature character sequence: edges with angle between 5° and 30° on either end 
        # and a length below <curvyLengthThresh> get a 'C', else a '0'
        sequence = ''.join(
            'C' if vector.length<curvyLengthThresh and vector.hasAnglesForCurvedFeature() else '0' \
            for vector in polygon.getVectors()
        )
        
        sequenceLength = len(sequence)
        sequence = sequence+sequence # allow cyclic pattern
        
        self.matchPattern(
            sequence, sequenceLength,
            FeatureDetection.curvedPattern,
            BldgPolygonFeature.curved,
            polygon, manager
        )
    
    def detectSmallFeatures(self, polygon, manager):
        """
        Detects small patterns (rectangular and triangular).
        """
        
        # Long edges (>=lengthThresh):
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
        
        sequence = ''.join(
            (
                'L' if ( vector.sin > sin_hi and vector.next.sin > sin_hi ) else (
                    'R' if ( vector.sin < -sin_hi and vector.next.sin < -sin_hi ) else 'O'
                )
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
        
        self.matchPattern(
            sequence, sequenceLength,
            FeatureDetection.convexRectPattern,
            BldgPolygonFeature.rectangle,
            polygon, manager
        )
        sequence = re.sub(FeatureDetection.convexRectPattern, lambda m: '1' * len(m.group()), sequence)

        self.matchPattern(
            sequence, sequenceLength,
            FeatureDetection.convexTriPattern,
            BldgPolygonFeature.triangle,
            polygon, manager
        )
        sequence = re.sub(FeatureDetection.convexTriPattern, lambda m: '2' * len(m.group()), sequence)
        
        self.matchPattern(
            sequence, sequenceLength,
            FeatureDetection.concaveRectPattern,
            BldgPolygonFeature.rectangle,
            polygon, manager
        )
        sequence = re.sub(FeatureDetection.concaveRectPattern, lambda m: '3' * len(m.group()), sequence)
        
        self.matchPattern(
            sequence, sequenceLength,
            FeatureDetection.concaveTriPattern,
            BldgPolygonFeature.triangle,
            polygon, manager
        )
    
    def matchPattern(self, sequence, sequenceLength, pattern, featureId, polygon, manager):
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
        