import re

from .feature_detection import FeatureDetection

from defs.building import curvedLengthFactor, sin_lo, sin_me

from building.feature import Curved


def hasAnglesForCurvedFeature(vector):
    """
    Checks if <vector.edge> has an angle between 5° and 30° on both ends of <vector.edge>
    """
    return sin_lo<abs(vector.sin)<sin_me and sin_lo<abs(vector.next.sin)<sin_me


class CurvedFeatures(FeatureDetection):
    
    # a sequence of two or more 'C' matches as curvy sequence.
    # possibly with ends 'S' (start) or 'E' (end)
    curvedPattern = re.compile(r"S?(C){3,}")
    
    def do(self, manager):
        for building in manager.buildings:
            # detect curved features
            self.detectCurvedFeatures(building.polygon)
    
    def detectCurvedFeatures(self, polygon):
        numLowAngles = sum(
            1 for vector in polygon.getVectorsAll() if hasAnglesForCurvedFeature(vector)
        )
        if not numLowAngles:
            # debug
            for vector in polygon.getVectorsAll():
                vector.featureSymbol = '0'
            return

        # Calculate a length threshold as a mean of the vector lengths among the vectors
        # that satisfy the condition <hasAnglesForCurvedFeature(vector)>
        curvedLengthThreshold = curvedLengthFactor / numLowAngles *\
            sum(
                vector.length for vector in polygon.getVectorsAll()\
                if hasAnglesForCurvedFeature(vector)
            )

        # Feature character sequence: edges with angle between 5° and 30° on either end 
        # and a length below <curvyLengthThresh> get a 'C', else a '0'
        # Originally the conditions below had:
        # ... sin_lo<abs(vector.sin)<sin_me ...
        # ... sin_lo<abs(vector.next.sin)<sin_me ...
        sequence = ''.join(
            '0' if vector.length>curvedLengthThreshold else ( \
                'C' if 0.<abs(vector.sin)<sin_me else ( \
                    'S' if 0.<abs(vector.next.sin)<sin_me else 'K'
                )
            )
            for vector in polygon.getVectorsAll()
        )
        
        # debug
        self.debugSetFeatureSymbols(polygon, sequence)

        sequenceLength = len(sequence)
        sequence = sequence+sequence # allow cyclic pattern
        
        matches = [r for r in FeatureDetection.curvedPattern.finditer(sequence)]
        numMatches = len(matches)
        if matches:
            # Check if need to skip the first match. It can be the case that
            # a curved feature is detected at the very beginning of <sequence>,
            # but actually the curved feature starts before the first character
            firstMatchIndex = int(
                numMatches > 1 and not matches[0].span()[0] and matches[-1].span()[1] >= sequenceLength
            )
            for matchIndex in range(firstMatchIndex, numMatches):
                matchSpanIndices = matches[matchIndex].span()
                if 0 <= matchSpanIndices[0] < sequenceLength:
                    startIndex = matchSpanIndices[0]
                    endIndex = (matchSpanIndices[1]-1) % sequenceLength
                    Curved(
                        polygon.vectors[
                            polygon.numEdges - 1 - startIndex if polygon.reversed else startIndex
                        ], # start vector
                        polygon.vectors[
                            polygon.numEdges - 1 - endIndex if polygon.reversed else endIndex
                        ] # end vector
                    )