import re
from defs.building import BldgPolygonFeature, curvyLengthFactor
from building.feature import Feature


class FeatureDetection:
    
    # a sequence of four or more 'C' matches as curvy sequence
    curvedFeaturePattern = re.compile(r"(C){4,}")

    def __init__(self):
        pass 

    def do(self, manager):

        for building in manager.buildings:
            polygon = building.polygon
            
            polygon.prepareVectorsByIndex()

            # detect curved features
            self.detectCurvedFeatures(polygon, manager)

            # a primitive filter to avoid spiky edge detection for all buildings
            #nLongEdges = np.where( vectorData[:,0]>=lengthThresh  )[0].shape[0]
            #nShortEdges = np.where( vectorData[:,0]<lengthThresh )[0].shape[0]
            #smallFeatures = None
            #if (nLongEdges and nShortEdges > 2) or nShortEdges > 5:
            #    smallFeatures = self.detectsmallFeatures(vectorData,vectors)
    
    def detectCurvedFeatures (self, polygon, manager):
        numLowAngles = sum(
            1 for vector in polygon.getVectors() if vector.hasAnglesForCurvedFeature()
        )
        if not numLowAngles:
            return None

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
        
        matches = [
            c for c in FeatureDetection.curvedFeaturePattern.finditer(
                sequence+sequence # adjacent sequence for circularity
            )
        ]
        if matches:
            N = len(sequence)
            for curvySeg in matches:
                s = curvySeg.span()
                if 0 <= s[0] < N:
                    Feature(
                        BldgPolygonFeature.curved,
                        polygon.getVectorByIndex(s[0]), # start vector
                        polygon.getVectorByIndex( (s[1]-1)%N ), # end vector
                        False, # skip
                        manager
                    )

    # Detects small patterns (rectangular and triangular).
    # Returns:
    #   (firstEdge,lastEdge,patternClass) : Tuple of first edge of sequence and last edge of sequence and the pattern class.
    #                                       If firstEdge is lastEdge, the whole building polygon is a curvy sequence 
    #   None                              : No patterns found 
    def detectsmallFeatures(self,vectorData,vectors):
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
        length = vectorData[:,0]
        sineStart = vectorData[:,1]
        sineEnd = vectorData[:,2]

        sequence =  "".join(  np.where( (length>=lengthThresh), \
                                ( np.where( ((sineStart>sin_hi) & (sineEnd>sin_hi)), 'L', \
                                  np.where( ((sineStart<-sin_hi) & (sineEnd<-sin_hi)), 'R', 'O') )  \
                                ), \
                                ( np.where( ((sineStart>sin_me) & (sineEnd>sin_me)), 'l', \
                                  np.where( ((sineStart<-sin_me) & (sineEnd<-sin_me)), 'r',  \
                                  np.where( (sineStart<-sin_me)&(sineEnd>sin_me), '>', \
                                  np.where( (sineStart>sin_me)&(sineEnd<-sin_me), '<', 'o') ) ) )\
                                ) )
                            )

        N = len(sequence)
        sequence = sequence+sequence # allow cyclic pattern
        smallFeatures = []

        # convex rectangular features
        pattern = re.compile(r"(>[L|l]<)")
        matches = [r for r in pattern.finditer(sequence)]
        if matches:
            for featureSeg in matches:
                s = featureSeg.span()
                if s[0] < N and s[0] >= 0:
                    smallFeatures.append( ( vectors[s[0]], vectors[(s[1]-1)%N], FeatureClass.rectangular ) )
        sequence = re.sub(pattern, lambda m: ('1' * len(m.group())), sequence)

        # convex triangular features
        triangle = r">(>|<|l){1,}"
        left_triangle = r"(l<)" # special case for triangular part of rectangle
        pattern = re.compile(triangle + r"|" + left_triangle)
        matches = [r for r in pattern.finditer(sequence)]
        if matches:
            for featureSeg in matches:
                s = featureSeg.span()
                if s[0] < N and s[0] >= 0:
                    smallFeatures.append( ( vectors[s[0]], vectors[(s[1]-1)%N], FeatureClass.triangular ) )
        sequence = re.sub(pattern, lambda m: ('2' * len(m.group())), sequence)

        # concave rectangular features
        pattern = re.compile(r"([\-|<][R,r][\+|>])")
        matches = [r for r in pattern.finditer(sequence)]
        if matches:
            for featureSeg in matches:
                s = featureSeg.span()
                if s[0] < N and s[0] >= 0:
                    smallFeatures.append( ( vectors[s[0]], vectors[(s[1]-1)%N], FeatureClass.rectangular ) )
        sequence = re.sub(pattern, lambda m: ('3' * len(m.group())), sequence)

        # concave triangular features
        pattern = re.compile(r"<(>|<|r){1,}")
        matches = [r for r in pattern.finditer(sequence)]
        if matches:
            for featureSeg in matches:
                s = featureSeg.span()
                if s[0] < N and s[0] >= 0:
                    smallFeatures.append( ( vectors[s[0]], vectors[(s[1]-1)%N], FeatureClass.triangular ) )
        sequence = re.sub(pattern, lambda m: ('4' * len(m.group())), sequence)

        if smallFeatures:
            return smallFeatures
        else:
            return None
