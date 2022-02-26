from mathutils import Vector
from defs.facade_classification import *


class FacadeClassification:
    
    def __init__(self, unskipFeaturesAction=None):
        self.unskipFeaturesAction = unskipFeaturesAction
    
    def do(self, manager):
        
        unskipFeaturesAction = self.unskipFeaturesAction
        
        for building in manager.buildings:

            # The <edge> could have been already visited earlier if it is the shared one
            for vector in building.polygon.getVectors():
                edge = vector.edge
                if not edge.cl and edge.hasSharedBldgVectors():
                    edge.cl = FacadeClass.shared

            # Find front facades
            accepted_level = self.classifyFrontFacades(building)
            
            # process crossed facades
            for vector in building.polygon.getVectors():
                edge = vector.edge
                if edge.cl in CrossedFacades:
                    if WayLevel[edge.visInfo.waySegment.way.category] <= accepted_level:
                        edge.cl = FacadeClass.front
                        edge.visInfo.value = 1.
                    else:
                        edge.cl = FacadeClass.unknown

            # Find side facades
            self.classifySideFacades(building)

            # Mark the remaining building edges as back
            for vector in building.polygon.getVectors():
                if not vector.edge.cl:
                    vector.edge.cl = FacadeClass.back
            
            if unskipFeaturesAction:
                unskipFeaturesAction.unskipFeatures(building.polygon)

    def classifyFrontFacades(self, building):
        maxSight = 0.
        maxEdge = None
        accepted_level = 0

        for way_level in range(1, MaxWayLevel+1): # way-level corresponds to way-categories C1, C2, C3
            for vector in building.polygon.getVectors():
                edge = vector.edge
                visInfo = edge.visInfo
                if visInfo.value and not edge.cl == FacadeClass.shared and \
                        WayLevel[visInfo.waySegment.way.category] == way_level:
                    edgeSight = visInfo.value * visInfo.dx/(visInfo.dx+visInfo.dy) * visInfo.waySegment.avgDist/visInfo.distance
                    # compute maximum sight for later use
                    if edgeSight > maxSight:
                        maxSight, maxEdge = edgeSight, edge
                    # For each building edge satisfying the conditions 
                    #   1) the category of the stored way is equal to a category from the category set AND 
                    #   2) the edge sight is larger than the parameter FrontFacadeSight 
                    #   3) dy < dx*VisibilityAngleFactor:
                    # Mark the building edge as front
                    if edgeSight >= FrontFacadeSight:
                        edge.cl = FacadeClass.front
                        accepted_level = way_level
                elif edge.cl in CrossedFacades and WayLevel[visInfo.waySegment.way.category] == way_level:
                    accepted_level = way_level
                   

            # If there is at least one building edge satisfying the above condition:
            #   do some post-processing and then
            #   break the cycle of WayClasses C1, C2
            if accepted_level:
                # process eventuall facet facades at corners 
                for vector in building.polygon.getVectors():
                    edge = vector.edge
                    visInfo = edge.visInfo
                    # done only for unclassified facades between two front facades
                    # where there is an angle between their way-segments
                    if visInfo.value and not edge.cl == FacadeClass.shared and \
                        vector.prev.edge.cl == FacadeClass.front and \
                        vector.next.edge.cl == FacadeClass.front:
                            uNext = vector.next.edge.visInfo.waySegment.unitVector
                            uPrev = vector.prev.edge.visInfo.waySegment.unitVector
                            # dot product of unit vectors is cosine of angle 
                            # between these way-segments
                            if abs(uPrev @ uNext) < CornerFacadeWayAngle:
                                edge.cl = FacadeClass.front
                                accepted_level = way_level

                break

        if not accepted_level:
            # Else: Find the building edge with the maximum visibility satisfying the conditions 
            #   1) the category of the stored way is equal to a category from the category set AND 
            #   2) dy < dx: (already checked in facade_visibility.py)
            # Mark the building edge as front
            if maxSight:
                maxEdge.cl = FacadeClass.front       
            # If no front building edge was found, mark one as front
            else:
                self.frontOfInvisibleBuilding(building)

        return accepted_level if accepted_level else MaxWayLevel+1
 
    def frontOfInvisibleBuilding(self, building): 
        # If no front building edge was found, mark one as front (e.g. the longest one)
        longest = 0.
        longestEdge = None
        for vector in building.polygon.getVectors():
            if vector.edge.cl != FacadeClass.shared:
                x = vector.vector[0]
                y = vector.vector[1]
                lengthSquared = x*x+y*y
                if lengthSquared > longest:
                    longest = lengthSquared
                    longestEdge = vector.edge
        if longestEdge:
            longestEdge.cl = FacadeClass.front

    def classifySideFacades(self, building):    
        for vector in building.polygon.getVectors():
            edge = vector.edge
            if not edge.cl:
                if vector.prev.edge.cl in FrontLikeFacades or vector.next.edge.cl in FrontLikeFacades:
                    edge.cl = FacadeClass.side


def _inheritFacadeClass(saFeature):
    """
    Inherit a facade class from a vector with a skipped straight angle feature
    to the vectors that form that straight angle feature
    """
    cl = saFeature.startVector.edge.cl
    # process the starting edge that was skipped
    saFeature.startEdge[0].cl = cl
    # process the rest of edges that form <saFeature>
    _vector = saFeature.startNextVector
    while True:
        _vector.edge.cl = cl
        if _vector is saFeature.endVector:
            break
        _vector = _vector.next


class FacadeClassificationPart:
    
    def do(self, manager):
        # process the building parts for each building
        for building in manager.buildings:
            if building.parts:
                for part in building.parts:
                    self.classify(part)
    
    def classify(self, part):
        bldgFacadeClassesInherited = False
        bldgPolygon = part.polygon.building.polygon
        # assign facade class to the edges of the building part
        for vector in part.polygon.getVectors():
            if not vector.edge.cl:
                if not bldgFacadeClassesInherited:
                    saFeature = bldgPolygon.saFeature
                    if saFeature:
                        # Inherit the facade class from a vector with the straight angle feature
                        # to the vectors that form the straight angle feature.
                        while True:
                            # iterate through all straight angle features
                            if saFeature.skipped:
                                # <saFeature> doesn't contain nested straight angle features
                                # represented by the class <StraightAnglePart>.
                                _inheritFacadeClass(saFeature)
                            else:
                                # <saFeature> has one or more nested straight angle features represented
                                # by the class <StraightAnglePart>
                                
                                if saFeature.parent:
                                    # If <saFeature.parent> isn't equal to <None>, it means that
                                    # <saFeature.startVector> is the resulting vector after skipping
                                    # some vectors because they form a straight angle feature represented
                                    # by the class <StraightAnglePart>
                                    _inheritFacadeClass(saFeature.parent)
                                # process the rest of vectors that form <saFeature>
                                _vector = saFeature.startVector.next
                                while True:
                                    if _vector.feature:
                                        # It can be only nested straight angle feature represented
                                        # by the class <StraightAnglePart>. For that feature we can only have
                                        # the attribute <skipped> equal to <True>.
                                        _inheritFacadeClass(_vector.feature)
                                    if _vector is saFeature.endVector:
                                        break
                                    _vector = _vector.next
                            
                            if saFeature.prev:
                                saFeature = saFeature.prev
                            else:
                                break
                    
                    bldgFacadeClassesInherited = True
                
                if not vector.edge.cl:
                    # We try to estimate the facade class for <vector.edge> by projecting the vectors
                    # of the building footprint on it and checking if the projection overlaps with <vector>
                    # We ingore complex things like concave parts of the building footprint.
                    
                    # an overlap of a projected vector of the building footprint with <vector>
                    maxOverlap = 0.
                    maxOverlapClass = FacadeClass.unknown
                    unitVector = vector.unitVector
                    # a normal to <unitVector> pointing outwards
                    nVector = Vector((unitVector[1], -unitVector[0]))
                    x1, x2, y = vector.v1.dot(unitVector), vector.v2.dot(unitVector), vector.v1.dot(nVector)
                    for bldgVector in bldgPolygon.getVectors():
                        _bldgVector, _v1, _v2 = bldgVector.vector, bldgVector.v1, bldgVector.v2
                        # The conditions < _bldgVector.v1.dot(nVector) >=y > and < _bldgVector.v2.dot(nVector) >= y > mean
                        # that the vector <_bldgVector> must be located above <unitVector>
                        # along the axis <nVector> 
                        if unitVector.dot(_bldgVector) > 0. and _v1.dot(nVector) >= y or _v2.dot(nVector) >= y:
                            _x1, _x2 = _v1.dot(unitVector), _v2.dot(unitVector)
                            if _x1 < x2 and x1 < _x2:
                                if x2 < _x2:
                                    overlap = x2 - _x1
                                elif _x1 < x1:
                                    overlap = _x2 - x1
                                else:
                                    overlap = _x2 - _x1
                                
                                if overlap > maxOverlap:
                                    maxOverlap = overlap
                                    maxOverlapClass = bldgVector.edge.cl
                    
                    vector.edge.cl = maxOverlapClass