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
