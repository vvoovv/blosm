from defs.facade_classification import *


class FacadeClassification:
    
    def __init__(self):
        pass
    
    def do(self, manager):
            
        for building in manager.buildings:
            # find maximum building dimension
            maxX, maxY = -999999., -999999.
            minX, minY = 999999., 999999.
            for vector in building.polygon.getVectors():
                v1 = vector.v1
                maxX = max(v1[0],maxX)
                maxY = max(v1[1],maxY)
                minX = min(v1[0],minX)
                minY = min(v1[1],minY)
            maxBuildDimension = max(maxX-minX,maxY-minY)
          
            # The <edge> could have been already visited earlier if it is the shared one
            for vector in building.polygon.getVectors():
                edge = vector.edge
                visInfo = edge.visInfo
                if not edge.cl and edge.hasSharedBldgVectors():
                    edge.cl = FacadeClass.shared
                elif hasattr(visInfo,'distance'):
                    if visInfo.distance/maxBuildDimension > maxDistanceRatio:
                        visInfo.value = 0.

            # Find front facades
            self.classifyFrontFacades(building)

            # Find side facades
            self.classifySideFacades(building)

            # Mark the remaining building edges as back
            for vector in building.polygon.getVectors():
                if not vector.edge.cl:
                    vector.edge.cl = FacadeClass.back

    def classifyFrontFacades(self, building):
        maxVisibility = 0.
        maxEdge = None
        accepted_level = 0

        for way_level in range(1, MaxWayLevel+1): # way-level corresponds to way-categories C1, C2, C3
            for vector in building.polygon.getVectors():
                edge = vector.edge
                visInfo = edge.visInfo
                if visInfo.value and \
                        ( visInfo.numMostlyPerpWaySegments >= 2 or visInfo.mostlyParallelToWaySegment() ) and\
                        WayLevel[visInfo.waySegment.way.category] == way_level:
                    if edge.cl in CrossedFacades:
                        # deadend becomes front, while passage remains
                        accepted_level = way_level
                        if edge.cl == FacadeClass.deadend:
                            edge.cl = FacadeClass.front
                        continue
                    if not edge.cl:
                        if visInfo.value > maxVisibility:
                            maxVisibility, maxEdge = (visInfo.value, edge)
                        # For each building edge satisfying the conditions 
                        #   1) the category of the stored way is equal to a category from the category set AND 
                        #   2) visibility > 0.75 AND 
                        #   3) dy < dx*VisibilityAngleFactor:
                        # Mark the building edge as front
                        if visInfo.value >= FrontFacadeVisibility:
                            edge.cl = FacadeClass.front
                            accepted_level = way_level
            # If there is at least one building edge satisfying the above condition:
            #   Break the cycle of WayClasses C1, C2, C2
            if accepted_level:
                # only those facades at dead-end of a way are accepted, that are within the accepted way-level
                # the rest is set as unknown for frther processing
                for vector in building.polygon.getVectors():
                    edge = vector.edge
                    if edge.cl == FacadeClass.deadend:
                        if WayLevel[edge.visInfo.waySegment.way.category] > accepted_level:
                            edge.cl = FacadeClass.unknown
                        else:
                            edge.cl = FacadeClass.front
                break

        if not accepted_level:
            # Else: Find the building edge with the maximum visibility satisfying the conditions 
            #   1) the category of the stored way is equal to a category from the category set AND 
            #   2) dy < dx: (already checked in facade_visibility.py)
            # Mark the building edge as front
            if maxVisibility:
                maxEdge.cl = FacadeClass.front       
            # If no front building edge was found, mark one as front
            else:
                self.frontOfInvisibleBuilding(building)

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
