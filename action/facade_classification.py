from math import tan, pi

from way import Category, facadeVisibilityWayCategories

class FacadeClass:
    unknown = 0
    front = 1
    side = 2
    back = 3
    shared = 4
    passage = 5
    deadend = 6

CrossedFacades = [FacadeClass.deadend, FacadeClass.passage]
FrontLikeFacades = [FacadeClass.front, FacadeClass.passage]

FrontFacadeVisibility = 0.75                        # visibility required to classify as front facade
VisibilityAngle = 50                                # maximum angle in ° between way-segment and facade to be accepted as visible
VisibilityAngleFact = tan(pi*VisibilityAngle/180.)  # Factor used in angle condition: VisibilityAngleFact*dx > dy

WayLevel = dict((category,1) for category in facadeVisibilityWayCategories)
WayLevel[Category.service] = 2
MaxWayLevel = 2

class FacadeClassification:
    
    def __init__(self):
        pass
    
    def do(self, manager):
            
        for building in manager.buildings:
            # The <edge> could have been already visited earlier if it is the shared one
            for vector in building.polygon.getVectors():
                edge = vector.edge
                if not edge.cl and edge.hasSharedBldgVectors():
                    edge.cl = FacadeClass.shared

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

        for way_level in range(1,MaxWayLevel+1): # way-level corresponds to way-categories C1, C2, C3
            for vector in building.polygon.getVectors():
                edge = vector.edge
                visInfo = edge.visInfo
                if visInfo.value and WayLevel[visInfo.waySegment.way.category] == way_level:
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
                        #   3) dy < dx*VisibilityAngleFact:  (already checked in facade_visibility.py)
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
            x = vector.vector[0]
            y = vector.vector[1]
            lengthSquared = x*x+y*y
            if lengthSquared > longest:
                longest = lengthSquared
                longestEdge = vector.edge
        longestEdge.cl = FacadeClass.front

    def classifySideFacades(self, building):    
        for vector in building.polygon.getVectors():
            edge = vector.edge
            if not edge.cl:
                if vector.prev.edge.cl in FrontLikeFacades or vector.next.edge.cl in FrontLikeFacades:
                    edge.cl = FacadeClass.side
