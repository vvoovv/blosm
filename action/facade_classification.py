

class FacadeClass:
    unknown = 0
    front = 1
    side = 2
    back = 3
    shared = 4
    passage = 5

# way categories
C1 = ['primary', 'secondary', 'tertiary', 'residential', 'pedestrian', 'unclassified']
C2 = ['living_street']
C3 = ['service']
cAll = C1+C2+C3

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
            self.classifyFrontFacades(building, cAll)

            # Find side facades
            self.classifySideFacades(building)

            # Mark the remaining building edges as back
            for vector in building.polygon.getVectors():
                if not vector.edge.cl:
                    vector.edge.cl = FacadeClass.back

    def classifyFrontFacades(self, building, wayC):
        maxVisibility = 0.
        maxEdge = None
        hasFrontFacade = False
        for vector in building.polygon.getVectors():
            edge = vector.edge
            if edge.cl == FacadeClass.front: # set as crossing facade
                hasFrontFacade = True
                continue
            visInfo = edge.visInfo
            if not edge.cl and visInfo.value > 0. and visInfo.waySegment.way.category in wayC:
                if visInfo.value > maxVisibility:
                    maxVisibility, maxEdge = (visInfo.value, edge)
                # For each building edge satisfying the conditions 
                #   1) the category of the stored way is equal to a category from the category set AND 
                #   2) visibility > 0.75 AND 
                #   3) dy <= dx:
                # Mark the building edge as front
                if visInfo.value >= 0.5 and visInfo.dx >= visInfo.dy:
                    edge.cl = FacadeClass.front
                    hasFrontFacade = True

        if not hasFrontFacade:
            # Else: Find the building edge with the maximum visibility satisfying the conditions 
            #   1) the category of the stored way is equal to a category from the category set AND 
            #   2) dy < dx:
            # Mark the building edge as front
            if maxVisibility:
                maxEdge.cl = FacadeClass.front
        
            # If no front building edge was found, mark one as front
            else:
                self.classifyInvisibleBuilding(building)

    def classifyInvisibleBuilding(self, building):
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
        frontLikeFacades = [FacadeClass.front, FacadeClass.passage]
        for vector in building.polygon.getVectors():
            edge = vector.edge
            if not edge.cl:
                if vector.prev.edge.cl in frontLikeFacades or vector.next.edge.cl in frontLikeFacades:
                    edge.cl = FacadeClass.side
