

class FacadeClass:
    unknown = 0
    front = 1
    side = 2
    back = 3
    shared = 4
    crossed = 5


class FacadeClassification:
    
    def __init__(self):
        pass
    
    def do(self, manager):
        
        for building in manager.buildings:
            for vector in building.polygon.getVectors():
                edge = vector.edge
                # The <edge> could have been already visited earlier if it is the shared one
                if not edge.cl:
                    if edge.hasSharedBldgVectors():
                        edge.cl = FacadeClass.shared
                    else:
                        pass
                        # classify edge here