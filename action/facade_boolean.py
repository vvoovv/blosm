

class FacadeBoolean:
    
    def __init__(self):
        pass
    
    def do(self, footprint):
        building = footprint.building
        polygon = footprint.polygon
        
        for vector in polygon.getVectors():
            edge = vector.edge
            bldgVectors = edge.vectors
            
            if bldgVectors:
                if len(bldgVectors) == 1:
                    # No neighbor building and no neighbor building parts
                    # All vectors (the building footprint and building part) that share the <edge> and
                    # have the same direction
                    if building.parts:
                        self.processVectorsSameDir(vector)
                    # If there are no building parts, then there is no need to
                    # process intersecting or overlapping facades
                else:
                    # we have here: len(edge.vectors) == 2
                    
                    if building.parts:
                        self.processVectorsSameDir(vector)
                    
                    neighborBldgVector = bldgVectors[1] if bldgVectors[0].polygon is polygon else bldgVectors[0]
                    neighborBldgParts = neighborBldgVector.polygon.building.parts
                    if neighborBldgParts and neighborBldgVector.facade:
                        # added <neighborBldgVector.facade> in the condition above as a hack
                        self.processVectorsOppDir(vector, neighborBldgVector)
                    
                    if not building.parts and not neighborBldgParts and not neighborBldgVector.facade.processed:
                        # No vectors that define building parts. There are only <bldgVectors>
                        # <vector> defines the footprint of <building>
                        vector.facade.geometry.subtract(vector.facade, neighborBldgVector.facade)
            else:
                # <edge> is located inside the footprint of <building>
                self.processVectorsSameDir(vector)
                
                self.processVectorsOppDir(vector, None)

            # mark <vector> as processed
            vector.facade.processed = True
    
    def processVectorsSameDir(self, vector):
        vectors = vector.edge.partVectors12 if vector.direct else vector.edge.partVectors21
        if not vectors:
            return
        for _vector in vectors:
            if not vector is _vector and not _vector.facade.processed:
                vector.facade.geometry.join(vector.facade, _vector.facade)
    
    def processVectorsOppDir(self, vector, neighborBldgVector):
        vectors = vector.edge.partVectors21 if vector.direct else vector.edge.partVectors12
        # Check if the related building has been already processed. It's enough to check only one facade.
        # It means that all related facades have been already processed
        if not vectors or vectors[0].facade.processed:
            return
        
        geometry = vector.facade.geometry
        for _vector in vectors:
            vector.facade.geometry.subtract(vector.facade, _vector.facade)
        
        if neighborBldgVector and neighborBldgVector.polygon.building.alsoPart:
            geometry.subtract(vector.facade, neighborBldgVector.facade)