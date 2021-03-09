

class FacadeVisibility:
    
    def __init__(self):
        self.app = None
        self.kdTree = None
        self.bldgVerts = None
        self.vertIndexToBldg = []
    
    def do(self, manager):
        # check if have a way manager
        if not self.app.managersById["ways"]:
            return
        
        buildings = manager.buildings
        
        # Create an instance of <util.polygon.Polygon> for each <building>,
        # remove straight angles for them and calculate the total number of vertices
        for building in buildings:
            if not building.polygon:
                building.initPolygon(manager.data)
                if not building.polygon:
                    continue
            if not building.visibility:
                building.initVisibility()
        
        # the total number of vertices
        totalNumVerts = sum(building.polygon.n for building in buildings if building.polygon)
        # create mapping between the index of the vertex and index of the building in <buildings>
        self.vertIndexToBldg.extend(
            bldgIndex for bldgIndex, building in enumerate(buildings) if building.polygon for _ in range(building.polygon.n)
        )
        
        self.createKdTree(buildings, totalNumVerts)
        
        self.calculateFacadeVisibility(buildings)
    
    def cleanup(self):
        self.kdTree = None
        self.bldgVerts = None
        self.vertIndexToBldg.clear()

    def calculateFacadeVisibility(self, buildings):
        for way in self.app.managersById["ways"].getAllWays():
            # calculate facade visibility here
            pass


class FacadeVisibilityBlender(FacadeVisibility):
    
    def createKdTree(self, buildings, totalNumVerts):
        from mathutils.kdtree import KDTree
        kdTree = KDTree(totalNumVerts)
        
        # fill in <self.bldgVerts>
        self.bldgVerts = tuple( vert for building in buildings if building.polygon for vert in building.polygon.verts )
        
        for index, vert in enumerate(self.bldgVerts):
            kdTree.insert(vert, index)
        kdTree.balance()
        self.kdTree = kdTree
        


class FacadeVisibilityOther(FacadeVisibility):
    
    def createKdTree(self, buildings, totalNumVerts):
        from scipy.spatial import KDTree
        import numpy as np
        
        # allocate the memory for an empty numpy array
        bldgVerts = np.zeros((totalNumVerts, 2))
        index = 0
        for building in buildings:
            if building.polygon:
                for vert in building.polygon.verts:
                    bldgVerts[index] = (vert[0], vert[1])
                    index += 1
        self.kdTree = KDTree(bldgVerts)
        self.bldgVerts = bldgVerts