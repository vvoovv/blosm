from math import sqrt
import numpy as np


class FacadeVisibility:
    
    def __init__(self, searchRange=(10., 100.)):
        self.app = None
        self.kdTree = None
        self.bldgVerts = None
        self.searchWidthMargin = searchRange[0]
        self.searchHeight_2 = searchRange[1]*searchRange[1]
        self.vertIndexToBldgIndex = []
    
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
        self.vertIndexToBldgIndex.extend(
            bldgIndex for bldgIndex, building in enumerate(buildings) if building.polygon for _ in range(building.polygon.n)
        )
        
        # allocate the memory for an empty numpy array
        bldgVerts = np.zeros((totalNumVerts, 2))
        # fill in <self.bldgVerts>
        index = 0
        for building in buildings:
            if building.polygon:
                # store the index of the first vertex of <building> in <self.bldgVerts> 
                building.auxIndex = index
                for vert in building.polygon.verts:
                    bldgVerts[index] = (vert[0], vert[1])
                    index += 1
        self.bldgVerts = bldgVerts
        
        self.createKdTree(buildings, totalNumVerts)
        
        self.calculateFacadeVisibility(manager)
    
    def cleanup(self):
        self.kdTree = None
        self.bldgVerts = None
        self.vertIndexToBldgIndex.clear()

    def calculateFacadeVisibility(self, manager):
        buildings = manager.buildings
        
        for way in self.app.managersById["ways"].getAllWays():
            if not way.polyline:
                way.initPolyline()
            for segmentCenter, segmentUnitVector, segmentLength in way.segments:
                searchWidth = segmentLength/2. + self.searchWidthMargin
                # Get all building vertices for the buildings whose vertices are returned
                # by the query to the KD tree
                queryBldgIndices = set(
                    self.makeKdQuery(segmentCenter, sqrt(searchWidth*searchWidth + self.searchHeight_2))
                )
                queryBldgVertIndices = tuple(
                    vertIndex for bldgIndex in queryBldgIndices for vertIndex in range(buildings[bldgIndex].auxIndex, buildings[bldgIndex].auxIndex + buildings[bldgIndex].polygon.n)
                )
                queryBldgVerts = self.bldgVerts[queryBldgVertIndices,:]
                
                # transform <queryBldgVerts> to the system of reference of <way>
                matrix = np.array(( segmentUnitVector, (segmentUnitVector[1], -segmentUnitVector[0]) ))
                
                # shift origin to the segment center
                queryBldgVerts -= segmentCenter
                # rotate <queryBldgVerts>, output back to <queryBldgVerts>
                np.matmul(queryBldgVerts, matrix, out=queryBldgVerts)


class FacadeVisibilityBlender(FacadeVisibility):
    
    def createKdTree(self, buildings, totalNumVerts):
        from mathutils.kdtree import KDTree
        kdTree = KDTree(totalNumVerts)
        
        index = 0
        for building in buildings:
            if building.polygon:
                for vert in building.polygon.verts:
                    kdTree.insert(vert, index)
                    index += 1
        kdTree.balance()
        self.kdTree = kdTree
    
    def makeKdQuery(self, searchCenter, searchRadius):
        """
        Returns a generator of building indices in <manager.buldings>.
        The buildings indices aren't unique
        """
        return (
            self.vertIndexToBldgIndex[vertIndex] for _,vertIndex,_ in self.kdTree.find_range(searchCenter, searchRadius)
        )


class FacadeVisibilityOther(FacadeVisibility):
    
    def createKdTree(self, buildings, totalNumVerts):
        from scipy.spatial import KDTree
        
        self.kdTree = KDTree(self.bldgVerts)
    
    def makeKdQuery(self, searchCenter, searchRadius):
        """
        Returns a generator of building indices in <manager.buldings>.
        The buildings indices aren't unique
        """
        return (
            self.vertIndexToBldgIndex[vertIndex] for vertIndex in self.kdTree.query_ball_point(searchCenter, searchRadius)
        )