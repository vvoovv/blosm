from .roof_flat import RoofLeveled
from item.roof_hipped import RoofHipped as ItemRoofHipped

from util import zAxis


# auxiliary indices to deal with quadrangles
_prevIndices = (3, 0, 1, 2)
_indices = (0, 1, 2, 3)
_nextIndices = (1, 2, 3, 0)
_oppositeIndices = (2, 3, 0, 1)


class RoofHipped(RoofLeveled):
    
    height = 4.
    
    def __init__(self, data, itemStore, itemFactory, facadeRenderer, roofRenderer):
        super().__init__(data, itemStore, itemFactory, facadeRenderer, roofRenderer)
        self.extrudeTillRoof = True
        
        self.vector = []
        self.length = []
        self.cos = []
        self.sin = []
        self.distance = []
    
    def render(self, footprint):
        # <firstVertIndex> is the index of the first vertex of the polygon that defines the roof base
        firstVertIndex = self.getRoofFirstVertIndex(footprint)
        
        super().extrude(footprint)
        
        roofItem = ItemRoofHipped.getItem(self.itemFactory, footprint)
        
        # now generate the roof
        if footprint.polygon.n == 4:
            self.generateRoofQuadrangle(footprint, roofItem, firstVertIndex)
        else:
            self.generateRoof(footprint, roofItem, firstVertIndex)
            
        self.facadeRenderer.render(footprint, self.data)
        #self.roofRenderer.render(roofItem)
    
    def generateRoofQuadrangle(self, footprint, roofItem, firstVertIndex):
        verts = footprint.building.verts
        
        vector, length, cos, sin, distance = self.vector, self.length, self.cos, self.sin, self.distance
        vector.clear()
        length.clear()
        cos.clear()
        sin.clear()
        distance.clear()
        
        vector.extend(
            (verts[firstVertIndex + _next] - verts[firstVertIndex + i]) for i, _next in zip(_indices, _nextIndices)
        )
        
        length.extend(
            vector[i].length for i in _indices
        )
        
        cos.extend(
            -( vector[i].dot(vector[_prev]) ) / length[i]/length[_prev] \
                for _prev, i in zip(_prevIndices, _indices)
        )
        
        sin.extend(
            -(vector[i].cross(vector[_prev])[2]) / length[i]/length[_prev] \
                for _prev, i in zip(_prevIndices, _indices)
        )
        
        distance.extend(
            length[i]/( (1.+cos[i])/sin[i] + (1.+cos[_next])/sin[_next] ) \
                for i, _next in zip(_indices, _nextIndices)
        )
        
        if distance[0] == distance[1]:
            # the special case of the square footprint
            return
        else:
            # the first of the two newly created vertices of the hipped roof
            minDistanceIndex1 = min(_indices, key = lambda i: distance[i])
            minDistanceIndex2 = _oppositeIndices[minDistanceIndex1]
            
            # tangent of the roof pitch angle
            tan = footprint.roofHeight / max(distance[minDistanceIndex1], distance[minDistanceIndex2])
            
            vertIndex1 = len(verts)
            verts.append( self.getRoofVert(verts[firstVertIndex + minDistanceIndex1], minDistanceIndex1, tan) )
            
            vertIndex2 = vertIndex1 + 1
            verts.append( self.getRoofVert(verts[firstVertIndex + minDistanceIndex2], minDistanceIndex2, tan) )
            
            self.onFace(
                roofItem,
                (firstVertIndex + minDistanceIndex1, firstVertIndex + _nextIndices[minDistanceIndex1], vertIndex1),
                minDistanceIndex1
            )
            
            self.onFace(
                roofItem,
                (firstVertIndex + _nextIndices[minDistanceIndex1], firstVertIndex + minDistanceIndex2, vertIndex2, vertIndex1),
                _nextIndices[minDistanceIndex1]
            )
            
            self.onFace(
                roofItem,
                (firstVertIndex + minDistanceIndex2, firstVertIndex + _nextIndices[minDistanceIndex2], vertIndex2),
                minDistanceIndex2
            )
            
            self.onFace(
                roofItem,
                (firstVertIndex + _nextIndices[minDistanceIndex2], firstVertIndex + minDistanceIndex1, vertIndex1, vertIndex2),
                _nextIndices[minDistanceIndex2]
            )
    
    def onFace(self, roofItem, indices, edgeIndex):
        self.roofRenderer.r.createFace(
            roofItem.building,
            indices
        )
    
    def generateRoof(self, footprint, firstVertIndex):
        pass
    
    def getRoofVert(self, vert, i, tan):
        return\
            vert + \
            self.distance[i]/self.length[i] * \
            (
                zAxis.cross( self.vector[i] ) + \
                (1. + self.cos[i]) / self.sin[i] * self.vector[i]
            ) + \
            self.distance[i] * tan * zAxis