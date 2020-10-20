import math
from itertools import accumulate
from operator import add
from .roof_flat_multi import RoofMulti
from .roof_hipped import RoofHipped
from item.roof_hipped_multi import RoofHippedMulti as ItemRoofHippedMulti

from lib.bpypolyskel.bpypolyskel import polygonize

from util import zAxis

#from .roof_hipped import _dumpInput


class RoofHippedMulti(RoofMulti, RoofHipped):
    
    def __init__(self, data, itemStore, itemFactory, facadeRenderer, roofRenderer):
        super().__init__(data, itemStore, itemFactory, facadeRenderer, roofRenderer)
        
        self.holesInfo = []
    
    def getRoofItem(self, footprint):
        return ItemRoofHippedMulti.getItem(self.itemFactory, footprint)
    
    def render(self, footprint, roofItem):
        # <firstVertIndex> is the index of the first vertex of the polygon that defines the roof base
        firstVertIndex = self.getRoofFirstVertIndex(footprint)
        
        super().extrude(footprint, roofItem)
        
        # now generate the roof
        self.generateRoof(footprint, roofItem, firstVertIndex)
            
        self.facadeRenderer.render(footprint, self.data)
        self.roofRenderer.render(roofItem)
    
    def generateRoof(self, footprint, roofItem, firstVertIndex):
        verts = footprint.building.verts
        numPolygonVerts = footprint.polygon.n
        innerPolygons = roofItem.innerPolygons
        numHoles = len(innerPolygons)
        holesInfo = self.holesInfo
        holesInfo.clear()
        lastVertIndex = firstVertIndex + numPolygonVerts - 1
        
        length, unitVector, roofSideIndices = self.length, self.unitVector, self.roofSideIndices
        # cleanup
        length.clear()
        unitVector.clear()
        roofSideIndices.clear()
        
        unitVector.extend(
            (verts[i+1]-verts[i]) for i in range(firstVertIndex, lastVertIndex)
        )
        unitVector.append( (verts[firstVertIndex]-verts[lastVertIndex]) )
        
        length.extend(
            vec.length for vec in unitVector
        )
        
        for edgeIndex, vec in enumerate(unitVector):
            vec /= length[edgeIndex]
        
        holesOffset = firstVertIndex+numPolygonVerts
        holesInfo.append((holesOffset+innerPolygons[0].n, innerPolygons[0].n))
        holesInfo.extend(
            zip(
                (holesOffset + v for v in accumulate( (2*innerPolygons[i].n + innerPolygons[i+1].n for i in range(numHoles-1)), add)), (innerPolygons[i] for i in range(1, numHoles))
            )
        )
        
        #_dumpInput(verts, firstVertIndex, numPolygonVerts, holesInfo, None)
        
        # calculate polygons formed by the straight skeleton
        polygonize(
            verts,
            firstVertIndex,
            numPolygonVerts,
            holesInfo,
            footprint.roofHeight,
            0,
            roofSideIndices,
            None#unitVector
        )
        
        roofVerticalPosition = verts[firstVertIndex][2]
        
        # calculate tangent of the roof pitch angle
        tan = ( verts[ roofSideIndices[0][2] ][2] - roofVerticalPosition ) / \
        (verts[ roofSideIndices[0][2] ] - verts[ roofSideIndices[0][1] ]).dot( zAxis.cross(unitVector[0]) )
        factor = math.sqrt(1. + tan*tan)
        
        for indices in roofSideIndices:
            edgeIndex = indices[0] - firstVertIndex
            roofItem.addRoofSide(
                indices,
                tuple((0.0, 0.) for _index in range(len(indices))),
                # UV-coordinates
                #( (0., 0.), (length[edgeIndex], 0.) ) + tuple(
                #    (
                #        (verts[ indices[_index] ] - verts[ indices[0] ]).dot(unitVector[edgeIndex]),
                #        (verts[ indices[_index] ][2] - roofVerticalPosition) * factor
                #    ) for _index in range(2, len(indices))
                #),
                edgeIndex,
                self.itemFactory
            )
        