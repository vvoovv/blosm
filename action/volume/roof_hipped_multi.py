import math
from itertools import accumulate
from operator import add
from .roof_flat_multi import RoofMulti
from .roof_hipped import RoofHipped
from item.roof_hipped_multi import RoofHippedMulti as ItemRoofHippedMulti
from mathutils import Vector

from lib.bpypolyskel.bpypolyskel import polygonize

from util import zAxis

#from .roof_hipped import _dumpInput


class RoofHippedMulti(RoofMulti, RoofHipped):
    
    def __init__(self, data, itemStore, itemFactory, facadeRenderer, roofRenderer):
        super().__init__(data, itemStore, itemFactory, facadeRenderer, roofRenderer)
        
        self.holesInfo = []
        # Python dictionary used for mapping:
        # the index of the first face vertex -> counter index;
        # the face is formed by the straight skeleton;
        # counter indices: 0 for the outer polygon, 1 for the first hole and so on;
        # the dictionary is used if there two or more holes in the polygon
        self.faceToContourIndex = {}
    
    def getRoofItem(self, footprint):
        return ItemRoofHippedMulti.getItem(self.itemFactory, footprint)
    
    def extrudeInnerPolygons(self, footprint, roofItem):
        if footprint.noWalls:
            z = footprint.roofVerticalPosition
            # the basement of the roof
            footprint.building.verts.extend(
                Vector((v.x, v.y, z)) for innerPolygon in roofItem.innerPolygons for v in innerPolygon.verts
            )
            return
        super().extrudeInnerPolygons(footprint, roofItem)
    
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
        lastVertIndex = firstVertIndex + numPolygonVerts - 1
        
        length, unitVector, roofSideIndices = self.length, self.unitVector, self.roofSideIndices
        # cleanup
        length.clear()
        unitVector.clear()
        roofSideIndices.clear()
        
        holesInfo = self.holesInfo
        holesInfo.clear()
        
        # the outer contour
        unitVector.extend(
            (verts[i+1]-verts[i]) for i in range(firstVertIndex, lastVertIndex)
        )
        unitVector.append( (verts[firstVertIndex]-verts[lastVertIndex]) )
        
        
        if footprint.noWalls:
            _offset = firstVertIndex + numPolygonVerts
            holesInfo.append((_offset, innerPolygons[0].n))
            holesInfo.extend(
                zip(
                    (_offset + v for v in accumulate( (innerPolygons[i].n for i in range(numHoles-1)), add)), (innerPolygons[i].n for i in range(1, numHoles))
                )
            )
        else:
            _offset = firstVertIndex + numPolygonVerts + innerPolygons[0].n
            holesInfo.append((_offset, innerPolygons[0].n))
            holesInfo.extend(
                zip(
                    (_offset + v for v in accumulate( (innerPolygons[i].n + innerPolygons[i+1].n for i in range(numHoles-1)), add)), (innerPolygons[i].n for i in range(1, numHoles))
                )
            )
        
        # the holes
        for firstVertIndexHole,numVertsHole in holesInfo:
            lastVertIndexHole = firstVertIndexHole+numVertsHole-1
            unitVector.extend(
                (verts[i+1]-verts[i]) for i in range(firstVertIndexHole, lastVertIndexHole)
            )
            unitVector.append( (verts[firstVertIndexHole]-verts[lastVertIndexHole]) )

        if numHoles > 1:
            faceToContourIndex = self.faceToContourIndex
            faceToContourIndex.clear()
            for i in range(firstVertIndex, firstVertIndex+numPolygonVerts):
                faceToContourIndex[i] = firstVertIndex
            _offset = numPolygonVerts
            for firstVertIndexHole,numVertsHole in holesInfo:
                for i in range(firstVertIndexHole, firstVertIndexHole+numVertsHole):
                    faceToContourIndex[i] = firstVertIndexHole - _offset
                _offset += numVertsHole

        length.extend(
            vec.length for vec in unitVector
        )
        
        for edgeIndex, vec in enumerate(unitVector):
            vec /= length[edgeIndex]
        
        
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
            unitVector
        )
        
        roofVerticalPosition = verts[firstVertIndex][2]
        
        # calculate tangent of the roof pitch angle
        tan = ( verts[ roofSideIndices[0][2] ][2] - roofVerticalPosition ) / \
        (verts[ roofSideIndices[0][2] ] - verts[ roofSideIndices[0][1] ]).dot( zAxis.cross(unitVector[0]) )
        factor = math.sqrt(1. + tan*tan)
        
        for indices in roofSideIndices:
            if numHoles == 1:
                edgeIndex = indices[0] - holesInfo[0][0] + numPolygonVerts\
                    if indices[0] >= holesInfo[0][0] else\
                    indices[0] - firstVertIndex
            else:
                edgeIndex = indices[0] - faceToContourIndex[indices[0]]
            roofItem.addRoofSide(
                indices,
                # UV-coordinates
                ( (0., 0.), (length[edgeIndex], 0.) ) + tuple(
                    (
                        (verts[ indices[_index] ] - verts[ indices[0] ]).dot(unitVector[edgeIndex]),
                        (verts[ indices[_index] ][2] - roofVerticalPosition) * factor
                    ) for _index in range(2, len(indices))
                ),
                edgeIndex,
                self.itemFactory
            )
        