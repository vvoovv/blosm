import math
from .roof_flat import RoofLeveled
from item.roof_hipped import RoofHipped as ItemRoofHipped

from lib.bpypolyskel.bpypolyskel import polygonize

from util import zAxis


def _dumpInput(verts, firstVertIndex, numPolygonVerts, holesInfo, unitVectors):
    with open("D:/tmp/bpypolyskel_test.py", 'w') as file:
        file.write("import mathutils\n")
        file.write("import matplotlib.pyplot as plt\n")
        file.write("import bpypolyskel\n")
        file.write("\n")
        
        file.write("verts = [\n")
        firstVert = True
        for vert in verts:
            if firstVert:
                firstVert = False
            else:
                file.write(",\n")
            file.write("    mathutils.Vector((%s,%s,%s))" % (vert[0], vert[1], vert[2]))
        file.write("\n]\n")
        
        if unitVectors:
            file.write("unitVectors = [\n")
            firstVert = True
            for unitVector in unitVectors:
                if firstVert:
                    firstVert = False
                else:
                    file.write(",\n")
                file.write("    mathutils.Vector((%s,%s,%s))" % (unitVector[0], unitVector[1], unitVector[2]))
            file.write("\n]\n")
        
        if holesInfo:
            file.write("holesInfo = [\n")
            firstHole = True
            for holeInfo in holesInfo:
                if firstHole:
                    firstHole = False
                else:
                    file.write(",\n")
                file.write("    (%s,%s)" % (holeInfo[0], holeInfo[1]))
            file.write("\n]\n")
        else:
            file.write("holesInfo = None\n")
        
        file.write("firstVertIndex = %s\n" % firstVertIndex)
        file.write("numPolygonVerts = %s\n" % numPolygonVerts)
        if unitVectors:
            file.write("faces = bpypolyskel.polygonize(verts, firstVertIndex, numPolygonVerts, holesInfo, 0.0, 0.5, None, unitVectors)")
        else:
            file.write("faces = bpypolyskel.polygonize(verts, firstVertIndex, numPolygonVerts, holesInfo, 0.0, 0.5, None, None)")
        file.write("\n")
        
        file.write("fig = plt.figure()\n")
        file.write("ax = fig.gca(projection='3d')\n")
        file.write("for face in faces:\n")
        file.write("    for edge in zip(face, face[1:] + face[:1]):\n")
        file.write("        p1 = verts[edge[0]]\n")
        file.write("        p2 = verts[edge[1]]\n")
        file.write("        ax.plot([p1.x,p2.x],[p1.y,p2.y],[p1.z,p2.z])\n")
        file.write("plt.show()\n")


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
        
        # unit vectors along the edges of footprint polygon
        self.unitVector = []
        self.roofSideIndices = []
        
        # vectors along the edges of the footprint polygon
        self.vector = []
        # lengths of the edges of the footprint polygon
        self.length = []
        # cosines of the angles of the footprint polygon
        self.cos = []
        # sines of the angles of the footprint polygon
        self.sin = []
        # distances between the edge of the footprint polygon and
        # the point of the edge event (i.e. where the bisectors originating
        # from the edge vertices meet)
        self.distance = []
    
    def getRoofItem(self, footprint):
        return ItemRoofHipped.getItem(self.itemFactory, footprint)
    
    def render(self, footprint, roofItem):
        # <firstVertIndex> is the index of the first vertex of the polygon that defines the roof base
        firstVertIndex = self.getRoofFirstVertIndex(footprint)
        
        super().extrude(footprint, roofItem)
        
        # now generate the roof
        if footprint.polygon.n == 4:
            self.generateRoofQuadrangle(footprint, roofItem, firstVertIndex)
        else:
            self.generateRoof(footprint, roofItem, firstVertIndex)
            
        self.facadeRenderer.render(footprint, self.data)
        self.roofRenderer.render(roofItem)
    
    def generateRoofQuadrangle(self, footprint, roofItem, firstVertIndex):
        verts = footprint.building.verts
        
        vector, length, cos, sin, distance = self.vector, self.length, self.cos, self.sin, self.distance
        # cleanup
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
            # The first of the two newly created vertices of the hipped roof:
            # we find the very first occurance of the edge event,
            # namely the index of the polygon edge with the minimum <distance>
            minDistanceIndex1 = min(_indices, key = lambda i: distance[i])
            minDistanceIndex1Next = _nextIndices[minDistanceIndex1]
            # The second of the two newly created vertices of the hipped roof.
            # Note that it's an assumption that the other edge event for the quadrangle
            # occurs for the polygon edge opposite to the edge with the index <minDistanceIndex1>
            minDistanceIndex2 = _oppositeIndices[minDistanceIndex1]
            minDistanceIndex2Next = _nextIndices[minDistanceIndex2]
            
            # tangent of the roof pitch angle
            tan = footprint.roofHeight / max(distance[minDistanceIndex1], distance[minDistanceIndex2])
            factor = math.sqrt(1. + tan*tan)
            
            # add two new vertices to the Python list <verts>
            vertIndex1 = len(verts)
            verts.append( self.getRoofVert(verts[firstVertIndex + minDistanceIndex1], minDistanceIndex1, tan) )
            
            vertIndex2 = vertIndex1 + 1
            verts.append( self.getRoofVert(verts[firstVertIndex + minDistanceIndex2], minDistanceIndex2, tan) )
            
            # variable below are used for the assignment of the UV-coordinates
            if self.setUvs:
                u1 = distance[minDistanceIndex1] * (1. + cos[minDistanceIndex1]) / sin[minDistanceIndex1]
                v1 = distance[minDistanceIndex1] * factor
                
                u2 = distance[minDistanceIndex2] * (1. + cos[minDistanceIndex2]) / sin[minDistanceIndex2]
                v2 = distance[minDistanceIndex2] * factor
            
            # Triangle of the hipped roof originating from the polygon edge
            # with the index <minDistanceIndex1>
            roofItem.addRoofSide(
                (firstVertIndex + minDistanceIndex1, firstVertIndex + minDistanceIndex1Next, vertIndex1),
                ( (0., 0.), (length[minDistanceIndex1], 0.), (u1, v1) ) if self.setUvs else None,
                minDistanceIndex1,
                self.itemFactory
            )
            
            # Quadrangle of the hipped roof originating from the polygon edge
            # with the index <minDistanceIndex1Next>
            roofItem.addRoofSide(
                (firstVertIndex + _nextIndices[minDistanceIndex1], firstVertIndex + minDistanceIndex2, vertIndex2, vertIndex1),
                (
                    (0., 0.),
                    (length[minDistanceIndex1Next], 0.),
                    (length[minDistanceIndex1Next] - u2, v2),
                    (length[minDistanceIndex1] - u1, v1)
                ) if self.setUvs else None,
                minDistanceIndex1Next,
                self.itemFactory
            )
            
            # Triangle of the hipped roof originating from the polygon edge
            # with the index <minDistanceIndex2>
            roofItem.addRoofSide(
                (firstVertIndex + minDistanceIndex2, firstVertIndex + minDistanceIndex2Next, vertIndex2),
                ( (0., 0.), (length[minDistanceIndex2], 0.), (u2, v2) ) if self.setUvs else None,
                minDistanceIndex2,
                self.itemFactory
            )
            
            # Quadrangle of the hipped roof originating from the polygon edge
            # with the index <minDistanceIndex2Next>
            roofItem.addRoofSide(
                (firstVertIndex + minDistanceIndex2Next, firstVertIndex + minDistanceIndex1, vertIndex1, vertIndex2),
                (
                    (0., 0.),
                    (length[minDistanceIndex2Next], 0.),
                    (length[minDistanceIndex2Next] - u1, v1),
                    (length[minDistanceIndex2] - u2, v2)
                ) if self.setUvs else None,
                minDistanceIndex2Next,
                self.itemFactory
            )
    
    def generateRoof(self, footprint, roofItem, firstVertIndex):
        verts = footprint.building.verts
        numPolygonVerts = footprint.polygon.n
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
        
        #if footprint.element.tags.get("dump") == "yes":
        #    _dumpInput(verts, firstVertIndex, numPolygonVerts, None, unitVector)
        #_dumpInput(verts, firstVertIndex, numPolygonVerts, None, unitVector)
        
        # calculate polygons formed by the straight skeleton
        polygonize(
            verts,
            firstVertIndex,
            numPolygonVerts,
            None,
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
            edgeIndex = indices[0] - firstVertIndex
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
    
    def getRoofVert(self, vert, i, tan):
        """
        Assuming the edge event occured for the polygon edge with the index <i>,
        get the vertex of the edge event (i.e. where the bisectors originating
        from the polygon edge vertices meet).
        <tan> defines the tangent of the roof pitch angle, so an additional offset
        <self.distance[i] * tan * zAxis> is added to the resulting vertex.
        """
        return\
            vert + \
            self.distance[i]/self.length[i] * \
            (
                zAxis.cross( self.vector[i] ) + \
                (1. + self.cos[i]) / self.sin[i] * self.vector[i]
            ) + \
            self.distance[i] * tan * zAxis