from mathutils import Vector
from . import ItemRendererTexture
from ..util import initUvAlongPolygonEdge
from grammar import smoothness

from util import zAxis


class RoofPyramidal(ItemRendererTexture):
        
    def render(self, roofItem):
        smoothFaces = roofItem.getStyleBlockAttr("faces") is smoothness.Smooth
        
        footprint = roofItem.footprint
        polygon = footprint.polygon
        verts = roofItem.building.renderInfo.verts
        # the index of the first vertex of the polygon that defines the roof base
        firstVertIndex = roofItem.firstVertIndex
        n = polygon.numEdges
        lastVertIndex = firstVertIndex+n-1
        
        roofHeight = footprint.roofHeight
        center = polygon.centerBB3d(footprint.roofVerticalPosition)
        
        # create a vertex at the center
        verts.append(center + roofHeight*zAxis)
        
        vectors = polygon.getVectors()
        # Cycle for <n-1> repetitions. <range(firstVertIndex, lastVertIndex)> gives
        # exactly <n-1> numbers. That's why it's the first one in <zip(..)>.
        for vertIndex, vector in zip(range(firstVertIndex, lastVertIndex), vectors):
            # Create a petal of quads, i.e. the quads are created along the generatrix
            
            # <uVec> is a unit vector along the base edge
            uVec, uv0, uv1 = initUvAlongPolygonEdge(vector)
            
            # create a triangle
            self.createFace(
                roofItem,
                smoothFaces,
                (vertIndex, vertIndex+1, -1),
                uVec, uv0, uv1
            )
        
        # create the closing triangle
        # <uVec> is a unit vector along the base edge
        uVec, uv0, uv1 = initUvAlongPolygonEdge(next(vectors))
        self.createFace(
            roofItem,
            smoothFaces,
            (lastVertIndex, firstVertIndex, -1),
            uVec, uv0, uv1
        )
    
    def createFace(self, roofItem, smooth, indices, uVec, uv0, uv1):
        face = self.r.createFace(roofItem.building, indices)
        if smooth:
            face.smooth = smooth
        
        # assign UV-coordinates
        verts = roofItem.building.renderInfo.verts
        # a 3D vector from the leftmost vertex at the bottom to the vertex at the top
        vec2 = verts[indices[2]]-verts[indices[0]]
        vec2u = vec2[0]*uVec[0] + vec2[1]*uVec[1] # i.e. vec2.dot(uVec)
        
        self.renderCladding(
            roofItem,
            face,
            # (uv0, uv1, (vec2u+uv0[0], (vec2 - vec2u*uVec).length+uv0[1]))
            # uv0[0] is equal to zero
            # uv0[1] is equal to zero
            ( uv0, uv1, (vec2u, ( vec2 - vec2u*Vector((uVec[0], uVec[1], 0.)) ).length) )
        )