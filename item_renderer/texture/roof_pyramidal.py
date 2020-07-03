import math
from .. import ItemRenderer
from ..util import initUvAlongPolygonEdge
from grammar import smoothness

from util import zAxis


class RoofPyramidal(ItemRenderer):
        
    def render(self, roofItem):
        smoothFaces = roofItem.getStyleBlockAttr("faces") is smoothness.Smooth
        
        footprint = roofItem.footprint
        polygon = footprint.polygon
        building = roofItem.building
        verts = building.verts
        # the index of the first vertex of the polygon that defines the roof base
        firstVertIndex = roofItem.firstVertIndex
        n = polygon.n
        lastVertIndex = firstVertIndex+n-1
        
        roofHeight = footprint.roofHeight
        center = polygon.centerBB(footprint.roofVerticalPosition)
        
        # create a vertex at the center
        verts.append(center + roofHeight*zAxis)
        
        for pi, vi in zip(range(n-1), range(firstVertIndex, lastVertIndex)):
            # Create a petal of quads, i.e. the quads are created along the generatrix
            
            # <uVec> is a unit vector along the base edge
            uVec, uv0, uv1 = initUvAlongPolygonEdge(polygon, pi, pi+1)
            
            # create a triangle
            self.createFace(
                roofItem,
                smoothFaces,
                (vi, vi+1, -1),
                uVec, uv0, uv1
            )
        
        # create the closing triangle
        # <uVec> is a unit vector along the base edge
        uVec, uv0, uv1 = initUvAlongPolygonEdge(polygon, -1, 0)
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
        verts = roofItem.building.verts
        vec2 = verts[indices[2]]-verts[indices[0]]
        vec2u = vec2.dot(uVec)
        
        self.renderCladding(
            roofItem,
            face,
            (uv0, uv1, (vec2u+uv0[0], (vec2 - vec2u*uVec).length+uv0[1]))
        )