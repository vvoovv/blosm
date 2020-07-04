import math
from .. import ItemRenderer
from ..util import initUvAlongPolygonEdge
from grammar import smoothness
from util import zAxis

Center = 1
MiddleOfTheLongesSide = 2


# Generatrix for a dome roof.
# generatix for dome is circle equation in the parameterized form x=x(t), y=y(t)
def generatrix_dome(rows):
    profile=[]
    for j in range(rows):
        x = math.cos(j/rows*math.pi/2)
        z = math.sin(j/rows*math.pi/2)
        profile.append((x,z))
    profile.append((0., 1.))
    return profile

# Generatrix for an onion roof.
# There is no explicit formula for an onion, that's why we will just write it as a set of vertex cordinates. 
# Note, that there are actually more than 5 typical onion types even for orthodox churches only.
# Other forms will be added later.
# (r, z) 
generatrix_onion =(
    (1.0000,    0.0000),
    (1.2971,    0.0999),
    (1.2971,    0.2462),
    (1.1273,    0.3608),
    (0.6219,    0.4785),
    (0.2131,    0.5984),
    (0.1003,    0.7243),
    (0.0000,    1.0000)
)


class RoofGeneratrix(ItemRenderer):
    
    def __init__(self, generatrix, basePointPosition, exportMaterials=False):
        super().__init__(exportMaterials)
        self.generatrix = generatrix
        # The variable below indicates if the last point of the generatrix is located at zero,
        # i.e. in the center of the underlying polygon
        self.hasCenter = not self.generatrix[-1][0]
        self.basePointPosition=basePointPosition
    
    def render(self, roofItem):
        smoothFaces = roofItem.getStyleBlockAttr("faces") is smoothness.Smooth
        sharpSideEdges = smoothFaces and roofItem.getStyleBlockAttr("sharpEdges") is smoothness.Side
        
        if sharpSideEdges:
            self.renderSharpSideEdges(roofItem)
        else:
            self.renderIgnoreEdges(roofItem, smoothFaces)
        
    def renderIgnoreEdges(self, roofItem, smoothFaces):
        """
        Render the roof with smooth or flat faces. Side edges aren't processed in any special way.
        """
        gen = self.generatrix
        footprint = roofItem.footprint
        polygon = footprint.polygon
        building = roofItem.building
        verts = building.verts
        # the index of the first vertex of the polygon that defines the roof base
        firstVertIndex = roofItem.firstVertIndex
        
        roofHeight = footprint.roofHeight
       
        if self.basePointPosition == MiddleOfTheLongesSide:
            center = polygon.middleOfTheLongestSide(footprint.roofVerticalPosition)
        else:
            center = polygon.centerBB(footprint.roofVerticalPosition)
        
        n = polygon.n
        numRows = len(self.generatrix)
        if self.hasCenter:
            numRows -= 1
        
        vertIndex = vertIndex0 = len(verts)
        
        verts.extend(
            center + gen[vi][0]*(verts[firstVertIndex+pi]-center) + gen[vi][1]*roofHeight*zAxis\
            for pi in range(n) for vi in range(1, numRows)
        )
        if self.hasCenter:
            # create a vertex at the center
            verts.append(center + gen[-1][1]*roofHeight*zAxis)
        
        for pi in range(n-1):
            # Create a petal of quads, i.e. the quads are created along the generatrix
            
            # <uVec> is a unit vector along the base edge
            uVec, uv0, uv1 = initUvAlongPolygonEdge(polygon, pi, pi+1)
            
            # The quad for the first row
            uv0, uv1 = self.createFace(
                roofItem,
                smoothFaces,
                (firstVertIndex+pi, firstVertIndex+pi+1, vertIndex+numRows-1, vertIndex),
                uVec, uv0, uv1
            )
            # The rest of the quads for the petal
            for vi,vi2 in zip(range(vertIndex, vertIndex+numRows-2), range(vertIndex+numRows-1, vertIndex+2*numRows-3)):
                uv0, uv1 = self.createFace(
                    roofItem,
                    smoothFaces,
                    (vi, vi2, vi2+1, vi+1),
                    uVec, uv0, uv1
                )
            if self.hasCenter:
                # Treat the case if the last point of the generatrix is located at zero,
                # i.e. in the center of the underlying polygon.
                # We create here a triangle instead of a quad
                self.createFace(
                    roofItem,
                    smoothFaces,
                    (vi+1, vi2+1, -1),
                    uVec, uv0, uv1
                )
            vertIndex += numRows-1
        
        # Create the closing petal of quads
        
        # <uVec> is a unit vector along the base edge
        uVec, uv0, uv1 = initUvAlongPolygonEdge(polygon, -1, 0)
        # The quad for first row
        uv0, uv1 = self.createFace(
            roofItem,
            smoothFaces,
            (firstVertIndex+n-1, firstVertIndex, vertIndex0, vertIndex),
            uVec, uv0, uv1
        )
        # The rest of the quads for the petal
        for vi,vi2 in zip(range(vertIndex, vertIndex+numRows-2), range(vertIndex0, vertIndex0+numRows-2)):
            uv0, uv1 = self.createFace(
                roofItem,
                smoothFaces,
                (vi, vi2, vi2+1, vi+1),
                uVec, uv0, uv1
            )
        if self.hasCenter:
            # Treat the case if the last point of the generatrix is located at zero,
            # i.e. in the center of the underlying polygon.
            # We create here a triangle instead of a quad.
            self.createFace(
                roofItem,
                smoothFaces,
                (vi+1, vi2+1, -1),
                uVec, uv0, uv1
            )
    
    def renderSharpSideEdges(self, roofItem):
        """
        Render the roof with smooth faces and sharp side eges of the faces
        """
        gen = self.generatrix
        footprint = roofItem.footprint
        polygon = footprint.polygon
        building = roofItem.building
        verts = building.verts
        # the index of the first vertex of the polygon that defines the roof base
        firstVertIndex = roofItem.firstVertIndex
        
        roofHeight = footprint.roofHeight
        roofVerticalPosition = footprint.roofVerticalPosition

        if self.basePointPosition == MiddleOfTheLongesSide:
            center = polygon.middleOfTheLongestSide(footprint.roofVerticalPosition)
        else:
            center = polygon.centerBB(footprint.roofVerticalPosition)
        
        n = polygon.n
        numRows = len(self.generatrix)
        if self.hasCenter:
            numRows -= 1
        
        vertIndexOffset = len(verts)
        vertIndexOffset2 = vertIndexOffset2_ = vertIndexOffset + numRows*n
        
        # Note that in contrast to <self.renderIgnoreEdges(..)> we also create the copies
        # of the vertices that define the top part of facades
        verts.extend(
            center + gen[vi][0]*(verts[firstVertIndex+pi]-center) + gen[vi][1]*roofHeight*zAxis\
            for pi in range(n) for vi in range(numRows)
        )
        # Create a copy of each vertex created above
        verts.extend(verts[vi] for vi in range(vertIndexOffset, vertIndexOffset2))
        if self.hasCenter:
            # Also create vertices at the center of the polygon if the last point of the generatrix
            # is located at zero, i.e. in the center of the underlying polygon.
            # We create <n> copies of the vertex.
            center = center + gen[-1][1]*roofHeight*zAxis
            centerIndexOffset = len(verts)
            verts.extend(center for _ in range(n))
        
        vertIndexOffset2 = vertIndexOffset2+numRows
        for pi in range(n-1):
            # Create a petal of quads, i.e. the quads are created along the generatrix.
            # In contrast to <self.renderIgnoreEdges(..)> we do not treat the very first row separately.
            
            # <uVec> is a unit vector along the base edge
            uVec, uv0, uv1 = initUvAlongPolygonEdge(polygon, pi, pi+1)
            for vi, vi2 in zip(range(vertIndexOffset, vertIndexOffset+numRows-1), range(vertIndexOffset2, vertIndexOffset2+numRows-1)):
                uv0, uv1 = self.createFace(
                    roofItem,
                    True,
                    (vi, vi2, vi2+1, vi+1),
                    uVec, uv0, uv1
                )
            if self.hasCenter:
                # Treat the case if the last point of the generatrix is located at zero,
                # i.e. in the center of the underlying polygon.
                # We create here a triangle instead of a quad
                self.createFace(
                    roofItem,
                    True,
                    (vi+1, vi2+1, centerIndexOffset+pi),
                    uVec, uv0, uv1
                )
            vertIndexOffset += numRows
            vertIndexOffset2 += numRows
        
        # Create the closing petal of quads
        
        # <uVec> is a unit vector along the base edge
        uVec, uv0, uv1 = initUvAlongPolygonEdge(polygon, -1, 0)
        for vi,vi2 in zip(range(vertIndexOffset, vertIndexOffset+numRows-1), range(vertIndexOffset2_, vertIndexOffset2_+numRows-1)):
            uv0, uv1 = self.createFace(
                roofItem,
                True,
                (vi, vi2, vi2+1, vi+1),
                uVec, uv0, uv1
            )
        if self.hasCenter:
            # Treat the case if the last point of the generatrix is located at zero,
            # i.e. in the center of the underlying polygon.
            # We create here a triangle instead of a quad.
            self.createFace(
                roofItem,
                True,
                (vertIndexOffset+numRows-1, vertIndexOffset2_+numRows-1, -1),
                uVec, uv0, uv1
            )
    
    def createFace(self, roofItem, smooth, indices, uVec, uv0, uv1):
        face = self.r.createFace(roofItem.building, indices)
        if smooth:
            face.smooth = smooth
        
        # assign UV-coordinates
        isQuad = len(indices)==4
        verts = roofItem.building.verts
        if isQuad:
            vec3 = verts[indices[3]]-verts[indices[0]]
            vec3u = vec3.dot(uVec)
            uv3 = (vec3u+uv0[0], (vec3 - vec3u*uVec).length+uv0[1])
        vec2 = verts[indices[2]]-verts[indices[0]]
        vec2u = vec2.dot(uVec)
        uv2 = (vec2u+uv0[0], (vec2 - vec2u*uVec).length+uv0[1])
        
        self.renderCladding(
            roofItem,
            face,
            (uv0, uv1, uv2, uv3) if isQuad else (uv0, uv1, uv2)
        )
        
        if isQuad:
            return uv3, uv2