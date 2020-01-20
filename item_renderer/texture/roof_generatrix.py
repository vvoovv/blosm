import math
from .. import ItemRenderer
from grammar import smoothness

from util import zAxis


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
    
    def __init__(self, generatrix):
        super().__init__(False)
        self.generatrix = generatrix
        # The variable below indicates if the last point of the generatrix is located at zero,
        # i.e. in the center of the underlying polygon
        self.hasCenter = not self.generatrix[-1][0]
    
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
        roofVerticalPosition = footprint.roofVerticalPosition
        
        center = polygon.centerBB(roofVerticalPosition)
        
        n = polygon.n
        numRows = len(self.generatrix)
        
        vertIndexOffset = len(verts)
        
        verts.extend(
            center + gen[gi][0]*(verts[firstVertIndex+vi]-center) + gen[gi][1]*roofHeight*zAxis\
            for gi in range(1, numRows-1 if self.hasCenter else numRows) for vi in range(n)
        )
                
        # the first row
        for vi in range(n-1):
            self.createFace(
                building,
                smoothFaces,
                (firstVertIndex+vi, firstVertIndex+vi+1, vertIndexOffset+vi+1, vertIndexOffset+vi)
            )
        # and the closing quad for the ring of the first row
        self.createFace(
            building,
            smoothFaces,
            (firstVertIndex+n-1, firstVertIndex, vertIndexOffset, vertIndexOffset+n-1)
        )
        
        # The rest of rows except the last row made of triangles ending at the center of
        # the underlying polygon
        for gi in range(1, numRows-2 if self.hasCenter else numRows-1):
            for vi in range(vertIndexOffset, vertIndexOffset+n-1):
                self.createFace(building, smoothFaces, (vi, vi+1, vi+n+1, vi+n))
            # and the closing quad for the ring
            self.createFace(
                building,
                smoothFaces,
                (vertIndexOffset+n-1, vertIndexOffset, vertIndexOffset+n, vertIndexOffset+2*n-1)
            )
            vertIndexOffset += n
        
        # Treat the case if the last point of the generatrix is located at zero,
        # i.e. in the center of the underlying polygon
        if self.hasCenter:
            # create a vertex at the center
            verts.append(center + gen[-1][1]*roofHeight*zAxis)
            # the last row made of triangles ending at the center of the underlying polygon
            for vi in range(vertIndexOffset, vertIndexOffset+n-1):
                self.createFace(building, smoothFaces, (vi, vi+1, -1))
            # and the closing triangle for the ring
            self.createFace(building, smoothFaces, (vertIndexOffset+n-1, vertIndexOffset, -1))
    
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
        
        center = polygon.centerBB(roofVerticalPosition)
        
        n = polygon.n
        numRows = len(self.generatrix)
        if self.hasCenter:
            numRows -= 1
        
        vertIndexOffset = len(verts)
        
        # Create two copies of each vertex.
        # Note that in contrast to <self.renderIgnoreEdges(..)> we also create the copies
        # of the vertices that define the top part of facades
        verts.extend(
            center + gen[gi][0]*(verts[firstVertIndex+vi]-center) + gen[gi][1]*roofHeight*zAxis\
            for vi in range(n) for gi in range(numRows) for _ in range(2)
        )
        if self.hasCenter:
            # Also create vertices at the center if the last point of the generatrix is located at zero,
            # i.e. in the center of the underlying polygon. We create <n> copies of the vertex.
            center = center + gen[-1][1]*roofHeight*zAxis
            centerIndexOffset = len(verts)
            verts.extend(center for _ in range(n))
        
        # In contrast to <self.renderIgnoreEdges(..)> we do not treat the very first row separately
        _vertIndexOffset = vertIndexOffset
        for pi in range(n-1):
            for vi in range(vertIndexOffset+1, vertIndexOffset+2*numRows-1, 2):
                self.createFace(
                    building,
                    True,
                    (vi, vi+2*numRows-1, vi+2*numRows+1, vi+2)
                )
            if self.hasCenter:
                self.createFace(
                    building,
                    True,
                    (vi+2, vi+2*numRows+1, centerIndexOffset+pi)
                )
            vertIndexOffset += 2*numRows
        
        # and the closing quad for the all rings
        for vi in range(numRows-1):
            self.createFace(
                building,
                True,
                (vertIndexOffset+2*vi+1, _vertIndexOffset+2*vi, _vertIndexOffset+2*vi+2, vertIndexOffset+2*vi+3)
            )
        if self.hasCenter:
            self.createFace(
                building,
                True,
                (vertIndexOffset+2*numRows-1, _vertIndexOffset+2*numRows-2, -1)
            )
    
    def createFace(self, building, smooth, indices):
        face = self.r.createFace(building, indices)
        if smooth:
            face.smooth = smooth