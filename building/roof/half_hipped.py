"""
This file is part of blender-osm (OpenStreetMap importer for Blender).
Copyright (C) 2014-2017 Vladimir Elistratov
prokitektura+support@gmail.com

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from mathutils import Vector
from .profile import RoofProfile, Slot, gabledRoof


class MiddleSlot(Slot):
    
    def __init__(self):
        super().__init__()
        # The first element of the Python lists <self.front> and <self.back> is <y> from <self.parts>,
        # the second one is vertex indices for a wall face
        # Front wall face
        self.front = [None, None]
        # Back wall face
        self.back = [None, None]
    
    def reset(self):
        super().reset()
        self.front[0] = None
        self.back[0] = None

    def processWallFace(self, indices, pv1, pv2):
        """
        The method is used to find the front and the back wall faces.
        The arguments of the method are described in the parent class.
        """
        y = self.parts[-1][0]
        front = self.front
        back = self.back
        noFront = front[0] is None
        noBack = back[0] is None
        if noFront or y < front[0]:
            if not noFront and noBack:
                back[0] = front[0]
                back[1] = front[1]
            front[0] = y
            front[1] = indices, pv1, pv2
        elif noBack or y > back[0]:
            back[0] = y
            back[1] = indices, pv1, pv2
            
            
class RoofHalfHipped(RoofProfile):
    
    widthFactor = 0.5

    def __init__(self):
        super().__init__(gabledRoof)
        # replace the middle slot defining the roof ridge
        slots = self.slots
        slots = (slots[0], MiddleSlot(), slots[2])
        slots[1].n = slots[2]
        self.slots = slots
        
    def make(self, bldgMaxHeight, roofMinHeight, bldgMinHeight, osm):
        super().make(bldgMaxHeight, roofMinHeight, bldgMinHeight, osm)
        # the middle slot defining the roof ridge
        slot = self.slots[1]
        front = slot.front
        back = slot.back
        slot = slot.parts
        # check if have front and back wall faces
        if back[0] is None:
            back = None
        if front[0] is None:
            front = None
        elif back is None:
            # check if <front> defines the wall face for the front side or for the back one
            if front[0] > slot[0][0]:
                back = front
                front = None
        if not front and not back:
            return True
        
        # indices for the ridge vertices
        indexFront = slot[0][1][0]
        indexBack = slot[-1][1][0]
        # the vector along the ridge
        ridgeVector = self.verts[indexBack] - self.verts[indexFront]
        ridgeLength = slot[-1][0] - slot[0][0]
        
        if front:
            self.makeHalfHipped(front, indexFront, ridgeVector, ridgeLength, roofMinHeight)
        if back:
            self.makeHalfHipped(back, indexBack, -ridgeVector, ridgeLength, roofMinHeight)
        return True
    
    def makeHalfHipped(self, wallFace, ridgeVertexIndex, ridgeVector, ridgeLength, roofMinHeight):
        verts = self.verts
        # the middle point
        sx = 0.5
        sz = verts[ridgeVertexIndex].z
        indices = self.polygon.indices
        wallFaceIndices, pv1, pv2 = wallFace[1]
        vertIndex = len(verts)
        v1 = verts[indices[pv1.i]]
        v2 = verts[indices[pv2.i]]
        factorX = (v2.x - v1.x) / (pv2.x - pv1.x)
        factorY = (v2.y - v1.y) / (pv2.x - pv1.x)
        
        # half of the length of the hipped part
        dx = self.widthFactor * abs(pv2.x - pv1.x) / 2.
        # distance from the middle slot
        dx1 = abs(sx - pv1.x)
        dx2 = abs(pv2.x - sx)
        if dx >= dx1:
            dx = 0.9 * dx1
        elif dx >= dx2:
            dx = 0.9 * dx2
        if pv2.x > pv1.x:
            x1 = sx - dx
            x2 = sx + dx
        else:
            x1 = sx + dx
            x2 = sx - dx
        
        z1 = verts[pv1.vertIndex].z
        # Z-coordinate can be calculated also via <z2 = verts[pv2.vertIndex].z>
        # Z-coordinate is the same for the left and for the right vertex because of the symmetry
        # relative to the middle point of the profile along the profile direction
        z = z1 + (sz - z1) * (x1 - pv1.x) / (sx - pv1.x)
        
        def getVertex(x):
            factor = x - pv1.x
            return Vector((
                v1.x + factor * factorX,
                v1.y + factor * factorY,
                z
            ))
        
        verts.append(getVertex(x1))
        verts.append(getVertex(x2))
        wallFaceIndices.pop()
        wallFaceIndices.append(vertIndex+1)
        wallFaceIndices.append(vertIndex)
        
        # Find the roof faces sharing the ridge vertex with index <ridgeVertexIndex>
        # A helper counter used to break the cycle below when everything is found
        counter = 2
        for roofFace in self.roofIndices:
            if roofFace[-1] == ridgeVertexIndex:
                # found the roof face to the left from the ridge vertex with index <ridgeVertexIndex>
                roofFace[-1] = vertIndex
                roofFace.append(ridgeVertexIndex)
                counter -= 1
            elif roofFace[0] == ridgeVertexIndex:
                # found the roof face to the right from the ridge vertex with index <ridgeVertexIndex>
                roofFace[0] = vertIndex + 1
                roofFace.insert(0, ridgeVertexIndex)
                counter -= 1
            if not counter:
                break
        
        # add the hipped face
        self.roofIndices.append((vertIndex, vertIndex + 1, ridgeVertexIndex))
        
        # Relative displacement for the ridge vertices
        # calculated through the tangent of the roof inclination angle
        # We assume that the hipped roof face has the same pitch as the gabled parts
        d = (self.h - z + roofMinHeight) * self.angleToHeight * self.polygonWidth / self.h / ridgeLength
        if d >= 0.5:
            d = 0.45
        
        # add displacement for the ridge vertex
        verts[ridgeVertexIndex] += d * ridgeVector