"""
This file is a part of Blosm addon for Blender.
Copyright (C) 2014-2018 Vladimir Elistratov
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

from . import Roof
from .profile import RoofProfile, gabledRoof
from .flat import RoofFlat
from .half_hipped import MiddleSlot


class RoofHipped(RoofProfile):
    """
    The hipped roof shape is implemented only for a quadrangle building outline through the correction of
    the gabled roof created by the parent class. For the other building outlines a flat roof is created.
    """
    
    def __init__(self):
        super().__init__(gabledRoof)
        # replace the middle slot defining the roof ridge
        slots = self.slots
        slots = (slots[0], MiddleSlot(slots[1].x), slots[2])
        slots[1].n = slots[2]
        self.slots = slots
    
    def init(self, element, data, osm, app):
        self.projections.clear()
        Roof.init(self, element, data, osm, app)
        if self.polygon.n == 4:
            self.makeFlat = False
            self.initProfile()
        else:
            self.makeFlat = True
            if self.noWalls:
                self.wallHeight = self.z2 - self.z1
    
    def getRoofHeight(self):
        # this is a hack, but we have to set <self.defaultHeight> here to calculate the roof height correctly
        self.defaultHeight = RoofProfile.defaultHeight if self.polygon.n == 4 else RoofFlat.defaultHeight
        return super().getRoofHeight() if self.polygon.n == 4 else RoofFlat.getRoofHeight(self)
    
    def make(self, osm):
        if self.makeFlat:
            return RoofFlat.make(self, osm)
        else:
            super().make(osm)
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
            # Relative displacement for the ridge vertex
            # calculated through the tangent of the roof inclination angle.
            # We assume that all roof faces have the same pitch.
            d = self.angleToHeight * self.polygonWidth / (slot[-1][0] - slot[0][0])
            if d >= 0.5:
                d = 0.45
            
            # Indices for the ridge vertices; the ridge for the gabled roof was created by the parent class.
            # Remember that parts in a slot are sorted by <y> of each part,
            # the element with the index 1 of each part is the Python list of vertex indices
            indexFront = slot[0][1][0]
            indexBack = slot[-1][1][0]
            # the vector along the ridge
            v = self.verts[indexBack] - self.verts[indexFront]
            
            if front:
                self.makeHipped(front, indexFront, d * v)
            if back:
                self.makeHipped(back, indexBack, -d * v)
        
        return True
    
    def makeHipped(self, wallFace, ridgeVertexIndex, displacement):
        """
        Corrects the gabled roof created by the parent class to create a hipped roof face
        
        Args:
            wallFace (list): Either <self.front> or <self.back>
            ridgeVertexIndex (int): The index of the ridge vertex belonging to either
                the front wall face or the back one
            displacement (Vector): A vector for displacement of the ridge vertex
                with the index <ridgeVertexIndex> to make a hipped roof face
        """
        # vertex indices for a wall face
        wallFaceIndices = wallFace[1][0]
        # the number of vertices in <wallFace>
        numVerts = len(wallFaceIndices)
        if numVerts == 3:
            # 3 vertices for a wall face actually means no wall face, so remove that wall face
            self.wallIndices.remove(wallFaceIndices)
            # create extra triangle for the hipped roof face
            self.roofIndices.append(wallFaceIndices)
        else:
            # Remove the ridge vertex from the wall face.
            if numVerts == 4 and wallFaceIndices[-1] == ridgeVertexIndex:
                # Treat the special case if skip1 == True (see the code in module <.profile>
                # for the definition of <skip1>)
                wallFaceIndices.pop()
                extraTriangle = (wallFaceIndices[0], wallFaceIndices[-1], ridgeVertexIndex)
            else:
                # treat the general case
                closingVertexIndex = wallFaceIndices.pop()
                wallFaceIndices[-1] = closingVertexIndex
                extraTriangle = (wallFaceIndices[-1], wallFaceIndices[-2], ridgeVertexIndex)
            # create extra triangle for the hipped roof face
            self.roofIndices.append(extraTriangle)
        # add displacement for the ridge vertex
        self.verts[ridgeVertexIndex] += displacement