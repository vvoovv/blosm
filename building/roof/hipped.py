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

from . import Roof
from .profile import RoofProfile
from .flat import RoofFlat
from .half_hipped import MiddleSlot


class RoofHipped(RoofProfile):
    """
    The hipped roof shape is implemented only for a quadrangle building outline through the correction of
    the gabled roof created by the parent class. For the other building outlines a flat roof is created.
    """
    
    def __init__(self, data):
        super().__init__(data)
        # replace the middle slot defining the roof ridge
        slots = self.slots
        slots = (slots[0], MiddleSlot(), slots[2])
        slots[1].n = slots[2]
        self.slots = slots
    
    def init(self, element, data, minHeight, osm):
        Roof.init(self, element, data, minHeight, osm)
        if self.polygon.n == 4:
            self.makeFlat = False
            self.defaultHeight = RoofProfile.defaultHeight
            self.initProfile()
        else:
            self.makeFlat = True
            self.defaultHeight = RoofFlat.defaultHeight
    
    def getHeight(self, op):
        return RoofFlat.getHeight(self, op) if self.makeFlat else super().getHeight(op)
    
    def make(self, bldgMaxHeight, roofMinHeight, bldgMinHeight, osm):
        if self.makeFlat:
            return RoofFlat.make(self, bldgMaxHeight, roofMinHeight, bldgMinHeight, osm)
        else:
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
        if len(wallFaceIndices) == 3:
            # 3 vertices for a wall face actually means no wall face, so remove that wall face
            self.wallIndices.remove(wallFaceIndices)
            # create extra triangle for the hipped roof face
            self.roofIndices.append(wallFaceIndices)
        else:
            # Remove the ridge vertex from the wall face;
            # the following line is equivalent to <wallFaceIndices.remove(ridgeVertexIndex)>
            wallFaceIndices.pop()
            # create extra triangle for the hipped roof face
            self.roofIndices.append( (wallFaceIndices[0], wallFaceIndices[-1], ridgeVertexIndex) )
        # add displacement for the ridge vertex
        self.verts[ridgeVertexIndex] += displacement