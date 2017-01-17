from . import Roof
from .profile import RoofProfile, Slot
from .flat import RoofFlat


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

    def processWallFace(self, indices):
        """
        A child class may provide realization of this methods
        Args:
            indices (list): Vertex indices for the wall face
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
            front[1] = indices
        elif noBack or y > back[0]:
            back[0] = y
            back[1] = indices


class RoofHipped(RoofProfile):
    
    def __init__(self, data):
        super().__init__(data)
        # replace the middle slot defining the roof ridge
        slots = self.slots
        slots = (slots[0], MiddleSlot(), slots[2])
        slots[1].n = slots[2]
        self.slots = slots
    
    def init(self, element, minHeight, osm):
        Roof.init(self, element, minHeight, osm)
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
            # Relative displacement for the ridge vertices
            # calculated through the tangent of the roof inclination angle
            d = self.angleToHeight * self.polygonWidth / (slot[-1][0] - slot[0][0])
            if d >= 0.5:
                d = 0.45
            
            # indices for the ridge vertices
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
        if len(wallFace[1]) == 3:
            self.wallIndices.remove(wallFace[1])
            # create extra triangle for the roof
            self.roofIndices.append(wallFace[1])
        else:
            wallFace[1].remove(ridgeVertexIndex)
            # create extra triangle for the roof
            self.roofIndices.append( (wallFace[1][0], wallFace[1][-1], ridgeVertexIndex) )
        # add displacement for the ridge vertex
        self.verts[ridgeVertexIndex] += displacement