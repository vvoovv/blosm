from . import Roof
from .profile import RoofProfile
from .flat import RoofFlat


class RoofHipped(RoofProfile):
    
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
            wallIndices = self.wallIndices
            roofIndices = self.roofIndices
            super().make(bldgMaxHeight, roofMinHeight, bldgMinHeight, osm)
            # the middle slot defining the roof ridge
            slot = self.slots[1].parts
            # Relative displacement for the ridge vertices
            # calculated through the tangent of the roof inclination angle
            d = self.angleToHeight * self.polygonWidth / (slot[-1][0] - slot[0][0])
            if d >= 0.5:
                d = 0.45
            # indices for the ridge vertices
            index1 = slot[0][1][0]
            index2 = slot[-1][1][0]
            # a helper counter to exit the cycle below
            indexCounter = 2
            # remove <index1> and <index2> from the related side in <wallIndices>
            for wall in wallIndices:
                if index1 in wall:
                    wall.remove(index1)
                    indexCounter -= 1
                elif index2 in wall:
                    wall.remove(index2)
                if not indexCounter:
                    break
            # create extra triangle for the roof
            roofIndices.append( (slot[-1][1][-2], slot[0][1][1], index1) )
            roofIndices.append( (slot[0][1][-2], slot[1][1][1], index2) )
            # ridge vertices
            v1 = self.verts[index1]
            v2 = self.verts[index2]
            # the vector along the ridge
            v = v2 - v1
            # displacements for the ridge vertices
            v1 += d * v
            v2 -= d * v
            return True