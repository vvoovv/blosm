from .. import ItemRenderer
from grammar import smoothness


class RoofProfile(ItemRenderer):
        
    def render(self, roofItem, roofGenerator):
        building = roofItem.building
        smoothFaces = roofItem.getStyleBlockAttr("faces") is smoothness.Smooth
        
        for roofSide in roofItem.roofSides:
            face = self.r.createFace(
                building,
                roofSide.indices
            )
            if smoothFaces:
                face.smooth = True
            cl = roofSide.getStyleBlockAttr("cl")
            if cl:
                self.renderClass(roofSide, cl, face, self.getUvs(roofSide, roofGenerator))
            else:
                self.renderCladding(roofSide, face, self.getUvs(roofSide, roofGenerator))
    
    def getUvs(self, roofSide, roofGenerator):
        roofVertexData = roofGenerator.roofVertexData
        slotIndex = roofSide.slotIndex
        slopes = roofGenerator.slopes
        #
        # Set texture coordinates <u> and <v>
        #
        # <roofVertexData[index]> is a Python tuple of three elements:
        # <roofVertexData[index][0]> indicates if the related roof vertex is located
        #     on the slot;
        # <roofVertexData[index][1]> is a slot index if <roofVertexData[index][0]> is equal to True;
        # <roofVertexData[index][1]> is a coordinate along profile part
        #     if <roofVertexData[index][0]> is equal to False;
        # <roofVertexData[index][2]> is a coordinate along Y-axis of the profile
        #     coordinate system
        return (
            (
                # U-coordinate: set it depending on the value of <slopes[slotIndex]>
                roofGenerator.maxY - roofVertexData[index][2]\
                if slopes[slotIndex] else\
                roofVertexData[index][2] - roofGenerator.minY,
                # V-coordinate
                (
                    0.\
                    if (slopes[slotIndex] and roofVertexData[index][1] == slotIndex) or\
                    (not slopes[slotIndex] and roofVertexData[index][1] == slotIndex+1) else\
                    roofGenerator.partLength[slotIndex]
                )
                if roofVertexData[index][0] else\
                # the related roof vertex isn't located on the slot
                roofVertexData[index][1]
            )
            for index in roofSide.indices
        )
        
    def setClassUvs(self, item, face, uvs, texUl, texVb, texUr, texVt):
        # convert generator to Python tuple: <tuple(uvs)>
        self._setRoofClassUvs(face, tuple(uvs), texUl, texVb, texUr, texVt)