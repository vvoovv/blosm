from . import ItemRendererTexture
from grammar import smoothness


class RoofProfile(ItemRendererTexture):
        
    def render(self, roofItem):
        smoothFaces = roofItem.getStyleBlockAttr("faces") is smoothness.Smooth
        
        for roofSide in roofItem.roofSides:
            face = self.r.createFace(
                roofItem.footprint,
                roofSide.indices
            )
            if smoothFaces:
                face.smooth = True
            cl = roofSide.getStyleBlockAttr("cl")
            if cl:
                self.renderClass(roofSide, cl, face, roofSide.uvs)
            else:
                self.renderCladding(roofSide, face, roofSide.uvs)
        
    def setClassUvs(self, item, face, uvs, texUl, texVb, texUr, texVt):
        # convert generator to Python tuple: <tuple(uvs)>
        self._setRoofClassUvs(face, tuple(uvs), texUl, texVb, texUr, texVt)