from . import ItemRendererTexture


class RoofHipped(ItemRendererTexture):
        
    def render(self, roofItem):
        building = roofItem.building
        
        for roofSide in roofItem.roofSides:
            face = self.r.createFace(
                building,
                roofSide.indices
            )
            
            cl = roofSide.getStyleBlockAttr("cl")
            if cl:
                self.renderClass(roofSide, cl, face, roofSide.uvs)
            else:
                self.renderCladding(roofSide, face, roofSide.uvs)
    
    def setClassUvs(self, item, face, uvs, texUl, texVb, texUr, texVt):
        # convert generator to Python tuple: <tuple(uvs)>
        self._setRoofClassUvs(face, tuple(uvs), texUl, texVb, texUr, texVt)