from .. import ItemRenderer
from util import zAxis


class RoofFlat(ItemRenderer):
    
    def render(self, roofItem):
        building = roofItem.building
        face = self.r.createFace(
            building,
            range(roofItem.firstVertIndex, roofItem.firstVertIndex+roofItem.footprint.polygon.n)
        )
        
        cl = roofItem.getStyleBlockAttr("cl")
        if cl:
            self.renderClass(roofItem, cl, face, None)
        else:
            self.renderCladding(roofItem, face, None)
    
    def setCladdingUvs(self, roofItem, face, claddingTextureInfo, uvs):
        textureWidthM = claddingTextureInfo["textureWidthM"]
        textureHeightM = claddingTextureInfo["textureHeightM"]
        
        polygon = roofItem.footprint.polygon
        verts = polygon.allVerts
        indices = polygon.indices
        
        # Arrange the texture along the longest edge of <polygon>,
        # so the longest edges surves as u-axis for the texture
        maxEdgeIndex = polygon.maxEdgeIndex
        offset = verts[indices[maxEdgeIndex]]
        uVec = (verts[indices[maxEdgeIndex+1]] - offset)
        uVec.normalize()
        vVec = zAxis.cross(uVec)

        self.r.setUvs(
            face,
            # a generator!
            (
                (
                    (verts[indices[i]]-offset).dot(uVec)/textureWidthM,
                    (verts[indices[i]]-offset).dot(vVec)/textureHeightM
                ) for i in range(polygon.n)
            ),
            self.r.layer.uvLayerNameCladding
        )
    
    def setClassUvs(self, item, face, uvs, texUl, texVb, texUr, texVt):
        polygon = item.footprint.polygon
        verts = polygon.allVerts
        indices = polygon.indices
        
        # Arrange the texture along the longest edge of <polygon>,
        # so the longest edges surves as u-axis for the texture
        maxEdgeIndex = polygon.maxEdgeIndex
        uVec = ( verts[indices[maxEdgeIndex+1]] - verts[indices[maxEdgeIndex]] )
        uVec.normalize()
        vVec = zAxis.cross(uVec)
        
        uvs = tuple(
            ( verts[indices[i]].dot(uVec), verts[indices[i]].dot(vVec) )\
            for i in range(polygon.n)
        )
        
        self._setRoofClassUvs(face, uvs, texUl, texVb, texUr, texVt)