from .. import ItemRenderer
from util import zAxis


class RoofFlat(ItemRenderer):
    
    def render(self, roofItem):
        building = roofItem.building
        face = self.r.createFace(building, roofItem.indices)
        
        self.renderCladding(building, roofItem, face, None)
    
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