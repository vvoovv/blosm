from . import ItemRendererTexture
from mathutils import Vector


class RoofFlat(ItemRendererTexture):
    
    def render(self, roofItem):
        face = self.r.createFace(
            roofItem.footprint,
            range(roofItem.firstVertIndex, roofItem.firstVertIndex+roofItem.footprint.polygon.numEdges)
        )
        
        cl = roofItem.getStyleBlockAttr("cl")
        if cl:
            self.renderClass(roofItem, cl, face, None)
        else:
            self.renderCladding(roofItem, face, None)
    
    def setCladdingUvs(self, roofItem, face, claddingTextureInfo, uvs):
        textureWidthM = claddingTextureInfo["textureWidthM"]
        textureHeightM = textureWidthM * claddingTextureInfo["textureSize"][1] / claddingTextureInfo["textureSize"][0]
        
        polygon = roofItem.footprint.polygon
        
        # Arrange the texture along the longest edge of <polygon>,
        # so the longest edges surves as u-axis for the texture
        uVec = polygon.getLongestVector()
        offset = uVec.v1
        uVec = uVec.unitVector
        vVec = Vector( (-uVec[1], uVec[0]) )

        self.r.setUvs(
            face,
            # a generator!
            (
                (
                    (vector.v2-offset).dot(uVec)/textureWidthM,
                    (vector.v2-offset).dot(vVec)/textureHeightM
                ) for vector in polygon.getVectors()
            ),
            roofItem.building.element.l,
            roofItem.building.element.l.uvLayerNameCladding
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