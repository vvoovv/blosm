import bmesh
from .. import ItemRenderer
from util import zAxis


def _getEdge(bmVert):
    return (
        bmVert.link_loops[0]\
        if bmVert.co[2] > bmVert.link_loops[1].link_loop_next.vert.co[2] else\
        bmVert.link_loops[1]
    ).edge


class RoofFlatMulti(ItemRenderer):
    
    def render(self, roofItem):
        building = roofItem.building
        # all needed BMesh verts have been already created during facade generation
        bmVerts = building.bmVerts
        
        # create a Python list of BMesh edges for the outer and inner polygons
        indexOffset = roofItem.firstVertIndex
        # treat the outer polygon
        polygon = roofItem.footprint.polygon
        edges = [_getEdge(bmVerts[i]) for i in range(indexOffset, indexOffset + polygon.n) ]
        
        # treat the inner polygons
        indexOffset += polygon.n
        for polygon in roofItem.innerPolygons:
            # skipping the verts for the lower cap
            indexOffset += polygon.n
            edges.extend(_getEdge(bmVerts[i]) for i in range(indexOffset, indexOffset + polygon.n))
            # skipping the verts for the upper cap
            indexOffset += polygon.n
        
        # <bmesh.ops.triangle_fill(..)> a magic function that does everything
        self.renderCladding(
            roofItem,
            tuple(
                face for face in bmesh.ops.triangle_fill(self.r.bm, use_beauty=False, use_dissolve=False, edges=edges)\
                ["geom"] if isinstance(face, bmesh.types.BMFace)
            ),
            None
        )
    
    def setCladdingUvs(self, roofItem, faces, claddingTextureInfo, uvs):
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
        
        for face in faces:
            self.r.setUvs(
                face,
                # a generator!
                (
                    (
                        (vert.co-offset).dot(uVec)/textureWidthM,
                        (vert.co-offset).dot(vVec)/textureHeightM
                    ) for vert in face.verts
                ),
                self.r.layer.uvLayerNameCladding
            )
    
    def setMaterial(self, faces, materialId):
        for face in faces:
            self.r.setMaterial(face, materialId)