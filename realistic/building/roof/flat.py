from mathutils import Vector
from building.roof.flat import RoofFlat


class RoofFlatRealistic(RoofFlat):
    
    def renderWalls(self):
        r = self.r
        bm = r.bm
        verts = self.verts
        materialIndex = r.getWallMaterialIndex(self.element)
        uvLayer = bm.loops.layers.uv[0]
        uvLayerSize = bm.loops.layers.uv[1]
        # create BMesh faces for the building walls
        for f in (bm.faces.new(verts[i] for i in indices) for indices in self.wallIndices):
            w = (f.verts[1].co - f.verts[0].co).length
            h = (f.verts[-1].co - f.verts[0].co).length
            f.loops[0][uvLayer].uv = (0., 0.)
            f.loops[1][uvLayer].uv = (w, 0.)
            f.loops[2][uvLayer].uv = (w, h)
            f.loops[3][uvLayer].uv = (0., h)
            for i in range(4):
                f.loops[i][uvLayerSize].uv = (w, h)
            f.material_index = materialIndex