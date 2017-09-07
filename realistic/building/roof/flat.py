from . import RoofRealistic
from building.roof.flat import RoofFlat, RoofFlatMulti


class RoofFlatRealistic(RoofRealistic, RoofFlat):
    
    def renderWalls(self):
        if self.mrw:
            self.renderFaces()
        else:
            super().renderWalls()
    
    def renderFaces(self):
        bm = self.r.bm
        verts = self.verts
        uvLayer = bm.loops.layers.uv[0]
        uvLayerSize = bm.loops.layers.uv[1]
        # create BMesh faces for the building walls
        for f in (bm.faces.new(verts[i] for i in indices) for indices in self.wallIndices):
            w = (f.verts[1].co - f.verts[0].co).length
            h = (f.verts[-1].co - f.verts[0].co).length
            size = (w, h)
            f.loops[0][uvLayer].uv = (0., 0.)
            f.loops[1][uvLayer].uv = (w, 0.)
            f.loops[2][uvLayer].uv = size
            f.loops[3][uvLayer].uv = (0., h)
            for i in range(4):
                f.loops[i][uvLayerSize].uv = size
            self.mrw.render(f)
    
    def renderRoofTextured(self):
        r = self.r
        bm = r.bm
        verts = self.verts
        uvLayer = bm.loops.layers.uv[0]
        # create BMesh faces for the building roof
        for f in (bm.faces.new(verts[i] for i in indices) for indices in self.roofIndices):
            loops = f.loops
            offset = loops[0].vert.co
            loops[0][uvLayer].uv = (0., 0.)
            for i in range(1, len(loops)):
                loops[i][uvLayer].uv = (loops[i].vert.co - offset)[:2]
            self.mrr.render(f)


class RoofFlatMultiRealistic(RoofRealistic, RoofFlatMulti):
    
    def renderFaces(self):
        """
        The override of the parent class method
        """
        if self.mrw:
            RoofFlatRealistic.renderFaces(self)
        else:
            super().renderFaces()