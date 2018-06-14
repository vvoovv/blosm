import bmesh
from . import RoofRealistic
from building.roof.flat import RoofFlat, RoofFlatMulti
from util import zAxis
from util.blender import pointNormalUpward


class RoofFlatRealistic(RoofRealistic, RoofFlat):
    
    def renderWalls(self):
        if self.mrw:
            bm = self.r.bm
            verts = self.verts
            uvLayer = bm.loops.layers.uv[0]
            uvLayerSize = bm.loops.layers.uv[1]
            # create BMesh faces for the building walls
            for f in (bm.faces.new(verts[i] for i in indices) for indices in self.wallIndices):
                w = (f.verts[1].co - f.verts[0].co).length
                h = self.wallHeight
                size = (w, h)
                f.loops[0][uvLayer].uv = (0., 0.)
                f.loops[1][uvLayer].uv = (w, 0.)
                f.loops[2][uvLayer].uv = size
                f.loops[3][uvLayer].uv = (0., h)
                for l in f.loops:
                    l[uvLayerSize].uv = size
                self.mrw.renderWalls(f, w)
        else:
            RoofFlat.renderWalls(self)
    
    def renderRoofTextured(self):
        r = self.r
        bm = r.bm
        verts = self.verts
        uvLayer = bm.loops.layers.uv[0]
        # Create BMesh face (exactly one) for the building roof.
        # The code below is simplify to take into account that
        # exactly one face is available.
        # For the general case with multiple faces see the code
        # in RoofFlatMultiRealistic.renderRoofTexturedMulti(..).
        for f in (bm.faces.new(verts[i] for i in indices) for indices in self.roofIndices):
            loops = f.loops
            # Arrange the texture along the first edge,
            # so the first edges surve as u-axis for the texture
            offset = loops[0].vert.co
            uVec = loops[1].vert.co - offset
            uVecLength = uVec.length
            uVec = uVec/uVecLength
            vVec = zAxis.cross(uVec)
            loops[0][uvLayer].uv = (0., 0.)
            loops[1][uvLayer].uv = (uVecLength, 0.)
            for i in range(2, len(loops)):
                vec = loops[i].vert.co - offset
                loops[i][uvLayer].uv = (vec.dot(uVec), vec.dot(vVec))
            self.mrr.renderRoof(f)


class RoofFlatMultiRealistic(RoofRealistic, RoofFlatMulti):
    
    def renderWalls(self):
        """
        The override of the parent class method
        """
        RoofFlatRealistic.renderWalls(self)
    
    def renderRoofTexturedMulti(self, geom):
        if self.mrr:
            uvLayer = self.r.bm.loops.layers.uv[0]
            offset = None
            # check the normal direction of the created faces and assign material to all BMesh faces
            for f in geom["geom"]:
                if isinstance(f, bmesh.types.BMFace):
                    pointNormalUpward(f)
                    loops = f.loops
                    # Arrange the texture along the first edge,
                    # so the first edges surve as u-axis for the texture
                    if offset:
                        startIndex = 0
                    else:
                        offset = loops[0].vert.co
                        uVec = loops[1].vert.co - offset
                        uVecLength = uVec.length
                        uVec = uVec/uVecLength
                        vVec = zAxis.cross(uVec)
                        loops[0][uvLayer].uv = (0., 0.)
                        loops[1][uvLayer].uv = (uVecLength, 0.)
                        startIndex = 2
                    for i in range(startIndex, len(loops)):
                        vec = loops[i].vert.co - offset
                        loops[i][uvLayer].uv = (vec.dot(uVec), vec.dot(vVec))
                    self.mrr.renderRoof(f)
        else:
            RoofFlatMulti.renderRoofTexturedMulti(self, geom)